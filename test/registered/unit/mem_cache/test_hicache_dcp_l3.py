"""CPU coverage for HiCache L3 under decode context parallelism."""

import os
import tempfile
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    StorageKeyNamespace,
    is_mla_storage_writer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.sim_storage import SimHiCacheStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_KEY = "1" * 64
PREFIX_KEY = "2" * 64
SECOND_PAGE_KEY = "3" * 64


def _storage_config(
    *,
    tp_rank: int,
    tp_size: int,
    dcp_rank: int,
    dcp_size: int,
    extra_config: dict | None = None,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="dcp-l3-test",
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        extra_config=extra_config,
    )


class TestStorageKeyNamespace(CustomTestCase):
    def test_dcp1_key_is_unchanged(self):
        key = "0123456789abcdef" * 4
        self.assertEqual(StorageKeyNamespace().scope(key), key)
        self.assertEqual(
            StorageKeyNamespace().scope_tp_shard(key, tp_rank=7, tp_size=8),
            key,
        )

    def test_dcp_key_is_restart_stable(self):
        key = "fedcba9876543210" * 4
        first = StorageKeyNamespace(dcp_rank=5, dcp_size=8).scope(key)
        second = StorageKeyNamespace(dcp_rank=5, dcp_size=8).scope(key)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)

    def test_dcp_rank_and_size_are_isolated(self):
        key = "a" * 64
        scoped = {
            StorageKeyNamespace(dcp_rank=rank, dcp_size=8).scope(key)
            for rank in range(8)
        }
        scoped.add(StorageKeyNamespace(dcp_rank=0, dcp_size=4).scope(key))
        self.assertEqual(len(scoped), 9)
        self.assertNotIn(key, scoped)

    def test_invalid_namespace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank=0, size=0"):
            StorageKeyNamespace(dcp_rank=0, dcp_size=0)
        with self.assertRaisesRegex(ValueError, "rank=8, size=8"):
            StorageKeyNamespace(dcp_rank=8, dcp_size=8)


class TestMLAStorageWriter(CustomTestCase):
    def test_tp8_dcp8_has_eight_writers(self):
        writers = [
            is_mla_storage_writer(
                _storage_config(
                    tp_rank=rank,
                    tp_size=8,
                    dcp_rank=rank,
                    dcp_size=8,
                )
            )
            for rank in range(8)
        ]
        self.assertEqual(writers, [True] * 8)

    def test_tp16_dcp8_suppresses_second_replica_group(self):
        writers = [
            is_mla_storage_writer(
                _storage_config(
                    tp_rank=rank,
                    tp_size=16,
                    dcp_rank=rank % 8,
                    dcp_size=8,
                )
            )
            for rank in range(16)
        ]
        self.assertEqual(writers, [True] * 8 + [False] * 8)

    def test_invalid_mla_topology_is_rejected(self):
        config = _storage_config(
            tp_rank=0,
            tp_size=12,
            dcp_rank=0,
            dcp_size=8,
        )
        with self.assertRaisesRegex(
            ValueError, "tp_size=12 must be divisible by dcp_size=8"
        ):
            is_mla_storage_writer(config)


class TestStorageConfigPlumbing(CustomTestCase):
    def test_runtime_dcp_topology_reaches_storage_config(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.mem_pool_device = object()
        controller.mem_pool_host = SimpleNamespace(layout="layer_first")
        controller.enable_storage_metrics = False
        controller.attn_cp_group = None
        parallel = SimpleNamespace(
            tp_rank=5,
            tp_size=8,
            pp_rank=0,
            pp_size=1,
            attn_dcp_rank=5,
            attn_dcp_size=8,
        )

        with (
            patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            config = controller._generate_storage_config("dcp-l3-test")

        self.assertEqual(config.dcp_rank, 5)
        self.assertEqual(config.dcp_size, 8)


class _RecordingStorage:
    def __init__(self):
        self.calls = []

    def _record(self, name, *args):
        self.calls.append((name, args))

    def batch_exists(self, keys, extra_info=None):
        self._record("batch_exists", keys, extra_info)
        return len(keys)

    def batch_get(self, keys, target_locations):
        self._record("batch_get", keys, target_locations)
        return [torch.full_like(target, 7) for target in target_locations]

    def batch_set(self, keys, values):
        self._record("batch_set", keys, values)
        return True

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self._record("batch_get_v1", keys, host_indices.clone(), extra_info)
        return [True] * len(keys)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self._record("batch_set_v1", keys, host_indices.clone(), extra_info)
        return [True] * len(keys)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        self._record("batch_exists_v2", keys, pool_transfers, extra_info)
        return PoolTransferResult(len(keys), {})

    def batch_get_v2(self, transfers, extra_info=None):
        self._record("batch_get_v2", transfers, extra_info)
        return {
            transfer.name: [True] * len(transfer.keys or []) for transfer in transfers
        }

    def batch_set_v2(self, transfers, extra_info=None):
        self._record("batch_set_v2", transfers, extra_info)
        return {
            transfer.name: [True] * len(transfer.keys or []) for transfer in transfers
        }

    def last(self, name):
        return next(
            args for call_name, args in reversed(self.calls) if call_name == name
        )


class _FakeHostPool:
    def __init__(self, *, page_size=2, dcp_rank=1, dcp_size=4):
        self.page_size = page_size
        self.dcp_rank = dcp_rank
        self.dcp_size = dcp_size
        self.get_offsets = []
        self.set_offsets = []

    def get_storage_indices(self, logical_indices):
        if self.dcp_size == 1:
            return logical_indices
        return logical_indices[self.dcp_rank :: self.dcp_size] // self.dcp_size

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.page_size, dtype=torch.uint8)

    def get_data_page(self, offset, flat=True):
        self.get_offsets.append(int(offset))
        return torch.full((self.page_size,), int(offset) + 3, dtype=torch.uint8)

    def set_from_flat_data_page(self, offset, data):
        self.set_offsets.append((int(offset), data.clone()))


def _base_controller():
    controller = HiCacheController.__new__(HiCacheController)
    controller.page_size = 8
    controller.mem_pool_host = _FakeHostPool()
    controller.storage_host_pool = controller.mem_pool_host
    controller.storage_backend = _RecordingStorage()
    controller.storage_key_namespace = StorageKeyNamespace(dcp_rank=1, dcp_size=4)
    controller.get_hash_str = lambda *args, **kwargs: [PAGE_KEY]
    controller.prefetch_sync_queue = Queue()
    return controller


class TestBaseControllerStorageBoundary(CustomTestCase):
    def test_exists_generic_get_set_scope_keys_and_translate_indices(self):
        controller = _base_controller()
        operation = SimpleNamespace(
            last_hash=None,
            token_ids=list(range(8)),
            prefix_keys=[PREFIX_KEY],
        )
        hashes, hit_tokens = controller._storage_hit_query(operation)
        self.assertEqual(hashes, [PAGE_KEY])
        self.assertEqual(hit_tokens, 8)

        exists_keys, exists_extra = controller.storage_backend.last("batch_exists")
        self.assertEqual(exists_keys, controller._scope_storage_keys([PAGE_KEY]))
        self.assertEqual(
            exists_extra.prefix_keys,
            controller._scope_storage_keys([PREFIX_KEY]),
        )

        backup = SimpleNamespace(
            hash_value=[PAGE_KEY, SECOND_PAGE_KEY],
            host_indices=torch.arange(16),
            prefix_keys=[PREFIX_KEY],
            completed_tokens=0,
        )
        controller.page_set_func = controller._generic_page_set
        controller._page_backup(backup)
        set_keys, _ = controller.storage_backend.last("batch_set")
        self.assertEqual(
            set_keys,
            controller._scope_storage_keys([PAGE_KEY, SECOND_PAGE_KEY]),
        )
        self.assertEqual(controller.mem_pool_host.get_offsets, [0, 2])
        self.assertEqual(backup.completed_tokens, 16)

        prefetch = PrefetchOperation("request", [])
        prefetch.hash_value = [PAGE_KEY, SECOND_PAGE_KEY]
        prefetch.host_indices = torch.arange(16)
        prefetch.prefix_keys = [PREFIX_KEY]
        controller.page_get_func = controller._generic_page_get
        completed_pages = controller._page_transfer(prefetch)
        get_keys, _ = controller.storage_backend.last("batch_get")
        self.assertEqual(
            get_keys,
            controller._scope_storage_keys([PAGE_KEY, SECOND_PAGE_KEY]),
        )
        self.assertEqual(
            [offset for offset, _ in controller.mem_pool_host.set_offsets], [0, 2]
        )
        self.assertEqual(completed_pages * controller.page_size, 16)

    def test_v1_get_set_keep_logical_completion_but_use_physical_rows(self):
        controller = _base_controller()
        backup = SimpleNamespace(
            hash_value=[PAGE_KEY],
            host_indices=torch.arange(8),
            prefix_keys=None,
            completed_tokens=0,
        )
        controller.page_set_func = controller._page_set_zero_copy
        controller._page_backup(backup)
        set_keys, set_indices, _ = controller.storage_backend.last("batch_set_v1")
        self.assertEqual(set_keys, controller._scope_storage_keys([PAGE_KEY]))
        torch.testing.assert_close(set_indices, torch.arange(2))
        self.assertEqual(backup.completed_tokens, 8)

        prefetch = PrefetchOperation("request", [])
        prefetch.hash_value = [PAGE_KEY]
        prefetch.host_indices = torch.arange(8)
        prefetch.prefix_keys = None
        controller.page_get_func = controller._page_get_zero_copy
        completed_pages = controller._page_transfer(prefetch)
        get_keys, get_indices, _ = controller.storage_backend.last("batch_get_v1")
        self.assertEqual(get_keys, controller._scope_storage_keys([PAGE_KEY]))
        torch.testing.assert_close(get_indices, torch.arange(2))
        self.assertEqual(completed_pages * controller.page_size, 8)

    def test_draft_kv_derived_read_scopes_keys_without_index_translation(self):
        # Since the host-pool refactor the DSpark draft is a KV-derived
        # sidecar PoolTransfer consumed by _page_transfer_kv_batch. Its keys
        # must be DCP-scoped like the target's, but its indices must stay
        # logical: the draft KV is replicated, not DCP-sharded.
        controller = _base_controller()
        group = _FakeHostPoolGroup()
        controller.mem_pool_host = group
        controller.storage_host_pool = group.get_pool(PoolName.KV)

        kv_calls = []

        def _recording_kv_get(operation, hashes, host_indices, extra_info=None):
            kv_calls.append((hashes, host_indices.clone()))
            return len(hashes)

        controller.page_get_func = _recording_kv_get

        logical_indices = torch.arange(8)
        draft_transfer = PoolTransfer(
            name=PoolName.DRAFT,
            host_indices=None,
            indices_from_pool=PoolName.KV,
        )
        operation = PrefetchOperation("request", [])
        hit_pages = controller._page_transfer_kv_batch(
            operation,
            [PAGE_KEY],
            logical_indices,
            HiCacheStorageExtraInfo(prefix_keys=None),
            [draft_transfer],
        )
        self.assertEqual(hit_pages, 1)

        kv_hashes, kv_indices = kv_calls[0]
        self.assertEqual(kv_hashes, controller._scope_storage_keys([PAGE_KEY]))
        torch.testing.assert_close(
            kv_indices,
            group.get_pool(PoolName.KV).get_storage_indices(logical_indices),
        )

        transfers, _ = controller.storage_backend.last("batch_get_v2")
        self.assertEqual(transfers[0].name, PoolName.DRAFT)
        self.assertEqual(transfers[0].keys, controller._scope_storage_keys([PAGE_KEY]))
        torch.testing.assert_close(transfers[0].host_indices, logical_indices)


class _FakeHostPoolGroup:
    def __init__(self):
        pools = {
            PoolName.KV: _FakeHostPool(),
            PoolName.MAMBA: _FakeHostPool(page_size=2, dcp_rank=0, dcp_size=1),
            PoolName.DRAFT: _FakeHostPool(page_size=2, dcp_rank=0, dcp_size=1),
        }
        self.entry_map = {
            name: SimpleNamespace(host_pool=pool) for name, pool in pools.items()
        }

    def get_pool(self, name):
        return self.entry_map[name].host_pool

    def get_storage_indices(self, logical_indices):
        return self.get_pool(PoolName.KV).get_storage_indices(logical_indices)


def _hybrid_controller(
    *, tp_rank: int = 1, tp_size: int = 4, dcp_rank: int = 1, dcp_size: int = 4
):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 8
    controller.mem_pool_host = _FakeHostPoolGroup()
    controller.storage_backend = _RecordingStorage()
    controller.storage_config = _storage_config(
        tp_rank=tp_rank,
        tp_size=tp_size,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
    )
    controller.storage_key_namespace = StorageKeyNamespace(
        dcp_rank=dcp_rank, dcp_size=dcp_size
    )
    controller.get_hash_str = lambda *args, **kwargs: [PAGE_KEY]
    controller.backup_skip = False
    controller.storage_backend_type = "file"
    controller.prefetch_sync_queue = Queue()
    return controller


def _hybrid_transfers():
    return [
        PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(2),
            keys=[PAGE_KEY],
        ),
        PoolTransfer(
            name=PoolName.DRAFT,
            host_indices=torch.arange(2),
            keys=[PAGE_KEY],
        ),
    ]


class TestHybridControllerStorageBoundary(CustomTestCase):
    def test_cache_operation_merge_preserves_pool_query_keys(self):
        transfers = [
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(0, 2),
                keys=[PAGE_KEY],
                query_keys=[PREFIX_KEY, PAGE_KEY],
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(2, 4),
                keys=[SECOND_PAGE_KEY],
                query_keys=[PREFIX_KEY, SECOND_PAGE_KEY],
            ),
        ]
        ops = [
            CacheOperation(
                host_indices=torch.arange(i * 2, (i + 1) * 2),
                device_indices=torch.arange(i * 2, (i + 1) * 2),
                node_id=i,
                pool_transfers=[transfer],
            )
            for i, transfer in enumerate(transfers)
        ]

        merged = CacheOperation.merge_ops(ops)

        self.assertEqual(len(merged.pool_transfers), 1)
        self.assertEqual(
            merged.pool_transfers[0].query_keys,
            [PREFIX_KEY, PAGE_KEY, PREFIX_KEY, SECOND_PAGE_KEY],
        )
        self.assertEqual(
            [transfer.query_keys for transfer in transfers],
            [[PREFIX_KEY, PAGE_KEY], [PREFIX_KEY, SECOND_PAGE_KEY]],
        )

    def test_tp16_dcp8_mamba_scope_is_tp_sharded_but_target_scope_is_shared(self):
        rank0 = _hybrid_controller(tp_rank=0, tp_size=16, dcp_rank=0, dcp_size=8)
        rank8 = _hybrid_controller(tp_rank=8, tp_size=16, dcp_rank=0, dcp_size=8)
        restarted_rank8 = _hybrid_controller(
            tp_rank=8, tp_size=16, dcp_rank=0, dcp_size=8
        )

        self.assertEqual(
            rank0._scope_storage_keys([PAGE_KEY]),
            rank8._scope_storage_keys([PAGE_KEY]),
        )

        transfers = [
            PoolTransfer(name=PoolName.MAMBA, keys=[PAGE_KEY]),
            PoolTransfer(name=PoolName.DRAFT, keys=[PAGE_KEY]),
        ]
        rank0_scoped = rank0._scope_pool_transfers(transfers)
        rank8_scoped = rank8._scope_pool_transfers(transfers)
        restarted_rank8_scoped = restarted_rank8._scope_pool_transfers(transfers)

        self.assertNotEqual(rank0_scoped[0].keys, rank8_scoped[0].keys)
        self.assertEqual(rank8_scoped[0].keys, restarted_rank8_scoped[0].keys)
        self.assertEqual(rank0_scoped[1].keys, rank8_scoped[1].keys)
        self.assertEqual([transfer.keys for transfer in transfers], [[PAGE_KEY]] * 2)

    def test_exists_transfers_carry_pool_scoped_candidate_prefix(self):
        controller = _hybrid_controller(tp_rank=8, tp_size=16, dcp_rank=0, dcp_size=8)
        transfers = _hybrid_transfers()
        query_keys = [PAGE_KEY, SECOND_PAGE_KEY]

        scoped = controller._scope_pool_query_transfers(transfers, query_keys)

        self.assertEqual(
            scoped[0].query_keys,
            controller._scope_pool_keys(PoolName.MAMBA, query_keys),
        )
        self.assertEqual(
            scoped[1].query_keys,
            controller._scope_storage_keys(query_keys),
        )
        self.assertTrue(all(transfer.query_keys is None for transfer in transfers))

    def test_storage_transfer_copies_scope_keys_and_normalize_per_pool(self):
        controller = _hybrid_controller()
        original = [
            PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.arange(8),
                keys=[PAGE_KEY],
            ),
            *_hybrid_transfers(),
        ]
        storage = controller._storage_pool_transfers(original)

        self.assertTrue(all(a is not b for a, b in zip(original, storage)))
        self.assertEqual([t.keys for t in original], [[PAGE_KEY]] * 3)
        self.assertEqual(
            [t.keys for t in storage],
            [
                controller._scope_storage_keys([PAGE_KEY]),
                controller._scope_pool_keys(PoolName.MAMBA, [PAGE_KEY]),
                controller._scope_storage_keys([PAGE_KEY]),
            ],
        )
        torch.testing.assert_close(storage[0].host_indices, torch.arange(2))
        torch.testing.assert_close(storage[1].host_indices, torch.arange(2))
        torch.testing.assert_close(storage[2].host_indices, torch.arange(2))

    def test_v2_exists_get_set_receive_scoped_transfer_copies(self):
        controller = _hybrid_controller()
        exists_transfers = _hybrid_transfers()
        query = SimpleNamespace(
            token_ids=list(range(8)),
            last_hash=None,
            prefix_keys=[PREFIX_KEY],
            pool_transfers=exists_transfers,
            pool_storage_result=PoolTransferResult.empty(),
        )
        hashes, hit_tokens = controller._storage_hit_query(query)
        self.assertEqual(hashes, [PAGE_KEY])
        self.assertEqual(hit_tokens, 8)
        exists_keys, scoped_exists, exists_extra = controller.storage_backend.last(
            "batch_exists_v2"
        )
        self.assertEqual(exists_keys, controller._scope_storage_keys([PAGE_KEY]))
        self.assertEqual(
            [t.keys for t in scoped_exists],
            [
                controller._scope_pool_keys(PoolName.MAMBA, [PAGE_KEY]),
                controller._scope_storage_keys([PAGE_KEY]),
            ],
        )
        self.assertEqual(
            [t.query_keys for t in scoped_exists],
            [
                controller._scope_pool_keys(PoolName.MAMBA, [PAGE_KEY]),
                controller._scope_storage_keys([PAGE_KEY]),
            ],
        )
        self.assertEqual(
            exists_extra.prefix_keys,
            controller._scope_storage_keys([PREFIX_KEY]),
        )
        self.assertEqual([t.keys for t in exists_transfers], [[PAGE_KEY]] * 2)

        get_transfers = _hybrid_transfers()
        prefetch = HybridPrefetchOperation("request", [], pool_transfers=get_transfers)
        prefetch.hash_value = [PAGE_KEY]
        prefetch.host_indices = torch.arange(8)
        with patch.object(
            HiCacheController,
            "_page_transfer",
            autospec=True,
            side_effect=lambda current, op: len(op.hash_value),
        ):
            controller._page_transfer(prefetch)
        scoped_get, _ = controller.storage_backend.last("batch_get_v2")
        self.assertEqual(
            [t.keys for t in scoped_get],
            [
                controller._scope_pool_keys(PoolName.MAMBA, [PAGE_KEY]),
                controller._scope_storage_keys([PAGE_KEY]),
            ],
        )
        self.assertEqual([t.keys for t in get_transfers], [[PAGE_KEY]] * 2)

        set_transfers = _hybrid_transfers()
        backup = HybridStorageOperation(
            torch.arange(8),
            [],
            hash_value=[PAGE_KEY],
            pool_transfers=set_transfers,
        )
        with patch.object(HiCacheController, "_page_backup", autospec=True):
            controller._page_backup(backup)
        scoped_set, _ = controller.storage_backend.last("batch_set_v2")
        self.assertEqual(
            [t.keys for t in scoped_set],
            [
                controller._scope_pool_keys(PoolName.MAMBA, [PAGE_KEY]),
                controller._scope_storage_keys([PAGE_KEY]),
            ],
        )
        self.assertEqual([t.keys for t in set_transfers], [[PAGE_KEY]] * 2)


class TestDCP8FileRoundTrip(CustomTestCase):
    dcp_size = 8
    physical_page_size = 2
    logical_page_size = physical_page_size * dcp_size

    @staticmethod
    def _device_pool() -> SimpleNamespace:
        size = 4
        kv_cache_dim = 6
        return SimpleNamespace(
            size=size,
            store_dtype=torch.float16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            layer_num=2,
            start_layer=0,
            end_layer=1,
            device="cpu",
            layers_to_capture=None,
            layer_shard_enabled=False,
            kv_buffer=[
                torch.zeros(size, 1, kv_cache_dim, dtype=torch.float16)
                for _ in range(2)
            ],
        )

    def _build_rank(self, directory: str, rank: int):
        device_pool = self._device_pool()
        host_pool = MLATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=self.logical_page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            dcp_size=self.dcp_size,
            dcp_rank=rank,
        )
        backend = HiCacheFile(
            _storage_config(
                tp_rank=rank,
                tp_size=self.dcp_size,
                dcp_rank=rank,
                dcp_size=self.dcp_size,
                extra_config={"max_size": "0", "min_free_space": "0"},
            ),
            file_path=directory,
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = self.logical_page_size
        controller.mem_pool_device = device_pool
        controller.mem_pool_host = host_pool
        controller.storage_host_pool = host_pool
        controller.storage_backend = backend
        controller.storage_key_namespace = StorageKeyNamespace(
            dcp_rank=rank, dcp_size=self.dcp_size
        )
        controller.page_get_func = controller._generic_page_get
        controller.page_set_func = controller._generic_page_set
        controller.get_hash_str = lambda *args, **kwargs: [PAGE_KEY]
        controller.prefetch_sync_queue = Queue()
        return controller, host_pool, backend

    def _logical_page(self) -> torch.Tensor:
        return torch.arange(self.logical_page_size, 2 * self.logical_page_size)

    @staticmethod
    def _physical_page(host_pool: MLATokenToKVPoolHost) -> torch.Tensor:
        return host_pool.kv_buffer[
            :, host_pool.page_size : 2 * host_pool.page_size, :, :
        ]

    @staticmethod
    def _assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

    def _assert_l2_load_sees_restored_rows(
        self,
        host_pool: MLATokenToKVPoolHost,
        expected: torch.Tensor,
    ) -> None:
        logical_page = self._logical_page()
        with patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_per_layer_mla",
            create=True,
        ) as transfer:
            host_pool.can_use_jit = False
            host_pool.load_to_device_per_layer(
                host_pool.device_pool,
                logical_page,
                logical_page.clone(),
                layer_id=0,
                io_backend="kernel",
            )

        kwargs = transfer.call_args.kwargs
        physical_rows = torch.arange(
            self.physical_page_size, 2 * self.physical_page_size
        )
        torch.testing.assert_close(kwargs["src_indices"], physical_rows)
        torch.testing.assert_close(kwargs["dst_indices"], physical_rows)
        self._assert_bytes_equal(
            kwargs["src"].index_select(0, physical_rows),
            expected[0],
        )

    def _load_and_verify(self, ranks, expected_pages) -> None:
        logical_page = self._logical_page()
        for rank, ((controller, host_pool, _), expected) in enumerate(
            zip(ranks, expected_pages)
        ):
            host_pool.kv_buffer.zero_()
            query = SimpleNamespace(
                last_hash=None,
                token_ids=list(range(self.logical_page_size)),
                prefix_keys=None,
            )
            hashes, hit_tokens = controller._storage_hit_query(query)
            self.assertEqual(hashes, [PAGE_KEY])
            self.assertEqual(hit_tokens, self.logical_page_size)

            prefetch = PrefetchOperation(f"rank-{rank}", [])
            prefetch.hash_value = [PAGE_KEY]
            prefetch.host_indices = logical_page
            prefetch.prefix_keys = None
            completed_pages = controller._page_transfer(prefetch)
            self.assertEqual(
                completed_pages * controller.page_size, self.logical_page_size
            )
            self._assert_bytes_equal(self._physical_page(host_pool), expected)
            self._assert_l2_load_sees_restored_rows(host_pool, expected)

    def test_eight_rank_file_round_trip_survives_restart(self):
        logical_page = self._logical_page()
        with tempfile.TemporaryDirectory(prefix="hicache_dcp8_l3_") as directory:
            first_generation = [
                self._build_rank(directory, rank) for rank in range(self.dcp_size)
            ]
            expected_pages = []
            expected_files = set()

            for rank, (controller, host_pool, backend) in enumerate(first_generation):
                page = self._physical_page(host_pool)
                pattern = (
                    torch.arange(page.numel(), dtype=torch.int64)
                    .add(rank * 100)
                    .to(host_pool.dtype)
                    .reshape(page.shape)
                )
                page.copy_(pattern)
                expected_pages.append(pattern.clone())

                backup = SimpleNamespace(
                    hash_value=[PAGE_KEY],
                    host_indices=logical_page,
                    prefix_keys=None,
                    completed_tokens=0,
                )
                controller._page_backup(backup)
                self.assertEqual(backup.completed_tokens, self.logical_page_size)

                scoped_key = controller.storage_key_namespace.scope(PAGE_KEY)
                expected_files.add(f"{backend._get_suffixed_key(scoped_key)}.bin")

            actual_files = {
                name for name in os.listdir(directory) if name.endswith(".bin")
            }
            self.assertEqual(actual_files, expected_files)
            self.assertEqual(len(actual_files), self.dcp_size)

            self._load_and_verify(first_generation, expected_pages)

            restarted = [
                self._build_rank(directory, rank) for rank in range(self.dcp_size)
            ]
            restarted_files = {
                f"{backend._get_suffixed_key(controller.storage_key_namespace.scope(PAGE_KEY))}.bin"
                for controller, _, backend in restarted
            }
            self.assertEqual(restarted_files, expected_files)
            self._load_and_verify(restarted, expected_pages)


class _BytePageHostPool:
    def __init__(self, values: list[int]):
        self.page_size = len(values)
        self.size_per_token = 1
        self.data = torch.tensor(values, dtype=torch.uint8)

    def get_data_page(self, page_offset: int, flat: bool = True) -> torch.Tensor:
        return self.data[page_offset : page_offset + self.page_size]

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self.page_size, dtype=torch.uint8)

    def set_from_flat_data_page(self, page_offset: int, data: torch.Tensor) -> None:
        self.data[page_offset : page_offset + self.page_size].copy_(data)


@pytest.mark.parametrize("max_size", ["0", "1Ki"])
def test_tp16_dcp8_file_mamba_sidecars_round_trip_per_tp_rank(max_size: str):
    with tempfile.TemporaryDirectory(prefix="hicache_dcp8_mamba_") as directory:
        configs = [
            _storage_config(
                tp_rank=tp_rank,
                tp_size=16,
                dcp_rank=0,
                dcp_size=8,
                extra_config={
                    "max_size": max_size,
                    "min_free_space": "0",
                    "enable_metadata_cache": True,
                    "metadata_ttl": -1,
                },
            )
            for tp_rank in (0, 8)
        ]
        controllers = [
            _hybrid_controller(
                tp_rank=config.tp_rank,
                tp_size=config.tp_size,
                dcp_rank=config.dcp_rank,
                dcp_size=config.dcp_size,
            )
            for config in configs
        ]
        target_keys = [
            controller._scope_storage_keys([PAGE_KEY])[0] for controller in controllers
        ]
        mamba_keys = [
            controller._scope_pool_transfers(
                [PoolTransfer(name=PoolName.MAMBA, keys=[PAGE_KEY])]
            )[0].keys[0]
            for controller in controllers
        ]

        assert target_keys[0] == target_keys[1]
        assert mamba_keys[0] != mamba_keys[1]

        source_values = ([10, 11], [80, 81])
        backends = [HiCacheFile(config, file_path=directory) for config in configs]
        source_pools = [_BytePageHostPool(values) for values in source_values]
        for backend, pool, key in zip(backends, source_pools, mamba_keys):
            backend.register_mem_host_pool_v2(pool, PoolName.MAMBA)
            result = backend.batch_set_v2(
                [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=torch.arange(pool.page_size),
                        keys=[key],
                    )
                ]
            )
            assert result[PoolName.MAMBA] == [True]

        assert backends[0]._evictor.is_storage_owner
        assert not backends[1]._evictor.is_storage_owner
        assert all(backend._tp_sharded_evictor.is_storage_owner for backend in backends)

        expected_files = {
            f"{backend._get_component_key(key, PoolName.MAMBA)}.bin"
            for backend, key in zip(backends, mamba_keys)
        }
        assert len(expected_files) == 2
        assert {name for name in os.listdir(directory) if name.endswith(".bin")} == (
            expected_files
        )

        assert backends[0].set(target_keys[0], torch.tensor([1, 2], dtype=torch.uint8))
        for backend, controller, target_key in zip(backends, controllers, target_keys):
            query_transfers = controller._scope_pool_query_transfers(
                [PoolTransfer(name=PoolName.MAMBA, keys=[PAGE_KEY])],
                [PAGE_KEY],
            )
            hit = backend.batch_exists_v2([target_key], query_transfers)
            assert hit.kv_hit_pages == 1
            assert hit.extra_pool_hit_pages[PoolName.MAMBA] == 1

        restarted = [HiCacheFile(config, file_path=directory) for config in configs]
        restored_pools = [_BytePageHostPool([0, 0]) for _ in configs]
        for backend, pool, key, expected_values in zip(
            restarted,
            restored_pools,
            mamba_keys,
            source_values,
        ):
            backend.register_mem_host_pool_v2(pool, PoolName.MAMBA)
            component_key = backend._get_component_key(key, PoolName.MAMBA)
            assert f"{component_key}.bin" in expected_files
            assert backend.metadata_cache.contains(component_key)
            if backend._tp_sharded_evictor.enabled:
                assert component_key in backend._tp_sharded_evictor._lru
            result = backend.batch_get_v2(
                [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=torch.arange(pool.page_size),
                        keys=[key],
                    )
                ]
            )
            assert result[PoolName.MAMBA] == [True]
            torch.testing.assert_close(
                pool.data,
                torch.tensor(expected_values, dtype=torch.uint8),
                rtol=0,
                atol=0,
            )

        restarted[0].clear()
        assert not restarted[0]._evictor._lru
        assert not restarted[0]._tp_sharded_evictor._lru


def test_sim_storage_exists_uses_tp_scoped_sidecar_query_keys():
    config = _storage_config(
        tp_rank=8,
        tp_size=16,
        dcp_rank=0,
        dcp_size=8,
        extra_config={
            "sim_write_gbps": 0,
            "sim_read_gbps": 0,
            "sim_op_latency_us": 0,
        },
    )
    controller = _hybrid_controller(tp_rank=8, tp_size=16, dcp_rank=0, dcp_size=8)
    backend = SimHiCacheStorage(config)
    pool = _BytePageHostPool([1, 2])
    backend.register_mem_host_pool_v2(pool, PoolName.MAMBA)

    target_key = controller._scope_storage_keys([PAGE_KEY])[0]
    transfer = controller._scope_pool_transfers(
        [
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(pool.page_size),
                keys=[PAGE_KEY],
            )
        ]
    )[0]
    backend.set(target_key, torch.empty(0))
    assert backend.batch_set_v2([transfer])[PoolName.MAMBA] == [True]

    query_transfer = controller._scope_pool_query_transfers(
        [PoolTransfer(name=PoolName.MAMBA, keys=[PAGE_KEY])], [PAGE_KEY]
    )
    result = backend.batch_exists_v2([target_key], query_transfer)
    assert result.kv_hit_pages == 1
    assert result.extra_pool_hit_pages[PoolName.MAMBA] == 1


def test_file_mamba_tp_scope_preserves_trailing_page_hit_semantics():
    with tempfile.TemporaryDirectory(
        prefix="hicache_dcp8_mamba_trailing_"
    ) as directory:
        rank8_config = _storage_config(
            tp_rank=8,
            tp_size=16,
            dcp_rank=0,
            dcp_size=8,
            extra_config={"max_size": "1Ki", "min_free_space": "0"},
        )
        controller = _hybrid_controller(tp_rank=8, tp_size=16, dcp_rank=0, dcp_size=8)
        target_writer = HiCacheFile(
            _storage_config(
                tp_rank=0,
                tp_size=16,
                dcp_rank=0,
                dcp_size=8,
                extra_config={"max_size": "1Ki", "min_free_space": "0"},
            ),
            file_path=directory,
        )
        backend = HiCacheFile(rank8_config, file_path=directory)
        canonical_keys = [PAGE_KEY, SECOND_PAGE_KEY]
        target_keys = controller._scope_storage_keys(canonical_keys)
        for key in target_keys:
            assert target_writer.set(key, torch.tensor([1], dtype=torch.uint8))

        mamba_tail_key = controller._scope_pool_keys(PoolName.MAMBA, [SECOND_PAGE_KEY])[
            0
        ]
        assert backend._set_value(
            mamba_tail_key,
            torch.tensor([2], dtype=torch.uint8),
            component_name=PoolName.MAMBA,
        )
        transfers = controller._scope_pool_query_transfers(
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=[SECOND_PAGE_KEY],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
            canonical_keys,
        )

        result = backend.batch_exists_v2(target_keys, transfers)

        assert result.kv_hit_pages == 2
        assert result.extra_pool_hit_pages[PoolName.MAMBA] == 2


if __name__ == "__main__":
    unittest.main()
