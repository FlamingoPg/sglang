# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.runtime.models.bridges.sensenova_u1 import (
    predict_u1_pixel_flow_from_srt,
    require_forward_context,
    should_apply_cfg,
    U1PixelFlowForwardContext,
    U1PixelFlowPrepared,
    u1_patchify,
    u1_unpatchify,
)


class SenseNovaU1PixelFlowDenoiser:
    def __init__(
        self,
        model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.model = model
        self.forward_batch_provider = forward_batch_provider

    def forward(self, prepared: U1PixelFlowPrepared) -> Any:
        import torch

        image_prediction = prepared.image_prediction

        for step_i in range(prepared.steps):
            timestep = prepared.timesteps[step_i]
            next_timestep = prepared.timesteps[step_i + 1]
            z = u1_patchify(
                image_prediction,
                prepared.patch_size * prepared.merge_size,
            )
            image_input = u1_patchify(
                image_prediction,
                prepared.patch_size,
                channel_first=True,
            )
            image_embeds = self.model.extract_feature(
                image_input.view(prepared.grid_h * prepared.grid_w, -1),
                gen_model=True,
                grid_hw=prepared.gen_grid_hw,
            ).view(1, prepared.token_h * prepared.token_w, -1)
            timestep_values = timestep.expand(prepared.token_h * prepared.token_w)
            timestep_embeddings = self.model.fm_modules["timestep_embedder"](
                timestep_values
            ).view(1, prepared.token_h * prepared.token_w, -1)
            if getattr(self.model.config, "add_noise_scale_embedding", False):
                noise_values = torch.full_like(
                    timestep_values,
                    prepared.noise_scale
                    / float(self.model.config.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + self.model.fm_modules[
                    "noise_scale_embedder"
                ](noise_values).view(1, prepared.token_h * prepared.token_w, -1)
            image_embeds = image_embeds + timestep_embeddings

            v_condition = self._predict_v(
                forward_context=prepared.condition,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            use_cfg = should_apply_cfg(prepared.cfg, timestep)
            v_pred = self._combine_cfg_velocity(
                prepared=prepared,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
                v_condition=v_condition,
                use_cfg=use_cfg,
            )
            if prepared.cfg.needs_cfg and use_cfg:
                v_pred = self._apply_cfg_renorm(
                    v_condition=v_condition,
                    v_pred=v_pred,
                    cfg_renorm_type=prepared.cfg.renorm_type,
                )

            z = z + (next_timestep - timestep) * v_pred
            image_prediction = u1_unpatchify(
                z,
                prepared.patch_size * prepared.merge_size,
                prepared.height,
                prepared.width,
            )
        return image_prediction

    def _combine_cfg_velocity(
        self,
        *,
        prepared: U1PixelFlowPrepared,
        image_embeds: Any,
        timestep: Any,
        z: Any,
        v_condition: Any,
        use_cfg: bool,
    ) -> Any:
        cfg = prepared.cfg
        if not use_cfg or not cfg.needs_cfg:
            return v_condition
        if cfg.img_scale == 1.0:
            v_img_condition = self._predict_v(
                forward_context=require_forward_context(prepared.img_condition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            return v_img_condition + cfg.text_scale * (v_condition - v_img_condition)
        if cfg.text_scale == cfg.img_scale:
            v_uncondition = self._predict_v(
                forward_context=require_forward_context(prepared.uncondition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            return v_uncondition + cfg.text_scale * (v_condition - v_uncondition)

        v_img_condition = self._predict_v(
            forward_context=require_forward_context(prepared.img_condition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
        )
        v_uncondition = self._predict_v(
            forward_context=require_forward_context(prepared.uncondition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
        )
        return (
            v_uncondition
            + cfg.text_scale * (v_condition - v_img_condition)
            + cfg.img_scale * (v_img_condition - v_uncondition)
        )

    def _predict_v(
        self,
        *,
        forward_context: U1PixelFlowForwardContext,
        image_embeds: Any,
        timestep: Any,
        z: Any,
    ) -> Any:
        forward_batch_context = self.forward_batch_provider(
            prepared=forward_context.prepared,
            g_query_embeds=image_embeds,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
        try:
            return predict_u1_pixel_flow_from_srt(
                self.model,
                image_embeds=image_embeds,
                indexes_image=forward_context.indexes_image,
                forward_batch=forward_batch,
                timestep=timestep,
                z=z,
            )
        finally:
            release = getattr(forward_batch_context, "release", None)
            if callable(release):
                release()

    @staticmethod
    def _apply_cfg_renorm(
        *,
        v_condition: Any,
        v_pred: Any,
        cfg_renorm_type: str,
    ) -> Any:
        if cfg_renorm_type == "none":
            return v_pred
        if cfg_renorm_type == "global":
            norm_v_condition = v_condition.norm(dim=(1, 2), keepdim=True)
            norm_v_cfg = v_pred.norm(dim=(1, 2), keepdim=True)
        elif cfg_renorm_type == "channel":
            norm_v_condition = v_condition.norm(dim=-1, keepdim=True)
            norm_v_cfg = v_pred.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                "Unsupported SenseNova U1 pixel-flow CFG renorm type: "
                f"{cfg_renorm_type}"
            )
        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
        return v_pred * scale
