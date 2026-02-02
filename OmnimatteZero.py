from typing import Optional, List, Tuple, Union, Dict, Callable, Any
from diffusers import LTXConditionPipeline
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents, LTXVideoCondition, \
    linear_quadratic_schedule, retrieve_timesteps
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
import torch
import torch.nn.functional as F
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class OmnimatteZero(LTXConditionPipeline):
    def my_prepare_latents(
            self,
            conditions: Optional[List[torch.Tensor]] = None,
            condition_strength: Optional[List[float]] = None,
            condition_frame_index: Optional[List[int]] = None,
            batch_size: int = 1,
            num_channels_latents: int = 128,
            height: int = 512,
            width: int = 704,
            num_frames: int = 161,
            num_prefix_latent_frames: int = 2,
            sigma: Optional[torch.Tensor] = None,
            latents: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if latents is not None and sigma is not None:
            if latents.shape != shape:
                raise ValueError(
                    f"Latents shape {latents.shape} does not match expected shape {shape}. Please check the input."
                )
            latents = latents.to(device=device, dtype=dtype)
            sigma = sigma.to(device=device, dtype=dtype)
            latents = sigma * noise + (1 - sigma) * latents
        else:
            latents = noise.clone()

        if len(conditions) > 0:
            condition_latent_frames_mask = torch.zeros(
                (batch_size, num_latent_frames), device=device, dtype=torch.float32
            )

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0

            data, mask, strength, frame_index = conditions[0], conditions[1], condition_strength[0], \
            condition_frame_index[0]
            # for data, strength, frame_index in zip(conditions, condition_strength, condition_frame_index):
            condition_latents = retrieve_latents(self.vae.encode(data), generator=generator)
            print(condition_latents.shape)
            condition_latents = self._normalize_latents(
                condition_latents, self.vae.latents_mean, self.vae.latents_std
            ).to(device, dtype=dtype)

            mask[mask < 0] = 0
            mask[mask > 0] = 100
            mask_latents = retrieve_latents(self.vae.encode(mask), generator=generator)
            conditioning_mask_shape = torch.logical_or(mask_latents[0].mean(0) > 0.01,mask_latents[0].mean(0) < -0.01)
            conditioning_mask_shape = conditioning_mask_shape.type(dtype).unsqueeze(0).unsqueeze(0)
            num_cond_frames = condition_latents.size(2)

            latents[:, :, :num_cond_frames] = torch.lerp(
                latents[:, :, :num_cond_frames], condition_latents, strength
            )
            condition_latent_frames_mask[:, :num_cond_frames] = strength

        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )
        if len(conditions) > 0:
            conditioning_mask = (1 - conditioning_mask_shape[0].reshape(1, -1))
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0
        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
            device=device,
        )

        latents = latents * (1 - conditioning_mask_shape) + noise * conditioning_mask_shape

        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        if len(conditions) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat([*extra_conditioning_mask, conditioning_mask], dim=1)

        return latents, conditioning_mask, video_ids, extra_conditioning_num_latents

    @torch.no_grad()
    def my_call(
            self,
            conditions: Union[LTXVideoCondition, List[LTXVideoCondition]] = None,
            image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
            video: List[PipelineImageInput] = None,
            frame_index: Union[int, List[int]] = 0,
            strength: Union[float, List[float]] = 1.0,
            denoise_strength: float = 1.0,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            height: int = 512,
            width: int = 704,
            num_frames: int = 161,
            frame_rate: int = 25,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 3,
            image_cond_noise_scale: float = 0.15,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            prompt_attention_mask: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            decode_timestep: Union[float, List[float]] = 0.0,
            decode_noise_scale: Optional[Union[float, List[float]]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 256,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            conditions=conditions,
            image=image,
            video=video,
            frame_index=frame_index,
            strength=strength,
            denoise_strength=denoise_strength,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]

            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]
        elif image is not None or video is not None:
            if not isinstance(image, list):
                image = [image]
                num_conditions = 1
            elif isinstance(image, list):
                num_conditions = len(image)
            if not isinstance(video, list):
                video = [video]
                num_conditions = 1
            elif isinstance(video, list):
                num_conditions = len(video)

            if not isinstance(frame_index, list):
                frame_index = [frame_index] * num_conditions
            if not isinstance(strength, list):
                strength = [strength] * num_conditions

        device = self._execution_device
        vae_dtype = self.vae.dtype

        # 3. Prepare text embeddings & conditioning image/video
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        conditioning_tensors = []
        is_conditioning_image_or_video = image is not None or video is not None
        if is_conditioning_image_or_video:
            for condition_image, condition_video, condition_frame_index, condition_strength in zip(
                    image, video, frame_index, strength
            ):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width)
                        .unsqueeze(2)
                        .to(device, dtype=vae_dtype)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(condition_video, height, width)
                    num_frames_input = condition_tensor.size(2)
                    num_frames_output = self.trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(device, dtype=vae_dtype)
                else:
                    raise ValueError("Either `image` or `video` must be provided for conditioning.")

                if condition_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form (k * {self.vae_temporal_compression_ratio} + 1) "
                        f"but got {condition_tensor.size(2)} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        # 4. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        sigmas = linear_quadratic_schedule(num_inference_steps)
        timesteps = sigmas * 1000
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        latent_sigma = None
        if denoise_strength < 1:
            sigmas, timesteps, num_inference_steps = self.get_timesteps(
                sigmas, timesteps, num_inference_steps, denoise_strength
            )
            latent_sigma = sigmas[:1].repeat(batch_size * num_videos_per_prompt)

        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self.my_prepare_latents(
            conditioning_tensors,
            strength,
            frame_index,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            sigma=latent_sigma,
            latents=latents,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        video_coords = video_coords.float()
        video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

        init_latents = latents.clone() if is_conditioning_image_or_video else None

        if self.do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if image_cond_noise_scale > 0 and init_latents is not None:
                    # Add timestep-dependent noise to the hard-conditioning latents
                    # This helps with motion continuity, especially when conditioned on a single frame
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        conditioning_mask,
                        generator,
                    )

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if is_conditioning_image_or_video:
                    conditioning_mask_model_input = (
                        torch.cat([conditioning_mask, conditioning_mask])
                        if self.do_classifier_free_guidance
                        else conditioning_mask
                    )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                if is_conditioning_image_or_video:
                    timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    timestep, _ = timestep.chunk(2)

                denoised_latents = self.scheduler.step(
                    -noise_pred, t, latents, per_token_timesteps=timestep, return_dict=False
                )[0]
                if is_conditioning_image_or_video:
                    tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                    latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
                else:
                    latents = denoised_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if is_conditioning_image_or_video:
            latents = latents[:, extra_conditioning_num_latents:]

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if output_type == "latent":
            video = latents
        else:
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                                     :, None, None, None, None
                                     ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
