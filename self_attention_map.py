# Self-attention map extraction using one noising-denoising step with hooks
# Generates total_mask.mp4 from video + object_mask by finding attention-based effects

from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image
import numpy as np
import os


class AttentionMapExtractor:
    def __init__(self, model, layer_indices=None, attention_type="self",
                 store_queries_keys=False, average_over_heads=True):
        self.model = model
        self.layer_indices = layer_indices
        self.attention_type = attention_type
        self.store_queries_keys = store_queries_keys
        self.average_over_heads = average_over_heads

        self._hooks = []
        self._attention_maps = {}
        self._queries = {}
        self._keys = {}
        self._values = {}
        self._is_extracting = False

    def _find_attention_modules(self):
        attention_modules = []
        layer_idx = 0

        for name, module in self.model.named_modules():
            is_attention = any([
                'attn' in name.lower(),
                'attention' in name.lower(),
                isinstance(module, nn.MultiheadAttention),
            ])

            is_self_attn = 'self' in name.lower() or 'attn1' in name.lower()
            is_cross_attn = 'cross' in name.lower() or 'attn2' in name.lower()

            if is_attention:
                if self.attention_type == "self" and not is_self_attn and is_cross_attn:
                    continue
                if self.attention_type == "cross" and not is_cross_attn and is_self_attn:
                    continue

                if self.layer_indices is not None and layer_idx not in self.layer_indices:
                    layer_idx += 1
                    continue

                attention_modules.append((name, module, layer_idx))
                layer_idx += 1

        return attention_modules

    def _create_hook(self, layer_idx, module_name):
        def hook(module, inputs, output):
            if not self._is_extracting:
                return

            attention_weights = None

            if isinstance(output, tuple) and len(output) >= 2:
                if isinstance(output[1], torch.Tensor):
                    attention_weights = output[1]

            if attention_weights is None and hasattr(output, 'attention_probs'):
                attention_weights = output.attention_probs

            if attention_weights is None and hasattr(module, '_attention_weights'):
                attention_weights = module._attention_weights

            if attention_weights is None:
                attention_weights = self._compute_attention_from_qk(module, inputs)

            if attention_weights is not None:
                if self.average_over_heads and attention_weights.dim() == 4:
                    attention_weights = attention_weights.mean(dim=1)

                self._attention_maps[layer_idx] = attention_weights.detach().cpu()

                if self.store_queries_keys:
                    self._extract_qkv(module, inputs, layer_idx)

        return hook

    def _compute_attention_from_qk(self, module, inputs):
        try:
            if hasattr(module, 'to_q') and hasattr(module, 'to_k'):
                hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs

                q = module.to_q(hidden_states)
                k = module.to_k(hidden_states)

                if hasattr(module, 'heads'):
                    batch_size, seq_len, _ = q.shape
                    head_dim = q.shape[-1] // module.heads
                    q = q.view(batch_size, seq_len, module.heads, head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_len, module.heads, head_dim).transpose(1, 2)

                scale = q.shape[-1] ** -0.5
                attention_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attention_weights = F.softmax(attention_weights, dim=-1)

                return attention_weights
        except:
            pass
        return None

    def _extract_qkv(self, module, inputs, layer_idx):
        try:
            hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
            if hasattr(module, 'to_q'):
                self._queries[layer_idx] = module.to_q(hidden_states).detach().cpu()
            if hasattr(module, 'to_k'):
                self._keys[layer_idx] = module.to_k(hidden_states).detach().cpu()
            if hasattr(module, 'to_v'):
                self._values[layer_idx] = module.to_v(hidden_states).detach().cpu()
        except:
            pass

    def register_hooks(self):
        self.remove_hooks()
        attention_modules = self._find_attention_modules()

        for name, module, layer_idx in attention_modules:
            hook = self._create_hook(layer_idx, name)
            handle = module.register_forward_hook(hook)
            self._hooks.append(handle)

        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def clear_maps(self):
        self._attention_maps = {}
        self._queries = {}
        self._keys = {}
        self._values = {}

    @contextmanager
    def extraction_context(self):
        self._is_extracting = True
        self.clear_maps()
        try:
            yield
        finally:
            self._is_extracting = False

    def get_attention_maps(self):
        return self._attention_maps.copy()

    def get_queries_keys_values(self):
        return self._queries.copy(), self._keys.copy(), self._values.copy()


class SelfAttentionMapExtraction:
    def __init__(self, pipeline, extraction_timestep=0.5, noise_level=0.5):
        self.pipeline = pipeline
        self.extraction_timestep = extraction_timestep
        self.noise_level = noise_level
        self.extractor = None

    def setup_extractor(self, layer_indices=None, store_queries_keys=False):
        transformer = self.pipeline.transformer
        self.extractor = AttentionMapExtractor(
            model=transformer, layer_indices=layer_indices,
            attention_type="self", store_queries_keys=store_queries_keys, average_over_heads=True)
        self.extractor.register_hooks()

    def cleanup(self):
        if self.extractor is not None:
            self.extractor.remove_hooks()
            self.extractor = None

    @torch.no_grad()
    def extract_from_video(self, video, prompt="", height=None, width=None, generator=None):
        if self.extractor is None:
            self.setup_extractor()

        device = self.pipeline._execution_device
        dtype = self.pipeline.transformer.dtype

        if not isinstance(video, torch.Tensor):
            video = self.pipeline.video_processor.preprocess_video(video, height, width)

        video = video.to(device=device, dtype=dtype)
        B, C, T, H, W = video.shape

        if height is None:
            height = H
        if width is None:
            width = W

        latents = self._encode_video(video, generator)

        noise = torch.randn_like(latents, generator=generator)
        timestep = torch.tensor([self.extraction_timestep * 1000], device=device)

        noisy_latents = self.pipeline.scheduler.add_noise(latents, noise, timestep.long())

        prompt_embeds, prompt_attention_mask, _, _ = self.pipeline.encode_prompt(
            prompt=prompt, negative_prompt=None, do_classifier_free_guidance=False,
            num_videos_per_prompt=1, device=device)

        num_latent_frames = latents.shape[2]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]

        video_coords = self.pipeline._prepare_video_ids(
            batch_size=B, num_latent_frames=num_latent_frames,
            latent_height=latent_height, latent_width=latent_width,
            patch_size_t=self.pipeline.transformer_temporal_patch_size,
            patch_size=self.pipeline.transformer_spatial_patch_size, device=device)
        video_coords = self.pipeline._scale_video_ids(
            video_coords, scale_factor=self.pipeline.vae_spatial_compression_ratio,
            scale_factor_t=self.pipeline.vae_temporal_compression_ratio, frame_index=0, device=device)

        packed_latents = self.pipeline._pack_latents(
            noisy_latents, self.pipeline.transformer_spatial_patch_size,
            self.pipeline.transformer_temporal_patch_size)

        with self.extractor.extraction_context():
            _ = self.pipeline.transformer(
                hidden_states=packed_latents.to(prompt_embeds.dtype),
                encoder_hidden_states=prompt_embeds,
                timestep=timestep.expand(B, -1).float(),
                encoder_attention_mask=prompt_attention_mask,
                video_coords=video_coords.float(),
                return_dict=False)

        attention_maps = self.extractor.get_attention_maps()
        return attention_maps, (num_latent_frames, latent_height, latent_width)

    def _encode_video(self, video, generator=None):
        latents = retrieve_latents(self.pipeline.vae.encode(video), generator=generator)
        latents = self.pipeline._normalize_latents(
            latents, self.pipeline.vae.latents_mean, self.pipeline.vae.latents_std)
        return latents

    @torch.no_grad()
    def extract_effects_mask(self, video, object_mask, height=512, width=768,
                             threshold=0.15, dilation_size=5, generator=None):
        """
        Extract effects (shadows, reflections) from self-attention maps
        video: input video (path or tensor)
        object_mask: binary mask of the object (path or tensor)
        Returns: effects_mask tensor
        """
        device = self.pipeline._execution_device
        dtype = self.pipeline.transformer.dtype

        # load video
        if isinstance(video, str):
            video = load_video(video)
        if not isinstance(video, torch.Tensor):
            video_tensor = self.pipeline.video_processor.preprocess_video(video, height, width)
        else:
            video_tensor = video
        video_tensor = video_tensor.to(device=device, dtype=dtype)

        # load object mask
        if isinstance(object_mask, str):
            object_mask = load_video(object_mask)
        if not isinstance(object_mask, torch.Tensor):
            mask_tensor = self.pipeline.video_processor.preprocess_video(object_mask, height, width)
        else:
            mask_tensor = object_mask
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        # binarize mask
        mask_binary = (mask_tensor.mean(dim=1, keepdim=True) > 0).float()

        B, C, T, H, W = video_tensor.shape

        # extract attention maps
        attention_maps, latent_dims = self.extract_from_video(video_tensor, height=H, width=W, generator=generator)
        num_latent_frames, latent_height, latent_width = latent_dims

        # downsample mask to latent space
        mask_latent = F.interpolate(
            mask_binary.squeeze(2) if mask_binary.dim() == 5 else mask_binary,
            size=(latent_height, latent_width), mode='nearest'
        )
        if mask_binary.dim() == 5:
            # handle video mask - average across frames for now
            mask_latent = F.interpolate(
                mask_binary[:, 0, :, :, :], size=(latent_height, latent_width), mode='nearest')

        mask_flat = mask_latent.flatten(1)  # (B, H*W)

        # aggregate attention from object regions
        effects_map = torch.zeros(B, latent_height * latent_width, device=device)

        for layer_idx, attn_map in attention_maps.items():
            attn_map = attn_map.to(device)
            if attn_map.dim() == 3:
                # (B, Q, K) - compute attention from object to other regions
                object_attention = attn_map * mask_flat.unsqueeze(1)  # attention to object
                effects_contribution = object_attention.sum(dim=-1)  # how much each query attends to object
                effects_map += effects_contribution
            elif attn_map.dim() == 4:
                # (B, heads, Q, K)
                attn_map = attn_map.mean(dim=1)
                object_attention = attn_map * mask_flat.unsqueeze(1)
                effects_contribution = object_attention.sum(dim=-1)
                effects_map += effects_contribution

        # normalize
        effects_map = effects_map / (len(attention_maps) + 1e-8)
        effects_map = effects_map.view(B, 1, latent_height, latent_width)

        # upsample back to original resolution
        effects_mask = F.interpolate(effects_map, size=(H, W), mode='bilinear', align_corners=False)

        # threshold and binarize
        effects_mask = (effects_mask > threshold).float()

        # dilate to cover more area
        if dilation_size > 0:
            kernel = torch.ones(1, 1, dilation_size, dilation_size, device=device)
            effects_mask = F.conv2d(effects_mask, kernel, padding=dilation_size // 2)
            effects_mask = (effects_mask > 0).float()

        # expand to video frames
        effects_mask = effects_mask.unsqueeze(2).expand(-1, -1, T, -1, -1)

        return effects_mask

    @torch.no_grad()
    def generate_total_mask(self, video_path, object_mask_path, output_path,
                            height=512, width=768, threshold=0.15, dilation_size=15, fps=24):
        """
        Generate total_mask.mp4 combining object_mask with attention-based effects
        """
        device = self.pipeline._execution_device
        dtype = self.pipeline.transformer.dtype

        print(f"Loading video from {video_path}")
        video = load_video(video_path)
        print(f"Loading object mask from {object_mask_path}")
        object_mask = load_video(object_mask_path)

        video_tensor = self.pipeline.video_processor.preprocess_video(video, height, width)
        video_tensor = video_tensor.to(device=device, dtype=dtype)

        mask_tensor = self.pipeline.video_processor.preprocess_video(object_mask, height, width)
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        B, C, T, H, W = video_tensor.shape

        # binarize object mask
        object_mask_binary = (mask_tensor.mean(dim=1, keepdim=True) > 0).float()

        print("Extracting self-attention maps...")
        effects_mask = self.extract_effects_mask(
            video_tensor, object_mask_binary, height=H, width=W,
            threshold=threshold, dilation_size=dilation_size)

        # combine object mask + effects mask
        total_mask = torch.clamp(object_mask_binary + effects_mask, 0, 1)

        # convert to video frames
        total_mask_rgb = total_mask.expand(-1, 3, -1, -1, -1)  # (B, 3, T, H, W)
        total_mask_rgb = (total_mask_rgb * 255).byte()

        # to PIL images
        frames = []
        for t in range(T):
            frame = total_mask_rgb[0, :, t].permute(1, 2, 0).cpu().numpy()
            frames.append(Image.fromarray(frame))

        print(f"Saving total mask to {output_path}")
        export_to_video(frames, output_path, fps=fps)

        return total_mask


def generate_total_mask_for_folder(pipeline, folder_path, output_folder=None,
                                   height=512, width=768, threshold=0.15, dilation_size=15):
    """
    Process a folder containing video.mp4 and object_mask.mp4, generate total_mask.mp4
    """
    if output_folder is None:
        output_folder = folder_path

    video_path = os.path.join(folder_path, "video.mp4")

    object_mask_path = os.path.join(folder_path, "object_mask.mp4")
    output_path = os.path.join(output_folder, "total_mask.mp4")

    if not os.path.exists(video_path):
        print(f"Video not found in {folder_path}")
        return None
    if not os.path.exists(object_mask_path):
        print(f"Object mask not found in {folder_path}")
        return None

    extractor = SelfAttentionMapExtraction(pipeline, extraction_timestep=0.5)
    extractor.setup_extractor()

    try:
        total_mask = extractor.generate_total_mask(
            video_path, object_mask_path, output_path,
            height=height, width=width, threshold=threshold, dilation_size=dilation_size)
        return total_mask
    finally:
        extractor.cleanup()


def visualize_attention_maps(attention_maps, save_path=None, num_cols=4, figsize=(16, 12)):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    num_layers = len(attention_maps)
    if num_layers == 0:
        print("No attention maps to visualize")
        return

    num_rows = (num_layers + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (layer_idx, attn_map) in enumerate(sorted(attention_maps.items())):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        if attn_map.dim() == 4:
            attn_map = attn_map[0].mean(0)
        elif attn_map.dim() == 3:
            attn_map = attn_map[0]

        im = ax.imshow(attn_map.numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(num_layers, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True, help="folder with video.mp4 and object_mask.mp4")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--dilation", type=int, default=15)
    parser.add_argument("--cache_dir", type=str)
    args = parser.parse_args()

    from OmnimatteZero import OmnimatteZero

    print("Loading pipeline...")
    pipe = OmnimatteZero.from_pretrained(
        "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    generate_total_mask_for_folder(
        pipe, args.video_folder,
        height=args.height, width=args.width,
        threshold=args.threshold, dilation_size=args.dilation)

    print("Done!")
