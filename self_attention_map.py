"""
Self-attention map extraction for OmnimatteZero.
Generates total_mask.mp4 from video + object_mask by finding attention-based effects (shadows, reflections).

Usage:
    python self_attention_map.py --video_folder ./example_videos/your_video_name
"""

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


def fix_num_frames_for_vae(num_frames: int, temporal_compression_ratio: int = 8) -> int:
    """
    Make sure num_frames is compatible with VAE: (k * ratio + 1).

    Args:
        num_frames: Original number of frames
        temporal_compression_ratio: VAE temporal compression ratio (default: 8)

    Returns:
        Valid frame count for VAE
    """
    k = (num_frames - 1) // temporal_compression_ratio
    valid_frames = k * temporal_compression_ratio + 1
    if valid_frames < 1:
        valid_frames = temporal_compression_ratio + 1
    return valid_frames


class AttentionMapExtractor:
    """
    Extracts attention maps from transformer attention layers using forward hooks.

    Args:
        model: The transformer model to extract attention from
        layer_indices: Optional list of layer indices to extract (None = all layers)
        attention_type: Type of attention to extract ("self" for attn1, "cross" for attn2)
        average_over_heads: Whether to average attention weights over heads
    """

    def __init__(self, model: nn.Module, layer_indices: Optional[List[int]] = None,
                 attention_type: str = "self", average_over_heads: bool = True):
        self.model = model
        self.layer_indices = layer_indices
        self.attention_type = attention_type
        self.average_over_heads = average_over_heads

        self._hooks = []
        self._attention_maps = {}
        self._is_extracting = False

    def _find_attention_modules(self) -> List[Tuple[str, nn.Module, int]]:
        """Find attention modules in the transformer model."""
        attention_modules = []
        layer_idx = 0

        for name, module in self.model.named_modules():
            name_parts = name.split('.')
            if len(name_parts) < 2:
                continue

            last_part = name_parts[-1]
            is_attn1 = last_part == 'attn1'
            is_attn2 = last_part == 'attn2'

            if not (is_attn1 or is_attn2):
                continue

            if self.attention_type == "self" and not is_attn1:
                continue
            if self.attention_type == "cross" and not is_attn2:
                continue

            if self.layer_indices is not None and layer_idx not in self.layer_indices:
                layer_idx += 1
                continue

            attention_modules.append((name, module, layer_idx))
            layer_idx += 1

        return attention_modules

    def _create_hook(self, layer_idx: int, module_name: str):
        """Create pre and post hooks for attention extraction."""
        stored_inputs = {}

        def pre_hook(module, args, kwargs):
            if not self._is_extracting:
                return
            if len(args) > 0:
                stored_inputs['hidden_states'] = args[0]
            elif 'hidden_states' in kwargs:
                stored_inputs['hidden_states'] = kwargs['hidden_states']

            if len(args) > 1:
                stored_inputs['encoder_hidden_states'] = args[1]
            elif 'encoder_hidden_states' in kwargs:
                stored_inputs['encoder_hidden_states'] = kwargs.get('encoder_hidden_states')

        def post_hook(module, args, kwargs, output):
            if not self._is_extracting:
                return

            try:
                hidden_states = stored_inputs.get('hidden_states')
                if hidden_states is None:
                    return

                encoder_hidden_states = stored_inputs.get('encoder_hidden_states')

                if encoder_hidden_states is None:
                    query = module.to_q(hidden_states)
                    key = module.to_k(hidden_states)
                else:
                    query = module.to_q(hidden_states)
                    key = module.to_k(encoder_hidden_states)

                if hasattr(module, 'norm_q') and module.norm_q is not None:
                    query = module.norm_q(query)
                if hasattr(module, 'norm_k') and module.norm_k is not None:
                    key = module.norm_k(key)

                batch_size, seq_len, _ = query.shape
                head_dim = query.shape[-1] // module.heads

                query = query.view(batch_size, seq_len, module.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

                scale = head_dim ** -0.5
                attention_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
                attention_weights = F.softmax(attention_weights, dim=-1)

                if self.average_over_heads:
                    attention_weights = attention_weights.mean(dim=1)

                self._attention_maps[layer_idx] = attention_weights.detach().cpu()

            except Exception as e:
                pass  # Silently skip layers with errors
            finally:
                stored_inputs.clear()

        return pre_hook, post_hook

    def register_hooks(self) -> int:
        """Register hooks on attention modules. Returns number of modules hooked."""
        self.remove_hooks()
        attention_modules = self._find_attention_modules()

        for name, module, layer_idx in attention_modules:
            pre_hook, post_hook = self._create_hook(layer_idx, name)
            pre_handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            post_handle = module.register_forward_hook(post_hook, with_kwargs=True)
            self._hooks.append(pre_handle)
            self._hooks.append(post_handle)

        return len(self._hooks) // 2

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def clear_maps(self):
        """Clear stored attention maps."""
        self._attention_maps = {}

    @contextmanager
    def extraction_context(self):
        """Context manager for attention extraction."""
        self._is_extracting = True
        self.clear_maps()
        try:
            yield
        finally:
            self._is_extracting = False

    def get_attention_maps(self) -> Dict[int, torch.Tensor]:
        """Get extracted attention maps."""
        return self._attention_maps.copy()


class SelfAttentionMapExtraction:
    """
    Orchestrates self-attention extraction from video for effects mask generation.

    Uses a single noising-denoising step to extract self-attention maps, which reveal
    how different spatial regions in the video are related (e.g., objects and their
    shadows/reflections).

    Args:
        pipeline: The diffusion pipeline (OmnimatteZero)
        extraction_timestep: Timestep for noise injection (0.0-1.0, default: 0.5)
        noise_level: Not used, kept for API compatibility
    """

    def __init__(self, pipeline, extraction_timestep: float = 0.5, noise_level: float = 0.5):
        self.pipeline = pipeline
        self.extraction_timestep = extraction_timestep
        self.noise_level = noise_level
        self.extractor = None

    def setup_extractor(self, layer_indices: Optional[List[int]] = None):
        """Setup the attention extractor with hooks."""
        transformer = self.pipeline.transformer
        self.extractor = AttentionMapExtractor(
            model=transformer, layer_indices=layer_indices,
            attention_type="self", average_over_heads=True)
        num_layers = self.extractor.register_hooks()
        print(f"Registered hooks on {num_layers} self-attention layers")

    def cleanup(self):
        """Remove hooks and cleanup."""
        if self.extractor is not None:
            self.extractor.remove_hooks()
            self.extractor = None

    @torch.no_grad()
    def extract_from_video(self, video: torch.Tensor, prompt: str = "",
                           height: Optional[int] = None, width: Optional[int] = None,
                           generator: Optional[torch.Generator] = None
                           ) -> Tuple[Dict[int, torch.Tensor], Tuple[int, int, int]]:
        """
        Extract self-attention maps from video.

        Args:
            video: Input video tensor (B, C, T, H, W)
            prompt: Optional text prompt
            height, width: Processing dimensions
            generator: Random generator for reproducibility

        Returns:
            Tuple of (attention_maps dict, latent_dimensions tuple)
        """
        if self.extractor is None:
            self.setup_extractor()

        device = self.pipeline._execution_device
        dtype = self.pipeline.transformer.dtype

        if not isinstance(video, torch.Tensor):
            video = self.pipeline.video_processor.preprocess_video(video, height, width)

        video = video.to(device=device, dtype=dtype)
        B, C, T, H, W = video.shape

        vae_temporal_compression = getattr(self.pipeline, 'vae_temporal_compression_ratio', 8)
        valid_frames = fix_num_frames_for_vae(T, vae_temporal_compression)
        if valid_frames != T:
            video = video[:, :, :valid_frames, :, :]
            T = valid_frames

        if height is None:
            height = H
        if width is None:
            width = W

        # Encode video to latents
        latents = self._encode_video(video, generator)

        # Add noise using flow matching
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
        sigma = self.extraction_timestep
        noisy_latents = (1 - sigma) * latents + sigma * noise

        timestep = torch.tensor([self.extraction_timestep * 1000], device=device)

        # Encode prompt
        prompt_embeds, prompt_attention_mask, _, _ = self.pipeline.encode_prompt(
            prompt=prompt, negative_prompt=None, do_classifier_free_guidance=False,
            num_videos_per_prompt=1, device=device)

        num_latent_frames = latents.shape[2]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]

        # Prepare positional encoding
        video_coords = self.pipeline._prepare_video_ids(
            B, num_latent_frames, latent_height, latent_width,
            self.pipeline.transformer_temporal_patch_size,
            self.pipeline.transformer_spatial_patch_size,
            device
        )
        video_coords = self.pipeline._scale_video_ids(
            video_coords, scale_factor=self.pipeline.vae_spatial_compression_ratio,
            scale_factor_t=self.pipeline.vae_temporal_compression_ratio, frame_index=0, device=device)

        packed_latents = self.pipeline._pack_latents(
            noisy_latents, self.pipeline.transformer_spatial_patch_size,
            self.pipeline.transformer_temporal_patch_size)

        # Forward pass with attention extraction
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

    def _encode_video(self, video: torch.Tensor,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Encode video to latent space."""
        latents = retrieve_latents(self.pipeline.vae.encode(video), generator=generator)
        latents = self.pipeline._normalize_latents(
            latents, self.pipeline.vae.latents_mean, self.pipeline.vae.latents_std)
        return latents

    @torch.no_grad()
    def extract_effects_mask(self, video, object_mask, height: int = 512, width: int = 768,
                             threshold: Optional[float] = None, dilation_size: int = 3,
                             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Extract effects mask (shadows, reflections) using self-attention.

        Computes per-frame spatial attention from each position to object regions,
        identifying areas that are semantically related to the object.

        Args:
            video: Input video (path or tensor)
            object_mask: Binary mask of the object (path or tensor)
            height, width: Processing resolution
            threshold: Threshold for binarizing attention map (None = adaptive)
            dilation_size: Kernel size for morphological dilation (default: 3)
            generator: Random generator for reproducibility

        Returns:
            Effects mask tensor (B, 1, T, H, W)
        """
        device = self.pipeline._execution_device
        dtype = self.pipeline.transformer.dtype

        # Load video
        if isinstance(video, str):
            video = load_video(video)
        if not isinstance(video, torch.Tensor):
            video_tensor = self.pipeline.video_processor.preprocess_video(video, height, width)
        else:
            video_tensor = video
        video_tensor = video_tensor.to(device=device, dtype=dtype)

        # Load object mask
        if isinstance(object_mask, str):
            object_mask = load_video(object_mask)
        if not isinstance(object_mask, torch.Tensor):
            mask_tensor = self.pipeline.video_processor.preprocess_video(object_mask, height, width)
        else:
            mask_tensor = object_mask
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        # Binarize mask
        mask_binary = (mask_tensor.mean(dim=1, keepdim=True) > 0).float()

        B, C, T, H, W = video_tensor.shape

        # Extract attention maps
        attention_maps, latent_dims = self.extract_from_video(video_tensor, height=H, width=W, generator=generator)
        num_latent_frames, latent_height, latent_width = latent_dims
        spatial_size = latent_height * latent_width

        vae_temporal_compression = getattr(self.pipeline, 'vae_temporal_compression_ratio', 8)

        # Downsample mask to latent space for each frame
        mask_latent_frames = []
        for t in range(num_latent_frames):
            pixel_t = min(t * vae_temporal_compression, T - 1)
            mask_frame = mask_binary[:, :, pixel_t, :, :]
            mask_latent = F.interpolate(mask_frame, size=(latent_height, latent_width), mode='nearest')
            mask_latent_frames.append(mask_latent)

        mask_latent_all = torch.stack(mask_latent_frames, dim=2)

        # Compute per-frame spatial attention to object
        per_frame_effects = []

        for frame_t in range(num_latent_frames):
            frame_effects_sum = torch.zeros(B, spatial_size, device=device, dtype=dtype)

            for layer_idx, attn_map in attention_maps.items():
                attn_map = attn_map.to(device)
                if attn_map.dim() == 4:
                    attn_map = attn_map.mean(dim=1)

                seq_len = attn_map.shape[-1]
                expected_seq_len = num_latent_frames * spatial_size

                if seq_len != expected_seq_len:
                    continue

                # Reshape to (B, num_frames, H*W, num_frames, H*W)
                attn_reshaped = attn_map.view(B, num_latent_frames, spatial_size, num_latent_frames, spatial_size)

                # Extract attention FROM frame_t TO all frames
                attn_from_frame_t = attn_reshaped[:, frame_t, :, :, :]

                # Sum attention to object regions across all frames
                frame_attention_to_obj = torch.zeros(B, spatial_size, device=device, dtype=dtype)
                for src_t in range(num_latent_frames):
                    src_mask = mask_latent_all[:, 0, src_t, :, :].view(B, -1)
                    attn_to_obj = (attn_from_frame_t[:, :, src_t, :] * src_mask.unsqueeze(1)).sum(dim=-1)
                    frame_attention_to_obj += attn_to_obj

                frame_effects_sum += frame_attention_to_obj

            # Normalize by number of layers
            frame_effects = frame_effects_sum / (len(attention_maps) + 1e-8)
            frame_effects = frame_effects.view(B, latent_height, latent_width)
            per_frame_effects.append(frame_effects)

        # Stack per-frame effects
        effects_latent = torch.stack(per_frame_effects, dim=1)

        # Upsample spatially
        effects_spatial = F.interpolate(
            effects_latent.view(B * num_latent_frames, 1, latent_height, latent_width),
            size=(H, W), mode='bilinear', align_corners=False
        ).view(B, num_latent_frames, H, W)

        # Interpolate temporally
        effects_temporal = F.interpolate(
            effects_spatial.unsqueeze(1),
            size=(T, H, W), mode='trilinear', align_corners=False
        ).squeeze(1)

        effects_mask = effects_temporal.unsqueeze(1)

        # Adaptive thresholding if not specified
        if threshold is None:
            flat_effects = effects_mask.view(-1)
            threshold = flat_effects.mean() + 0.5 * flat_effects.std()
            print(f"Using adaptive threshold: {threshold:.4f}")

        effects_mask_binary = (effects_mask > threshold).float()

        # Apply dilation to smooth edges
        if dilation_size > 0:
            kernel = torch.ones(1, 1, dilation_size, dilation_size, device=device)
            dilated_frames = []
            for t in range(T):
                frame = effects_mask_binary[:, :, t, :, :]
                frame_dilated = F.conv2d(frame, kernel, padding=dilation_size // 2)
                frame_dilated = (frame_dilated > 0).float()
                dilated_frames.append(frame_dilated)
            effects_mask_binary = torch.stack(dilated_frames, dim=2)

        return effects_mask_binary

    @torch.no_grad()
    def generate_total_mask(self, video_path: str, object_mask_path: str, output_path: str,
                            height: int = 512, width: int = 768,
                            threshold: Optional[float] = None, dilation_size: int = 3,
                            fps: int = 24) -> torch.Tensor:
        """
        Generate total_mask.mp4 combining object_mask with attention-based effects.

        Args:
            video_path: Path to input video
            object_mask_path: Path to object mask video
            output_path: Path to save total mask video
            height, width: Processing resolution
            threshold: Threshold for effects mask (None = adaptive)
            dilation_size: Dilation kernel size for smoothing
            fps: Output video frame rate

        Returns:
            Total mask tensor (B, 1, T, H, W)
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

        B, C, T_orig, H, W = video_tensor.shape

        # Fix frame count for VAE compatibility
        vae_temporal_compression = getattr(self.pipeline, 'vae_temporal_compression_ratio', 8)
        valid_frames = fix_num_frames_for_vae(T_orig, vae_temporal_compression)
        if valid_frames != T_orig:
            print(f"Adjusting frames from {T_orig} to {valid_frames} for VAE compatibility")
            video_tensor = video_tensor[:, :, :valid_frames, :, :]
            mask_tensor = mask_tensor[:, :, :valid_frames, :, :]

        T = video_tensor.shape[2]

        # Binarize object mask
        object_mask_binary = (mask_tensor.mean(dim=1, keepdim=True) > 0).float()

        print("Extracting self-attention maps...")
        effects_mask = self.extract_effects_mask(
            video_tensor, object_mask_binary, height=H, width=W,
            threshold=threshold, dilation_size=dilation_size)

        # Combine object mask + effects mask
        total_mask = torch.clamp(object_mask_binary + effects_mask, 0, 1)

        # Report mask statistics
        obj_pixels = object_mask_binary.sum().item()
        total_pixels = total_mask.sum().item()
        ratio = total_pixels / obj_pixels if obj_pixels > 0 else 0
        print(f"Mask coverage: object={obj_pixels:.0f}, total={total_pixels:.0f}, ratio={ratio:.2f}x")

        # Convert to video frames
        total_mask_rgb = total_mask.expand(-1, 3, -1, -1, -1)
        total_mask_rgb = (total_mask_rgb * 255).byte()

        frames = []
        for t in range(T):
            frame = total_mask_rgb[0, :, t].permute(1, 2, 0).cpu().numpy()
            frames.append(Image.fromarray(frame))

        print(f"Saving total mask to {output_path}")
        export_to_video(frames, output_path, fps=fps)

        return total_mask


def generate_total_mask_for_folder(pipeline, folder_path: str, output_folder: Optional[str] = None,
                                   height: int = 512, width: int = 768,
                                   threshold: Optional[float] = None, dilation_size: int = 3):
    """
    Process a folder containing video.mp4 and object_mask.mp4, generate total_mask.mp4.

    Args:
        pipeline: The diffusion pipeline (OmnimatteZero)
        folder_path: Path to folder with video.mp4 and object_mask.mp4
        output_folder: Output folder (default: same as input)
        height, width: Processing resolution
        threshold: Threshold for effects mask (None = adaptive)
        dilation_size: Dilation kernel size for smoothing

    Returns:
        Total mask tensor or None if files not found
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate total_mask.mp4 from video and object_mask using self-attention")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Folder containing video.mp4 and object_mask.mp4")
    parser.add_argument("--height", type=int, default=512,
                        help="Processing height (default: 512)")
    parser.add_argument("--width", type=int, default=768,
                        help="Processing width (default: 768)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for effects mask (default: adaptive)")
    parser.add_argument("--dilation", type=int, default=3,
                        help="Dilation size for smoothing edges (default: 3)")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="HuggingFace cache directory")
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
