# Temporal and Spatial Attention Guidance for OmnimatteZero
# Note: originally developed for LTX-0.9.1, we encountered a bug fetching the model (1/26)
# which we did not encounter during submission (5/25). Currently using LTX-0.9.7 without guidance.

from typing import Optional, List, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TAPNetCorrespondences:
    def __init__(self, model_path=None, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        try:
            from tapnet.torch import tapir_model
            if self.model_path is None:
                self.model = tapir_model.TAPIR(pyramid_level=1)
            else:
                self.model = tapir_model.TAPIR(pyramid_level=1)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
            self.model.eval()
        except ImportError:
            print("TAP-Net not installed, using fallback")
            self.model = None
    
    def compute_correspondences(self, video, query_points=None, num_points=256):
        B, C, T, H, W = video.shape
        
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return self._fallback_correspondences(video, num_points)
        
        video_tap = video.permute(0, 2, 3, 4, 1).contiguous()
        
        if query_points is None:
            query_points = self._generate_grid_query_points(B, T, H, W, num_points)
        
        with torch.no_grad():
            outputs = self.model(video_tap, query_points)
        
        return {
            'tracks': outputs['tracks'],
            'visibility': outputs['occlusion'] < 0.5,
            'confidence': 1.0 - outputs['occlusion'],
        }
    
    def _generate_grid_query_points(self, batch_size, num_frames, height, width, num_points):
        sqrt_n = int(np.sqrt(num_points))
        y = torch.linspace(0.1, 0.9, sqrt_n) * height
        x = torch.linspace(0.1, 0.9, sqrt_n) * width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        points = torch.stack([
            torch.zeros(sqrt_n * sqrt_n),
            yy.flatten(),
            xx.flatten(),
        ], dim=-1)
        
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
        return points.to(self.device)
    
    def _fallback_correspondences(self, video, num_points):
        B, C, T, H, W = video.shape
        sqrt_n = int(np.sqrt(num_points))
        actual_n = sqrt_n * sqrt_n
        
        y = torch.linspace(0.1, 0.9, sqrt_n, device=self.device) * H
        x = torch.linspace(0.1, 0.9, sqrt_n, device=self.device) * W
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        tracks = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        tracks = tracks.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        visibility = torch.ones(B, T, actual_n, device=self.device, dtype=torch.bool)
        confidence = torch.ones(B, T, actual_n, device=self.device)
        
        return {'tracks': tracks, 'visibility': visibility, 'confidence': confidence}


class TemporalAttentionGuidance:
    def __init__(self, correspondence_tracker=None, guidance_scale=1.0, temperature=0.1, use_soft_guidance=True):
        self.tracker = correspondence_tracker or TAPNetCorrespondences()
        self.guidance_scale = guidance_scale
        self.temperature = temperature
        self.use_soft_guidance = use_soft_guidance
        self._correspondences = None
        
    def precompute_correspondences(self, video, mask=None, num_points=256):
        self._correspondences = self.tracker.compute_correspondences(video, num_points=num_points)
        
    def compute_guidance_matrix(self, query_frame, key_frame, latent_height, latent_width, num_latent_frames):
        if self._correspondences is None:
            raise ValueError("call precompute_correspondences first")
        
        tracks = self._correspondences['tracks']
        visibility = self._correspondences['visibility']
        confidence = self._correspondences['confidence']
        
        B, T, N, _ = tracks.shape
        
        query_tracks = tracks[:, query_frame]
        key_tracks = tracks[:, key_frame]
        
        guidance = torch.zeros(B, latent_height * latent_width, latent_height * latent_width,
                               device=tracks.device, dtype=tracks.dtype)
        
        if self.use_soft_guidance:
            query_positions = self._tracks_to_latent_positions(query_tracks, latent_height, latent_width)
            key_positions = self._tracks_to_latent_positions(key_tracks, latent_height, latent_width)
            
            for b in range(B):
                for n in range(N):
                    if visibility[b, query_frame, n] and visibility[b, key_frame, n]:
                        q_pos = query_positions[b, n]
                        k_pos = key_positions[b, n]
                        
                        q_idx = int(q_pos[1]) * latent_width + int(q_pos[0])
                        k_idx = int(k_pos[1]) * latent_width + int(k_pos[0])
                        
                        if 0 <= q_idx < guidance.shape[1] and 0 <= k_idx < guidance.shape[2]:
                            guidance[b, q_idx, k_idx] += confidence[b, query_frame, n]
        else:
            for b in range(B):
                for n in range(N):
                    if visibility[b, query_frame, n] and visibility[b, key_frame, n]:
                        q_y = int(query_tracks[b, n, 1] * latent_height)
                        q_x = int(query_tracks[b, n, 0] * latent_width)
                        k_y = int(key_tracks[b, n, 1] * latent_height)
                        k_x = int(key_tracks[b, n, 0] * latent_width)
                        
                        q_idx = q_y * latent_width + q_x
                        k_idx = k_y * latent_width + k_x
                        
                        if 0 <= q_idx < guidance.shape[1] and 0 <= k_idx < guidance.shape[2]:
                            guidance[b, q_idx, k_idx] = 1.0
        
        guidance = guidance / (guidance.sum(dim=-1, keepdim=True) + 1e-8)
        return guidance
    
    def _tracks_to_latent_positions(self, tracks, latent_height, latent_width):
        scaled_tracks = tracks.clone()
        scaled_tracks[..., 0] = torch.clamp(scaled_tracks[..., 0], 0, latent_width - 1)
        scaled_tracks[..., 1] = torch.clamp(scaled_tracks[..., 1], 0, latent_height - 1)
        return scaled_tracks
    
    def apply_guidance(self, attention_scores, query_frame_indices, key_frame_indices, latent_height, latent_width):
        if self._correspondences is None or self.guidance_scale == 0:
            return attention_scores
        
        B, num_heads, Q, K = attention_scores.shape
        unique_q_frames = query_frame_indices.unique()
        unique_k_frames = key_frame_indices.unique()
        
        for q_frame in unique_q_frames:
            for k_frame in unique_k_frames:
                if q_frame == k_frame:
                    continue
                
                guidance_matrix = self.compute_guidance_matrix(
                    int(q_frame), int(k_frame), latent_height, latent_width, len(unique_q_frames))
                
                q_mask = query_frame_indices == q_frame
                k_mask = key_frame_indices == k_frame
                
                guidance_expanded = guidance_matrix.unsqueeze(1)
                
                attention_scores[:, :, q_mask, :][:, :, :, k_mask] = (
                    (1 - self.guidance_scale) * attention_scores[:, :, q_mask, :][:, :, :, k_mask] +
                    self.guidance_scale * guidance_expanded
                )
        
        return attention_scores


class SpatialAttentionGuidance:
    def __init__(self, guidance_scale=1.0, layers_to_guide=None, guidance_schedule="constant"):
        self.guidance_scale = guidance_scale
        self.layers_to_guide = layers_to_guide
        self.guidance_schedule = guidance_schedule
        self._reference_attention_maps = None
        
    def set_reference_attention_maps(self, attention_maps):
        self._reference_attention_maps = attention_maps
        
    def get_guidance_scale_for_step(self, current_step, total_steps):
        progress = current_step / max(total_steps - 1, 1)
        
        if self.guidance_schedule == "constant":
            return self.guidance_scale
        elif self.guidance_schedule == "linear_decay":
            return self.guidance_scale * (1 - progress)
        elif self.guidance_schedule == "cosine":
            return self.guidance_scale * (1 + np.cos(np.pi * progress)) / 2
        return self.guidance_scale
    
    def apply_guidance(self, attention_scores, layer_index, current_step, total_steps, mask=None):
        if self._reference_attention_maps is None:
            return attention_scores
        
        if self.layers_to_guide is not None and layer_index not in self.layers_to_guide:
            return attention_scores
        
        if layer_index not in self._reference_attention_maps:
            return attention_scores
        
        reference_attn = self._reference_attention_maps[layer_index]
        guidance_scale = self.get_guidance_scale_for_step(current_step, total_steps)
        
        if guidance_scale == 0:
            return attention_scores
        
        if reference_attn.shape != attention_scores.shape:
            reference_attn = F.interpolate(
                reference_attn.view(reference_attn.shape[0], -1, *reference_attn.shape[2:]),
                size=attention_scores.shape[2:], mode='bilinear', align_corners=False
            ).view_as(attention_scores)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            guided_attention = (
                attention_scores * (1 - mask_expanded) +
                ((1 - guidance_scale) * attention_scores + guidance_scale * reference_attn) * mask_expanded
            )
        else:
            guided_attention = (1 - guidance_scale) * attention_scores + guidance_scale * reference_attn
        
        return guided_attention


class CombinedAttentionGuidance:
    def __init__(self, temporal_guidance=None, spatial_guidance=None, temporal_weight=0.5, spatial_weight=0.5):
        self.temporal_guidance = temporal_guidance
        self.spatial_guidance = spatial_guidance
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        
        total_weight = temporal_weight + spatial_weight
        if total_weight > 0:
            self.temporal_weight /= total_weight
            self.spatial_weight /= total_weight
    
    def apply_guidance(self, attention_scores, layer_index, current_step, total_steps,
                       query_frame_indices=None, key_frame_indices=None,
                       latent_height=None, latent_width=None, mask=None):
        result = attention_scores
        
        if (self.temporal_guidance is not None and self.temporal_weight > 0 and
            query_frame_indices is not None and key_frame_indices is not None):
            temporal_result = self.temporal_guidance.apply_guidance(
                attention_scores, query_frame_indices, key_frame_indices, latent_height, latent_width)
            result = (1 - self.temporal_weight) * result + self.temporal_weight * temporal_result
        
        if self.spatial_guidance is not None and self.spatial_weight > 0:
            spatial_result = self.spatial_guidance.apply_guidance(
                result, layer_index, current_step, total_steps, mask)
            result = (1 - self.spatial_weight) * result + self.spatial_weight * spatial_result
        
        return result


def create_attention_guidance_hook(guidance, layer_index, current_step, total_steps, 
                                   latent_height, latent_width, mask=None):
    def hook(module, args, kwargs, output):
        if isinstance(output, tuple):
            attention_output, attention_scores = output[0], output[1] if len(output) > 1 else None
        else:
            return output
        
        if attention_scores is None:
            return output
        
        guided_scores = guidance.apply_guidance(
            attention_scores, layer_index=layer_index, current_step=current_step,
            total_steps=total_steps, latent_height=latent_height, latent_width=latent_width, mask=mask)
        
        if hasattr(module, 'recompute_with_scores'):
            attention_output = module.recompute_with_scores(guided_scores, kwargs.get('value'))
        
        return (attention_output, guided_scores) if isinstance(output, tuple) else attention_output
    
    return hook
