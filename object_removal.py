import os
from tqdm import tqdm
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

from OmnimatteZero import OmnimatteZero


def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


pipe = OmnimatteZero.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-diffusers", torch_dtype=torch.bfloat16,
                                     cache_dir="/inputs/huggingface_cache")
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    "a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers", vae=pipe.vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()

base_dir = "example_videos"
for f_name in tqdm(os.listdir(base_dir)):
    print(f"Current video: {f_name}")
    files = os.listdir(f"{base_dir}/{f_name}")
    try:
        video = load_video(f"./{base_dir}/{f_name}/video.mp4")
    except:
        print(f"video {f_name} not found. Skip it.")
        continue
    condition1 = LTXVideoCondition(video=video, frame_index=0)
    mask = load_video(f"./{base_dir}/{f_name}/total_mask.mp4")
    condition2 = LTXVideoCondition(video=mask, frame_index=0)

    prompt = "Empty"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    expected_height, expected_width = 512, 768
    num_frames = len(video)

    downscaled_height, downscaled_width = int(expected_height), int(
        expected_width)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height,
                                                                                        downscaled_width)
    video = pipe.my_call(
        conditions=[condition1, condition2],
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(1),
        output_type="pil",
    )
    video = video.frames[0]

    export_to_video(video, f"./results/{f_name}.mp4", fps=24)
