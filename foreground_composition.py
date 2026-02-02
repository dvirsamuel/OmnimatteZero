from typing import Optional, Union
import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image
from OmnimatteZero import OmnimatteZero


def tensor_video_to_pil_images(video_tensor):
    """
    Converts a PyTorch tensor representing a video to a list of PIL Images.

    Args:
        video_tensor (torch.Tensor): A tensor of shape (1, frames, height, width, 3).
                                     Corresponds to batch size, frames, height, width, and RGB channels.

    Returns:
        List[Image.Image]: List of frames as PIL Images.
    """
    # Remove the batch dimension (shape: (frames, height, width, 3))
    video_tensor = video_tensor.squeeze(0)

    # Ensure the tensor is on CPU and convert to NumPy
    video_numpy = video_tensor.cpu().numpy()

    # Convert each frame to a PIL Image
    pil_images = [Image.fromarray(frame.astype('uint8')) for frame in video_numpy]

    return pil_images


class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):

    def forward(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        all, bg, mask, mask2, new_bg = sample
        posterior = self.encode(all).latent_dist
        if sample_posterior:
            z_all = posterior.sample(generator=generator)
        else:
            z_all = posterior.mode()

        posterior = self.encode(bg).latent_dist
        if sample_posterior:
            z_bg = posterior.sample(generator=generator)
        else:
            z_bg = posterior.mode()

        posterior = self.encode(mask).latent_dist
        if sample_posterior:
            z_mask = posterior.sample(generator=generator)
        else:
            z_mask = posterior.mode()

        posterior = self.encode(mask2).latent_dist
        if sample_posterior:
            z_mask2 = posterior.sample(generator=generator)
        else:
            z_mask2 = posterior.mode()

        posterior = self.encode(new_bg).latent_dist
        if sample_posterior:
            z_new_bg = posterior.sample(generator=generator)
        else:
            z_new_bg = posterior.mode()

        z_diff = z_all - z_bg
        z = z_new_bg + z_diff

        dec = self.decode(z, temb)
        dec2 = self.decode(z_diff, temb)
        if not return_dict:
            return (dec,)
        return dec, dec2, self.decode(z_mask, temb), self.decode(z_mask2, temb)

    def forward_encode(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        if not return_dict:
            return (z,)
        return z


pipe = OmnimatteZero.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-diffusers", torch_dtype=torch.bfloat16)
pipe.vae = MyAutoencoderKLLTXVideo.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-diffusers",
                                                   subfolder="vae", torch_dtype=torch.bfloat16)
pipe.to("cuda")

w, h = 768, 512

video_folder = "swan_lake"
# video with the object to extract
video_p = load_video(f"./example_videos/{video_folder}/video.mp4")
video_p = pipe.video_processor.preprocess_video(video_p, width=w, height=h).bfloat16().cuda()

# video of the background only without the object and its effects
video_bg = load_video(f"./results/{video_folder}.mp4")
video_bg = pipe.video_processor.preprocess_video(video_bg, width=w, height=h).bfloat16().cuda()

# the object mask
video_mask = load_video(f"./example_videos/{video_folder}/object_mask.mp4")
video_mask = pipe.video_processor.preprocess_video(video_mask, width=w, height=h).bfloat16().cuda()

# the total masked used the remove the object and its effects
video_mask2 = load_video(f"./example_videos/{video_folder}/total_mask.mp4")
video_mask2 = pipe.video_processor.preprocess_video(video_mask2, width=w, height=h).bfloat16().cuda()

# the new background video to be composed with the object
video_new_bg = load_video("./results/cat_reflection.mp4")
video_new_bg = pipe.video_processor.preprocess_video(video_new_bg, width=w, height=h).bfloat16().cuda()

nframes = min(video_new_bg.shape[2], video_p.shape[2])
# make sure all videos have the same #frames
video_p = video_p[:, :, :nframes, :, :]
video_bg = video_bg[:, :, :nframes, :, :]
video_mask = video_mask[:, :, :nframes, :, :]
video_mask2 = video_mask2[:, :, :nframes, :, :]
video_new_bg = video_new_bg[:, :, :nframes, :, :]

with torch.no_grad():
    x, foreground, z_mask, z_mask2 = pipe.vae([video_p, video_bg, video_mask, video_mask2, video_new_bg],
                                              temb=torch.tensor(0.0, device="cuda", dtype=torch.bfloat16))
    noise = x.sample
    foreground = foreground.sample
    video_mask = z_mask.sample
    video_mask2 = z_mask2.sample
    video_mask = (video_mask.cpu().float() > 0).type(video_bg.dtype).cuda()
    video_mask2 = (video_mask2.cpu().float() > 0).type(video_bg.dtype).cuda()

    # extract foreground layer with pixel injection
    foreground = foreground * (1 - video_mask) + video_p * (video_mask)
    foreground = foreground * (video_mask2)
    video_foreground = tensor_video_to_pil_images(
        ((pipe.video_processor.postprocess_video(foreground, output_type='pt')[0] * 255).long().permute(0, 2, 3, 1)))
    export_to_video(video_foreground, f"/tmp/foreground.mp4", fps=24)

    # latent addition to new background
    noise = noise * (1 - video_mask) + video_p * (video_mask)
    video = tensor_video_to_pil_images(
        ((pipe.video_processor.postprocess_video(noise, output_type='pt')[0] * 255).long().permute(0, 2, 3, 1)))
    export_to_video(video, f"/tmp/latent_addition.mp4", fps=24)

    # apply refinement (few noising denoising steps)
    condition_latents = retrieve_latents(pipe.vae.encode(noise), generator=None)
    condition_latents = pipe._normalize_latents(
        condition_latents, pipe.vae.latents_mean, pipe.vae.latents_std
    ).to(noise.device, dtype=noise.dtype)
    prompt = ""
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    expected_height, expected_width = video[0].size[1], video[0].size[0]
    num_frames = len(video)
    # Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=expected_width,
        height=expected_height,
        num_frames=num_frames,
        denoise_strength=0.3,
        num_inference_steps=10,
        latents=condition_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(0),
        output_type="pil",
    ).frames[0]
    export_to_video(video, f"/results/refinement.mp4", fps=24)
