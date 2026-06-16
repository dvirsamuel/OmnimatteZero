"""
Refine an attention-based total_mask with SAM2 (two-stage Omnimatte masking).

The diffusion self-attention (self_attention_map.py) tells us *which* regions are the
object and its effects (shadows/reflections); SAM2 tells us *exactly where*. This script
seeds SAM2's video predictor on the first frame with points sampled from the object mask
and from the attention effect region, propagates the segmentation across the clip, and
unions it with the object mask to produce a crisp, temporally consistent total_mask.

Requires `transformers>=4.54` (HuggingFace SAM2 integration). The model weights
(facebook/sam2.1-hiera-large) are downloaded automatically on first use.

Usage:
    # first generate the attention mask
    python self_attention_map.py --video_folder ./example_videos/cat_reflection
    # then refine it with SAM2
    python refine_with_sam2.py --video_folder ./example_videos/cat_reflection
"""
import argparse
import os

import numpy as np
import cv2
import torch
from diffusers.utils import load_video, export_to_video
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor


def parse_args():
    p = argparse.ArgumentParser(description="Refine total_mask.mp4 with SAM2")
    p.add_argument("--video_folder", type=str, required=True,
                   help="Folder containing video.mp4, object_mask.mp4 and the attention mask")
    p.add_argument("--input_mask", type=str, default="total_mask.mp4",
                   help="Attention mask used to seed SAM2 (default: total_mask.mp4)")
    p.add_argument("--output_mask", type=str, default="total_mask.mp4",
                   help="Filename for the refined mask (default: overwrite total_mask.mp4)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--model", type=str, default="facebook/sam2.1-hiera-large")
    p.add_argument("--n_obj_points", type=int, default=6,
                   help="Positive points sampled from the object mask")
    p.add_argument("--n_eff_points", type=int, default=10,
                   help="Positive points sampled from the attention effect region")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="HuggingFace cache directory for the SAM2 weights")
    return p.parse_args()


def load_binary_video(path, height, width):
    """Load an mp4 mask as a (T, H, W) boolean array at the given resolution."""
    frames = []
    for im in load_video(path):
        gray = cv2.resize(np.array(im.convert("L")), (width, height),
                          interpolation=cv2.INTER_NEAREST)
        frames.append(gray > 127)
    return np.stack(frames)


def sample_points(mask, n, rng):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def main():
    args = parse_args()
    H, W = args.height, args.width
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    folder = args.video_folder
    video_path = os.path.join(folder, "video.mp4")
    object_path = os.path.join(folder, "object_mask.mp4")
    mask_path = os.path.join(folder, args.input_mask)
    for pth in (video_path, object_path, mask_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)

    video = load_video(video_path)
    obj = load_binary_video(object_path, H, W)
    attn = load_binary_video(mask_path, H, W)
    T = min(len(video), obj.shape[0], attn.shape[0])
    frames = [video[i].convert("RGB").resize((W, H)) for i in range(T)]

    print(f"Loading SAM2: {args.model}")
    processor = Sam2VideoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = Sam2VideoModel.from_pretrained(args.model, dtype=dtype,
                                           cache_dir=args.cache_dir).to(device).eval()

    # Seed SAM2 on frame 0 with points from the object and the attention effect region.
    effect0 = attn[0] & (~obj[0])
    points = np.concatenate(
        [p for p in (sample_points(obj[0], args.n_obj_points, rng),
                     sample_points(effect0, args.n_eff_points, rng)) if len(p)],
        axis=0)

    with torch.inference_mode():
        session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=[1],
            input_points=[[points.tolist()]],
            input_labels=[[[1] * len(points)]])
        _ = model(inference_session=session, frame_idx=0)

        sam_masks = {}
        for out in model.propagate_in_video_iterator(session):
            masks = processor.post_process_masks([out.pred_masks], original_sizes=[[H, W]],
                                                 binarize=True)[0]
            sam_masks[out.frame_idx] = masks.squeeze(1).any(dim=0).cpu().numpy().astype(bool)

    out_frames = []
    for i in range(T):
        sm = sam_masks.get(i, np.zeros((H, W), bool))
        total = (sm | obj[i]).astype(np.uint8) * 255  # always keep the object
        out_frames.append(Image.fromarray(np.stack([total] * 3, axis=-1)))

    out_path = os.path.join(folder, args.output_mask)
    export_to_video(out_frames, out_path, fps=24)
    print(f"Saved refined mask to {out_path}")


if __name__ == "__main__":
    main()
