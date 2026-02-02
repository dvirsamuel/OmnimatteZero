# OmnimatteZero

[![Paper](https://img.shields.io/badge/arXiv-2503.18033-b31b1b.svg)](https://arxiv.org/abs/2503.18033)

Official implementation of **OmnimatteZero: Training-Free Video Matting and Compositing via Latent Diffusion Models**

OmnimatteZero is a training-free approach for video matting, object removal, and layer composition using pre-trained video diffusion models. It leverages the powerful priors learned by video generation models to achieve high-quality results without any task-specific training.

## Features

- **Object Removal**: Remove objects and their effects (shadows, reflections) from videos
- **Foreground Extraction**: Extract foreground layers with associated effects
- **Layer Composition**: Compose extracted foreground layers onto new backgrounds
- All operations are **training-free** and work with off-the-shelf video diffusion models

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- PyTorch 2.4+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/OmnimatteZero.git
cd OmnimatteZero

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The main dependencies include:
- `torch>=2.4.0`
- `diffusers>=0.31.0`
- `transformers>=4.49.0`
- `accelerate>=1.1.1`

## Quick Start

### Data Preparation

Your input data should be organized as follows:

```
example_videos/
├── your_video_name/
│   ├── video.mp4          # Original video with object
│   ├── object_mask.mp4    # Mask of the object only
│   └── total_mask.mp4     # Mask including object + effects (shadows, reflections)
```

- **video.mp4**: The original input video
- **object_mask.mp4**: Binary mask video showing only the object pixels (white = object, black = background)
- **total_mask.mp4**: Binary mask video showing the object AND its effects (shadows, reflections, etc.)

You can generate masks using [SAM2](https://github.com/facebookresearch/segment-anything-2).

### Generating total_mask from object_mask

If you only have the object_mask, you can automatically generate total_mask using self-attention:

```bash
python self_attention_map.py --video_folder ./example_videos/your_video_name
```

This uses the diffusion model's self-attention to find regions that are related to the object (like shadows and reflections).

---

## Object Removal

Remove an object and its effects (shadows, reflections) from a video.

### Usage

```bash
python object_removal.py
```

### Configuration

Edit the following parameters in `object_removal.py`:

```python
# Input directory containing video folders
base_dir = "example_videos"

# Output resolution
expected_height, expected_width = 512, 768

# Generation parameters
num_inference_steps = 30  # More steps = higher quality but slower
```

### How it works

1. Loads the original video and total mask
2. Uses the mask to indicate regions to inpaint
3. The diffusion model fills in the masked regions while maintaining temporal consistency
4. Outputs a clean background video without the object

### Output

Results are saved to the `results/` directory:
```
results/
├── video_name.mp4
```

### Important Note on Attention Guidance

> **⚠️ Note (Updated 1/26):** While our paper describes Temporal Attention Guidance and Spatial Attention Guidance using TAP-Net for improved temporal consistency, these features were originally developed for LTX-Video-0.9.1. We recently encountered a bug while fetching the LTX-0.9.1 model (as of 1/26) which we did not encounter during paper submission (5/25). We are working to fix this issue.
>
> **In the meantime, we have upgraded to LTX-Video-0.9.7**, which we found achieves good object removal results **without** the explicit temporal and spatial attention guidance. The current code runs without attention guidance, but the implementation can be found in `attention_guidance.py` for reference.

---

## Foreground Extraction & Layer Composition

Extract the foreground layer (object + its effects) from a video and compose it onto a new background.

### Prerequisites

Before running, you need:
1. The original video with the object
2. A clean background video (from object removal)
3. Object mask and total mask videos
4. A new background video to compose onto

### Usage

```bash
python foreground_composition.py
```

### Configuration

Edit the following parameters in `foreground_composition.py`:

```python
# Output resolution
w, h = 768, 512

# Video folder name
video_folder = "swan_lake"

# New background to compose onto
video_new_bg = load_video("./results/cat_reflection.mp4")
```

### How it works

1. Encodes both the original video and clean background to latent space
2. Computes the latent difference (foreground = original - background)
3. Uses pixel injection for the object region to preserve details
4. Encodes the new background video to latent space
5. Adds the foreground latents to the new background latents
6. Applies refinement through a few noising-denoising steps

### Output

Results are saved to `results/`:
```
results/
├── foreground.mp4       # Extracted foreground layer
├── latent_addition.mp4  # Latent addition result (before refinement)
└── refinement.mp4       # Final refined composition
```


---

### Generation Parameters Tips
- **num_inference_steps**: 30 is a good default; increase for better quality
- **guidance_scale**: Default is 3.0; adjust based on prompt specificity
- **denoise_strength**: For refinement, 0.3 works well; lower values preserve more details

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{samuel2025omnimattezero,
  author    = {Dvir Samuel and Matan Levy and Nir Darshan and Gal Chechik and Rami Ben-Ari},
  title     = {OmnimatteZero: Fast Training-free Omnimatte with Pre-trained Video Diffusion Models},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  year      = {2025}
}
```

---

## Acknowledgments

- [LTX-Video](https://github.com/Lightricks/LTX-Video) for the base video diffusion model
- [Diffusers](https://github.com/huggingface/diffusers) for the diffusion pipeline infrastructure
- [TAP-Net](https://github.com/google-deepmind/tapnet) for point tracking

---

## Troubleshooting

### Slow Generation
- Reduce `num_inference_steps`
- Use smaller video resolution
- Ensure all packages are installed as specified in `requirements.txt`
