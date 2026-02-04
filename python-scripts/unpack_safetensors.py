import torch
from diffusers import StableDiffusionPipeline
# Use StableDiffusionXLPipeline for SDXL models
# from diffusers import StableDiffusionXLPipeline

# Path to your .safetensors file
checkpoint_path = "/media/misty/M2 Drive/models/diffusion_models/step-46000-fixed.safetensors"
# Where to save the Diffusers folder
output_path = "./home/misty/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/e8fa147e7413241c5aa5146a8ae60dc38ade08ae"

# Load the pipeline from the single file
pipe = StableDiffusionPipeline.from_single_file(
    checkpoint_path,
    torch_dtype=torch.float16,
    local_files_only=True
)

# Save in Diffusers format
pipe.save_pretrained(output_path, safe_serialization=True)
print(f"Model converted and saved to {output_path}")

