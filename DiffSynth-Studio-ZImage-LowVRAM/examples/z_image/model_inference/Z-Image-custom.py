from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch
import os

# === CONFIGURATION ===
# Path to your custom trained transformer model (.safetensors)
CUSTOM_MODEL_PATH = "/media/misty/M2 Drive/models/diffusion_models/step-5000.safetensors"

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Low VRAM config - offloads models to CPU when not in use
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# Load the pipeline with custom transformer (no download for transformer)
print("Loading pipeline (low VRAM mode)...")
print(f"Using custom transformer: {CUSTOM_MODEL_PATH}")
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # Use path= for custom transformer (skips download)
        ModelConfig(path=CUSTOM_MODEL_PATH, **vram_config),
        # Still need text encoder and VAE from base model
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
print("Pipeline loaded successfully!")

# Get prompt from user
prompt = input("\nEnter your prompt: ")

# Generate image
print("\nGenerating image...")
image = pipe(prompt=prompt, seed=42, rand_device="cuda", num_inference_steps=50, cfg_scale=4)

# Save to same folder as script
output_path = os.path.join(SCRIPT_DIR, "output.jpg")
image.save(output_path)
print(f"\nImage saved to: {output_path}")
