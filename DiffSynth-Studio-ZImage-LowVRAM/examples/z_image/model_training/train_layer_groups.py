"""
Z-Image Training with Layer Group Offloading

This training script enables full fine-tuning on low-VRAM GPUs (12-16GB) by:
1. Splitting the DIT's 30 transformer layers into groups (default 6 groups of 5 layers)
2. Processing batches of images through each group before swapping
3. Storing boundary activations on CPU RAM between groups

Memory profile:
- GPU: ~8-12GB (one layer group + gradients + activations)
- CPU RAM: ~4-8GB (boundary activations)
- Swaps: 12 per batch (vs 60+ per image with naive per-layer swapping)

Usage:
    python train_layer_groups.py \
        --dataset_base_path /path/to/dataset \
        --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,..." \
        --num_layer_groups 6 \
        --images_per_group_batch 20 \
        --max_pixels 262144
"""

import torch
import torch.nn as nn
import os
import argparse
import gc
import math
import random
import time
import psutil
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from collections import defaultdict
from safetensors.torch import save_file, load_file

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from diffsynth.core import UnifiedDataset, ModelConfig
from diffsynth.core.loader import load_state_dict
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.pipelines.z_image import ZImagePipeline
from diffsynth.diffusion import DiffusionTrainingModule, FlowMatchScheduler
from diffsynth.diffusion.runner import ModelLogger
from diffsynth.diffusion.loss import FlowMatchSFTLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ResourceStats:
    """Statistics for a single profiling section."""
    total_time: float = 0.0
    call_count: int = 0
    gpu_mem_start: float = 0.0
    gpu_mem_end: float = 0.0
    gpu_mem_peak: float = 0.0
    cpu_mem_start: float = 0.0
    cpu_mem_end: float = 0.0


class ResourceProfiler:
    """
    Profiles GPU/CPU memory and timing for training operations.

    Usage:
        profiler = ResourceProfiler(enabled=True)
        with profiler.section("forward_pass"):
            # ... do work ...
        profiler.report()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats: Dict[str, ResourceStats] = defaultdict(ResourceStats)
        self._current_section: Optional[str] = None
        self._section_start_time: float = 0.0
        self._section_start_gpu: float = 0.0
        self._section_start_cpu: float = 0.0

    def get_gpu_memory_gb(self) -> float:
        """Get current GPU memory allocated in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def get_gpu_reserved_gb(self) -> float:
        """Get current GPU memory reserved in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**3
        return 0.0

    def get_cpu_memory_gb(self) -> float:
        """Get current process CPU memory in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3

    def get_system_memory_gb(self) -> Tuple[float, float, float]:
        """Get system memory (total, available, used) in GB."""
        mem = psutil.virtual_memory()
        return mem.total / 1024**3, mem.available / 1024**3, mem.used / 1024**3

    class _SectionContext:
        def __init__(self, profiler: 'ResourceProfiler', name: str):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            if self.profiler.enabled:
                self.profiler._start_section(self.name)
            return self

        def __exit__(self, *args):
            if self.profiler.enabled:
                self.profiler._end_section(self.name)

    def section(self, name: str) -> '_SectionContext':
        """Context manager for profiling a section."""
        return self._SectionContext(self, name)

    def _start_section(self, name: str):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._current_section = name
        self._section_start_time = time.perf_counter()
        self._section_start_gpu = self.get_gpu_memory_gb()
        self._section_start_cpu = self.get_cpu_memory_gb()

        stats = self.stats[name]
        stats.gpu_mem_start = self._section_start_gpu
        stats.cpu_mem_start = self._section_start_cpu

    def _end_section(self, name: str):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - self._section_start_time
        gpu_mem = self.get_gpu_memory_gb()
        cpu_mem = self.get_cpu_memory_gb()

        stats = self.stats[name]
        stats.total_time += elapsed
        stats.call_count += 1
        stats.gpu_mem_end = gpu_mem
        stats.cpu_mem_end = cpu_mem
        stats.gpu_mem_peak = max(stats.gpu_mem_peak, gpu_mem)

        self._current_section = None

    def log_memory(self, label: str = ""):
        """Log current memory state."""
        if not self.enabled:
            return
        gpu_alloc = self.get_gpu_memory_gb()
        gpu_reserved = self.get_gpu_reserved_gb()
        cpu_mem = self.get_cpu_memory_gb()
        sys_total, sys_avail, sys_used = self.get_system_memory_gb()

        print(f"  [{label}] GPU: {gpu_alloc:.2f}GB alloc / {gpu_reserved:.2f}GB reserved | "
              f"Process RAM: {cpu_mem:.1f}GB | System RAM: {sys_used:.1f}/{sys_total:.1f}GB")

    def report(self):
        """Print a summary report of all profiled sections."""
        if not self.enabled or not self.stats:
            return

        print("\n" + "=" * 80)
        print("RESOURCE PROFILER REPORT")
        print("=" * 80)

        # Sort by total time
        sorted_stats = sorted(self.stats.items(), key=lambda x: x[1].total_time, reverse=True)

        total_time = sum(s.total_time for _, s in sorted_stats)

        print(f"{'Section':<40} {'Time':>10} {'%':>6} {'Calls':>6} {'Avg':>10} {'GPU Δ':>8}")
        print("-" * 80)

        for name, stats in sorted_stats:
            pct = (stats.total_time / total_time * 100) if total_time > 0 else 0
            avg_time = stats.total_time / stats.call_count if stats.call_count > 0 else 0
            gpu_delta = stats.gpu_mem_end - stats.gpu_mem_start

            print(f"{name:<40} {stats.total_time:>9.2f}s {pct:>5.1f}% {stats.call_count:>6} "
                  f"{avg_time*1000:>9.1f}ms {gpu_delta:>+7.2f}GB")

        print("-" * 80)
        print(f"{'TOTAL':<40} {total_time:>9.2f}s")

        # Memory summary
        print("\nMemory Summary:")
        print(f"  GPU Allocated: {self.get_gpu_memory_gb():.2f}GB")
        print(f"  GPU Reserved:  {self.get_gpu_reserved_gb():.2f}GB")
        print(f"  Process RAM:   {self.get_cpu_memory_gb():.1f}GB")
        sys_total, sys_avail, sys_used = self.get_system_memory_gb()
        print(f"  System RAM:    {sys_used:.1f}GB / {sys_total:.1f}GB ({sys_used/sys_total*100:.1f}% used)")
        print("=" * 80 + "\n")

    def reset(self):
        """Reset all statistics."""
        self.stats.clear()


def print_gpu_memory(prefix=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class LayerGroupOffloader:
    """
    Manages layer group offloading during training.

    This class handles:
    1. Splitting layers into groups
    2. Loading/offloading groups to/from GPU
    3. Processing batches through groups efficiently
    """

    def __init__(
        self,
        dit_model: nn.Module,
        num_groups: int = 6,
        computation_device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
    ):
        self.dit = dit_model
        self.num_groups = num_groups
        self.computation_device = computation_device
        self.dtype = dtype
        self.verbose = verbose

        # Get layers
        self.all_layers = list(dit_model.layers)
        self.num_layers = len(self.all_layers)
        self.layers_per_group = (self.num_layers + num_groups - 1) // num_groups

        # Create groups
        self.groups = []
        for i in range(0, self.num_layers, self.layers_per_group):
            group_layers = self.all_layers[i:i + self.layers_per_group]
            self.groups.append(group_layers)

        # All layers start on CPU
        for layer in self.all_layers:
            layer.to("cpu")

        torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            print(f"LayerGroupOffloader initialized:")
            print(f"  - {self.num_layers} total layers")
            print(f"  - {len(self.groups)} groups of ~{self.layers_per_group} layers each")
            for i, group in enumerate(self.groups):
                print(f"    Group {i}: {len(group)} layers")

    def load_group(self, group_idx: int):
        """Load a layer group to GPU"""
        if group_idx < 0 or group_idx >= len(self.groups):
            raise ValueError(f"Invalid group index: {group_idx}")

        for layer in self.groups[group_idx]:
            layer.to(self.computation_device, dtype=self.dtype)

        if self.verbose:
            print_gpu_memory(f"  After loading group {group_idx}: ")

    def offload_group(self, group_idx: int):
        """Offload a layer group to CPU"""
        if group_idx < 0 or group_idx >= len(self.groups):
            return

        for layer in self.groups[group_idx]:
            layer.to("cpu")

        torch.cuda.empty_cache()
        gc.collect()

    def get_group_layers(self, group_idx: int) -> List[nn.Module]:
        """Get layers in a group"""
        return self.groups[group_idx]

    def forward_through_group(
        self,
        group_idx: int,
        x: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        **layer_kwargs,
    ) -> torch.Tensor:
        """Forward pass through a single layer group"""
        for layer in self.groups[group_idx]:
            if use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, **layer_kwargs, use_reentrant=False
                )
            else:
                x = layer(x, **layer_kwargs)
        return x


class CPUOffloadedAdamW:
    """
    AdamW optimizer with CPU-offloaded momentum and variance states.

    This optimizer stores all optimizer states (momentum, variance) on CPU RAM
    and loads them to GPU one layer group at a time during the optimizer step.
    This allows using AdamW on models that don't fit in GPU memory.

    Key features:
    - First and second moment states stored on CPU
    - Parameters updated one layer group at a time
    - Compatible with layer group offloading training
    """

    def __init__(
        self,
        param_groups: List[Dict[str, Any]],
        lr: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        verbose: bool = False,
        state_dtype: torch.dtype = torch.float32,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.verbose = verbose
        # Note: state_dtype is accepted for API compatibility but optimizer states
        # are always stored in float32 for numerical stability
        self.state_dtype = state_dtype

        # Store optimizer state on CPU
        self.state: Dict[str, Dict[str, Any]] = {}

        # Track all parameter names for iteration
        self.param_names: List[str] = []
        for group in param_groups:
            for name in group.get('param_names', []):
                self.param_names.append(name)

        if verbose:
            print(f"CPUOffloadedAdamW initialized:")
            print(f"  - LR: {lr}, Betas: {betas}, Weight decay: {weight_decay}")
            print(f"  - Optimizer state dtype: {self.state_dtype} (always uses float32 internally)")
            print(f"  - {len(self.param_names)} parameters tracked")

    def _get_or_create_state(self, param_name: str, param_shape: torch.Size, dtype: torch.dtype) -> Dict[str, Any]:
        """Get existing state or create new one for a parameter"""
        if param_name not in self.state:
            self.state[param_name] = {
                'step': 0,
                'exp_avg': torch.zeros(param_shape, dtype=torch.float32, device='cpu'),
                'exp_avg_sq': torch.zeros(param_shape, dtype=torch.float32, device='cpu'),
            }
        return self.state[param_name]

    def step_for_param(
        self,
        param_name: str,
        param: torch.Tensor,
        grad: torch.Tensor,
    ):
        """
        Perform AdamW update for a single parameter.

        Args:
            param_name: Unique name for this parameter
            param: The parameter tensor (on GPU)
            grad: The gradient tensor (on GPU)
        """
        state = self._get_or_create_state(param_name, param.shape, param.dtype)

        # Increment step
        state['step'] += 1
        step = state['step']

        beta1, beta2 = self.betas

        # Move state to GPU for computation (in float32 for numerical stability)
        exp_avg = state['exp_avg'].to(param.device, dtype=torch.float32)
        exp_avg_sq = state['exp_avg_sq'].to(param.device, dtype=torch.float32)
        grad_f32 = grad.float()

        # Bias correction terms
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad_f32, alpha=1 - beta1)

        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1 - beta2)

        # Compute step size
        step_size = self.lr / bias_correction1

        # Denominator
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

        # Apply weight decay
        with torch.no_grad():
            param.mul_(1 - self.lr * self.weight_decay)

            # Apply update
            param.addcdiv_(exp_avg, denom, value=-step_size)

        # Move state back to CPU
        state['exp_avg'] = exp_avg.cpu()
        state['exp_avg_sq'] = exp_avg_sq.cpu()

    def zero_grad(self):
        """No-op since gradients are handled externally"""
        pass

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.lr

    def set_lr(self, lr: float):
        """Set learning rate"""
        self.lr = lr

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state for checkpointing."""
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": {k: {
                "step": v["step"],
                "exp_avg": v["exp_avg"].clone(),
                "exp_avg_sq": v["exp_avg_sq"].clone(),
            } for k, v in self.state.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict["lr"]
        self.betas = state_dict["betas"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        self.state = {k: {
            "step": v["step"],
            "exp_avg": v["exp_avg"].clone(),
            "exp_avg_sq": v["exp_avg_sq"].clone(),
        } for k, v in state_dict["state"].items()}


class LRScheduler:
    """
    Learning rate scheduler with warmup and multiple decay options.

    Supports:
    - cosine: Cosine decay to min_lr_ratio * base_lr
    - linear: Linear decay to min_lr_ratio * base_lr
    - constant: No decay after warmup
    """

    def __init__(
        self,
        optimizer: CPUOffloadedAdamW,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        scheduler_type: str = "cosine",
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.scheduler_type = scheduler_type
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step"""
        self.current_step += 1
        new_lr = self._get_lr()
        self.optimizer.set_lr(new_lr)

    def _get_lr(self) -> float:
        """Calculate learning rate for current step"""
        if self.current_step < self.num_warmup_steps:
            # Linear warmup
            warmup_progress = self.current_step / max(1, self.num_warmup_steps)
            return self.base_lr * warmup_progress

        if self.scheduler_type == "constant":
            return self.base_lr

        # Calculate progress through decay phase
        progress = (self.current_step - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        progress = min(1.0, progress)  # Clamp to [0, 1]

        if self.scheduler_type == "cosine":
            # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
        elif self.scheduler_type == "linear":
            # Linear decay
            decayed = 1.0 - progress * (1.0 - self.min_lr_ratio)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return self.base_lr * decayed

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.get_lr()

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing"""
        return {
            "current_step": self.current_step,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint"""
        self.current_step = state_dict["current_step"]
        self.base_lr = state_dict["base_lr"]


class ZImageLayerGroupTrainingModule(DiffusionTrainingModule):
    """
    Training module that uses layer group offloading for memory efficiency.

    KEY DIFFERENCE: All models are loaded to CPU first, then selectively
    moved to GPU during training to fit in limited VRAM.
    """

    def __init__(
        self,
        # Standard args
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=True,
        fp8_models=None,
        computation_device="cuda",
        # Layer group args
        num_layer_groups: int = 6,
        images_per_group_batch: int = 20,
        max_pixels: int = 262144,
        verbose: bool = True,
        # Profiling and optimization
        profile: bool = False,
        activation_cpu_dtype: torch.dtype = None,  # None = same as computation dtype
    ):
        super().__init__()

        self.num_layer_groups = num_layer_groups
        self.images_per_group_batch = images_per_group_batch
        self.verbose = verbose
        self.profile = profile
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.computation_device = computation_device
        self.dtype = torch.bfloat16
        # For CPU storage, use bfloat16 by default (saves RAM vs float32)
        self.activation_cpu_dtype = activation_cpu_dtype if activation_cpu_dtype else torch.bfloat16

        # Resource profiler for performance analysis
        self.profiler = ResourceProfiler(enabled=profile)

        # Parse FP8 models
        self.fp8_model_names = [] if fp8_models is None else fp8_models.split(",")

        print(f"\n{'='*60}")
        print("Loading models to CPU first (low-VRAM mode)")
        print(f"{'='*60}")

        # CRITICAL: Load ALL models to CPU first, not GPU
        # This is the key difference from standard training
        model_configs = self._parse_model_configs_for_cpu(
            model_paths, model_id_with_origin_paths
        )

        tokenizer_config = (
            ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )

        # Load pipeline with device="cpu" to keep everything on CPU initially
        self.pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",  # CRITICAL: Load to CPU
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        print_gpu_memory("After loading to CPU: ")

        # Setup training mode
        self.pipe.scheduler.set_timesteps(1000, training=True)
        trainable_list = [] if trainable_models is None else trainable_models.split(",")
        self.pipe.freeze_except(trainable_list)

        # Apply FP8 quantization to frozen models if requested
        self._apply_fp8_to_models()

        # Create layer group offloader for DIT
        if self.pipe.dit is not None:
            self.layer_offloader = LayerGroupOffloader(
                self.pipe.dit,
                num_groups=num_layer_groups,
                computation_device=computation_device,
                dtype=torch.bfloat16,
                verbose=verbose,
            )
        else:
            raise ValueError("DIT model not loaded!")

        # Keep small DIT components on CPU for now, move to GPU when needed
        self._dit_persistent_on_gpu = False

        # Image resizer to ensure proper dimensions
        self.image_resizer = ImageCropAndResize(
            max_pixels=max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        )

        # Storage for boundary activations
        self.boundary_storage: Dict[int, List[torch.Tensor]] = {}

        # Gradient accumulator (stores gradients on CPU between group processing)
        self.accumulated_grads: Dict[int, torch.Tensor] = {}

        print(f"\n{'='*60}")
        print("Initialization complete!")
        print(f"{'='*60}")
        print_gpu_memory("Final: ")

    def _parse_model_configs_for_cpu(self, model_paths, model_id_with_origin_paths):
        """Parse model configs - all models load to CPU"""
        model_configs = []

        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                config = self.parse_path_or_model_id(model_id_with_origin_path)
                # All models go to CPU initially
                model_configs.append(ModelConfig(
                    model_id=config.model_id,
                    origin_file_pattern=config.origin_file_pattern,
                    computation_device="cpu",
                    computation_dtype=torch.bfloat16,
                ))

        return model_configs

    def _apply_fp8_to_models(self):
        """Apply FP8 quantization to frozen models"""
        # NOTE: FP8 conversion is disabled for now because the transformers
        # library's Linear layers don't support FP8 weights directly.
        # Memory savings are achieved through CPU offloading instead.
        #
        # To enable FP8, the models would need to use the AutoWrappedLinear
        # layers from diffsynth.core.vram.layers which handle FP8 properly.
        if self.fp8_model_names:
            print(f"FP8 requested for: {self.fp8_model_names}")
            print("  NOTE: FP8 is disabled in layer group training mode.")
            print("  Memory savings come from CPU offloading instead.")
            print("  Models will use BF16 precision.")

    def _move_dit_persistent_to_gpu(self):
        """Move small DIT components to GPU"""
        if self._dit_persistent_on_gpu:
            return

        dit = self.pipe.dit
        device = self.computation_device
        dtype = self.dtype

        # Move small components
        components = ['t_embedder', 'cap_embedder', 'all_x_embedder', 'all_final_layer',
                      'noise_refiner', 'context_refiner']
        for name in components:
            if hasattr(dit, name):
                comp = getattr(dit, name)
                if comp is not None and isinstance(comp, nn.Module):
                    comp.to(device, dtype=dtype)

        # Parameters
        for name in ['x_pad_token', 'cap_pad_token']:
            if hasattr(dit, name):
                param = getattr(dit, name)
                if param is not None:
                    param.data = param.data.to(device, dtype=dtype)

        # Rope embedder (not a Module, needs special handling)
        if hasattr(dit, 'rope_embedder') and dit.rope_embedder is not None:
            if hasattr(dit.rope_embedder, 'freqs_cis') and dit.rope_embedder.freqs_cis is not None:
                dit.rope_embedder.freqs_cis = [f.to(device) for f in dit.rope_embedder.freqs_cis]

        self._dit_persistent_on_gpu = True
        print_gpu_memory("After moving DIT persistent components to GPU: ")

    def _move_dit_persistent_to_cpu(self):
        """Move small DIT components back to CPU"""
        if not self._dit_persistent_on_gpu:
            return

        dit = self.pipe.dit

        components = ['t_embedder', 'cap_embedder', 'all_x_embedder', 'all_final_layer',
                      'noise_refiner', 'context_refiner']
        for name in components:
            if hasattr(dit, name):
                comp = getattr(dit, name)
                if comp is not None and isinstance(comp, nn.Module):
                    comp.to("cpu")

        for name in ['x_pad_token', 'cap_pad_token']:
            if hasattr(dit, name):
                param = getattr(dit, name)
                if param is not None:
                    param.data = param.data.to("cpu")

        self._dit_persistent_on_gpu = False
        torch.cuda.empty_cache()

    def _move_text_encoder_to_gpu(self):
        """Temporarily move text encoder to GPU for encoding"""
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.to(self.computation_device)
            print_gpu_memory("After loading text_encoder to GPU: ")

    def _move_text_encoder_to_cpu(self):
        """Move text encoder back to CPU"""
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

    def _move_vae_to_gpu(self):
        """Temporarily move VAE to GPU"""
        if self.pipe.vae_encoder is not None:
            self.pipe.vae_encoder.to(self.computation_device)

    def _move_vae_to_cpu(self):
        """Move VAE back to CPU"""
        if self.pipe.vae_encoder is not None:
            self.pipe.vae_encoder.to("cpu")
        if self.pipe.vae_decoder is not None:
            self.pipe.vae_decoder.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    def clear_boundaries(self):
        """Clear stored boundary activations"""
        self.boundary_storage.clear()
        gc.collect()

    def store_boundary(self, group_idx: int, activations: List[torch.Tensor]):
        """Store boundary activations on CPU"""
        self.boundary_storage[group_idx] = [
            act.detach().cpu() for act in activations
        ]

    def get_boundary(self, group_idx: int) -> List[torch.Tensor]:
        """Retrieve boundary activations"""
        return self.boundary_storage.get(group_idx, [])

    def accumulate_gradients(self):
        """Accumulate current gradients to CPU storage"""
        for name, param in self.pipe.dit.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_id = id(param)
                grad_cpu = param.grad.detach().cpu()
                if param_id in self.accumulated_grads:
                    self.accumulated_grads[param_id] += grad_cpu
                else:
                    self.accumulated_grads[param_id] = grad_cpu
                param.grad = None  # Clear to save GPU memory

    def apply_accumulated_gradients(self):
        """Apply accumulated gradients back to parameters

        This loads each layer group, applies gradients, then offloads.
        Must be called before optimizer.step()
        """
        if not self.accumulated_grads:
            return

        if self.verbose:
            print(f"  Applying accumulated gradients ({len(self.accumulated_grads)} params)...")

        for group_idx in range(self.layer_offloader.num_groups):
            self.layer_offloader.load_group(group_idx)

            for layer_idx, layer in enumerate(self.layer_offloader.groups[group_idx]):
                layer_base_idx = group_idx * self.layer_offloader.layers_per_group + layer_idx
                for name, param in layer.named_parameters():
                    if param.requires_grad:
                        param_key = f"layers.{layer_base_idx}.{name}"
                        if param_key in self.accumulated_grads:
                            param.grad = self.accumulated_grads[param_key].to(
                                param.device, dtype=param.dtype
                            )

            # DON'T offload yet - optimizer needs params on GPU

        self.accumulated_grads.clear()

    def run_optimizer_step(self, optimizer: CPUOffloadedAdamW):
        """Run optimizer step one layer group at a time to save memory.

        Instead of loading all layers to GPU for optimizer, we:
        1. Load one group at a time
        2. Apply gradients to that group using AdamW (with CPU-offloaded states)
        3. Offload the group

        Args:
            optimizer: CPUOffloadedAdamW optimizer instance
        """
        if not self.accumulated_grads:
            if self.verbose:
                print("  No gradients to apply")
            return

        if self.verbose:
            lr = optimizer.get_lr()
            print(f"  Optimizer step ({len(self.accumulated_grads)} params, AdamW, lr={lr:.2e})...")
            self.profiler.log_memory("optimizer_step_start")

        num_groups = self.layer_offloader.num_groups

        with self.profiler.section("optimizer_step"):
            # Update each layer group one at a time with progress bar
            opt_pbar = tqdm(
                range(num_groups + 1),  # +1 for persistent params
                desc="  Optimizer step",
                leave=False,
                position=1,
            )
            for group_idx in range(num_groups):
                opt_pbar.set_postfix({"group": f"{group_idx+1}/{num_groups}"})

                with self.profiler.section(f"opt_group_{group_idx}"):
                    self.layer_offloader.load_group(group_idx)

                    for layer_idx, layer in enumerate(self.layer_offloader.groups[group_idx]):
                        layer_base_idx = group_idx * self.layer_offloader.layers_per_group + layer_idx
                        for name, param in layer.named_parameters():
                            if param.requires_grad:
                                param_key = f"layers.{layer_base_idx}.{name}"
                                if param_key in self.accumulated_grads:
                                    grad = self.accumulated_grads[param_key].to(
                                        param.device, dtype=param.dtype
                                    )
                                    optimizer.step_for_param(param_key, param, grad)
                                    del grad  # Clean up immediately

                    self.layer_offloader.offload_group(group_idx)
                opt_pbar.update(1)

            # Also update persistent DIT components (embedders, final layer, refiners)
            opt_pbar.set_postfix({"group": "persistent"})
            with self.profiler.section("opt_persistent"):
                self._move_dit_persistent_to_gpu()
                dit = self.pipe.dit
                for name, param in dit.named_parameters():
                    if param.requires_grad and not name.startswith("layers."):
                        param_key = f"persistent.{name}"
                        if param_key in self.accumulated_grads:
                            grad = self.accumulated_grads[param_key].to(
                                param.device, dtype=param.dtype
                            )
                            optimizer.step_for_param(param_key, param, grad)
                            del grad

                self._move_dit_persistent_to_cpu()
            opt_pbar.update(1)
            opt_pbar.close()

        self.accumulated_grads.clear()
        torch.cuda.empty_cache()

        if self.verbose:
            self.profiler.log_memory("optimizer_step_end")

    def prepare_batch_data(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prepare a batch of images for layer-group processing.

        Encodes all text prompts and images, samples timesteps, and prepares
        the initial hidden states. This is done before layer-group forward
        to minimize model swapping.

        Args:
            batch: List of dicts, each with 'image' and 'prompt' keys

        Returns:
            Dict containing all prepared data for batch processing
        """
        device = self.computation_device
        dtype = self.dtype
        cpu_dtype = self.activation_cpu_dtype
        dit = self.pipe.dit
        batch_size = len(batch)

        if self.verbose:
            print(f"  Preparing batch of {batch_size} images...")
            self.profiler.log_memory("batch_start")

        # ============================================
        # Phase 1: Text encoding (text encoder on GPU once)
        # ============================================
        with self.profiler.section("text_encoder_load"):
            self._move_text_encoder_to_gpu()

        all_prompt_embeds = []
        with self.profiler.section("text_encoding"):
            for item in batch:
                prompt_embeds = self.encode_prompt(item["prompt"])
                # Store in CPU dtype (bfloat16 saves RAM vs float32)
                all_prompt_embeds.append(prompt_embeds.to(cpu_dtype).cpu())

        with self.profiler.section("text_encoder_offload"):
            self._move_text_encoder_to_cpu()

        if self.verbose:
            self.profiler.log_memory("after_text_encoding")

        # ============================================
        # Phase 2: Image encoding (VAE on GPU once)
        # ============================================
        with self.profiler.section("vae_load"):
            self._move_vae_to_gpu()

        all_input_latents = []
        all_noise = []
        all_heights = []
        all_widths = []

        with self.profiler.section("vae_encoding"):
            for item in batch:
                image = item["image"]
                # Resize image to proper dimensions
                image = self.image_resizer(image)
                height, width = image.size[1], image.size[0]
                all_heights.append(height)
                all_widths.append(width)

                # Generate noise on CPU (saves GPU memory)
                noise = self.pipe.generate_noise(
                    (1, 16, height // 8, width // 8),
                    rand_device="cpu",
                    rand_torch_dtype=cpu_dtype,  # Use CPU dtype
                )
                all_noise.append(noise)  # Already on CPU

                # Encode image
                image_tensor = self.pipe.preprocess_image(image).to(device, dtype=dtype)
                with torch.no_grad():
                    input_latents = self.pipe.vae_encoder(image_tensor)
                # Store in CPU dtype
                all_input_latents.append(input_latents.to(cpu_dtype).cpu())

                # Explicitly delete GPU tensor
                del image_tensor

        with self.profiler.section("vae_offload"):
            self._move_vae_to_cpu()

        if self.verbose:
            self.profiler.log_memory("after_vae_encoding")

        # ============================================
        # Phase 3: Prepare DIT inputs (persistent components on GPU)
        # ============================================
        with self.profiler.section("dit_persistent_load"):
            self._move_dit_persistent_to_gpu()

        all_unified = []
        all_unified_freqs_cis = []
        all_timesteps = []
        all_targets = []
        all_patch_metadata = []
        all_t_noisy = []

        with self.profiler.section("dit_input_prep"):
            for i in range(batch_size):
                # Sample random timestep
                timestep = torch.randint(0, 1000, (1,), device=device)
                all_timesteps.append(timestep.cpu())  # Store on CPU

                # Move data to GPU temporarily
                noise = all_noise[i].to(device, dtype=dtype)
                input_latents = all_input_latents[i].to(device, dtype=dtype)
                prompt_embeds = all_prompt_embeds[i].to(device, dtype=dtype)

                t = timestep / 1000.0

                # Create noisy latents (flow matching)
                latents = (1 - t.view(-1, 1, 1, 1)) * input_latents + t.view(-1, 1, 1, 1) * noise

                # Target for loss - store in CPU dtype
                target = (noise - input_latents).to(cpu_dtype).cpu()
                all_targets.append(target)

                # Timestep embedding - DETACH to prevent graph issues during backward
                t_noisy = dit.t_embedder(1000 - timestep).detach()
                all_t_noisy.append(t_noisy.to(cpu_dtype).cpu())

                # Patchify
                latents_reshaped = rearrange(latents, "B C H W -> C B H W")
                x, cap_feats, patch_metadata = dit.patchify_and_embed(
                    [latents_reshaped], [prompt_embeds]
                )
                all_patch_metadata.append(patch_metadata)

                x = x[0].to(dtype=dtype)
                cap_feats = cap_feats[0].to(dtype=dtype)

                # Embed and prepare x
                x = dit.all_x_embedder["2-1"](x)
                x[torch.cat(patch_metadata.get("x_pad_mask"))] = dit.x_pad_token.to(dtype=dtype, device=device)
                x_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata.get("x_pos_ids"), dim=0).to(device))
                x = rearrange(x, "L C -> 1 L C")
                x_freqs_cis = rearrange(x_freqs_cis, "L C -> 1 L C")

                # Noise refiner
                for layer in dit.noise_refiner:
                    x = layer(x=x, attn_mask=None, freqs_cis=x_freqs_cis, adaln_input=t_noisy)

                # Cap embedder
                cap_feats = dit.cap_embedder(cap_feats)
                cap_feats[torch.cat(patch_metadata.get("cap_pad_mask"))] = dit.cap_pad_token.to(dtype=dtype, device=device)
                cap_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata.get("cap_pos_ids"), dim=0).to(device))
                cap_feats = rearrange(cap_feats, "L C -> 1 L C")
                cap_freqs_cis = rearrange(cap_freqs_cis, "L C -> 1 L C")

                # Context refiner
                for layer in dit.context_refiner:
                    cap_feats = layer(x=cap_feats, attn_mask=None, freqs_cis=cap_freqs_cis)

                # Unified sequence
                unified = torch.cat([x, cap_feats], dim=1)
                unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)

                # Store on CPU in efficient dtype
                # DETACH to prevent graph issues - unified has graphs from embedders/refiners
                # Note: freqs_cis is complex, don't cast it (keep original dtype)
                all_unified.append(unified.detach().to(cpu_dtype).cpu())
                all_unified_freqs_cis.append(unified_freqs_cis.detach().cpu())  # Keep complex dtype

                # Clean up GPU tensors from this iteration
                del noise, input_latents, prompt_embeds, latents, latents_reshaped
                del x, cap_feats, x_freqs_cis, cap_freqs_cis, unified, unified_freqs_cis, t_noisy

        # Clear intermediate lists that are no longer needed
        del all_noise, all_input_latents, all_prompt_embeds
        torch.cuda.empty_cache()

        if self.verbose:
            self.profiler.log_memory("after_dit_prep")

        return {
            "unified": all_unified,  # List of [1, L, C] on CPU in cpu_dtype
            "unified_freqs_cis": all_unified_freqs_cis,  # List of [1, L, C] on CPU
            "timesteps": all_timesteps,  # List of timesteps on CPU
            "t_noisy": all_t_noisy,  # Already stored above in cpu_dtype
            "targets": all_targets,  # List of [1, C, H, W] on CPU in cpu_dtype
            "patch_metadata": all_patch_metadata,
            "heights": all_heights,
            "widths": all_widths,
            "batch_size": batch_size,
        }

    def forward_batch_through_layers(
        self,
        prepared_data: Dict[str, Any],
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward all images through layer groups efficiently.

        Key optimization: Load each layer group once and process ALL images
        through it before moving to the next group.

        Swap count: O(num_groups) instead of O(num_images * num_groups)

        Args:
            prepared_data: Output from prepare_batch_data()

        Returns:
            Tuple of (final_outputs, all_activations)
            - final_outputs: List of unified outputs after all layers (on CPU)
            - all_activations: List of lists, activations[i][g] = activation for
              image i after group g (on CPU)
        """
        device = self.computation_device
        dtype = self.dtype
        cpu_dtype = self.activation_cpu_dtype
        batch_size = prepared_data["batch_size"]

        if self.verbose:
            print(f"  Forward pass: {batch_size} images through {self.layer_offloader.num_groups} layer groups")
            self.profiler.log_memory("forward_start")

        # Initialize activations storage
        # all_activations[i] = list of activations for image i (one per group boundary)
        # NOTE: We reference the same tensor, not clone, to save RAM
        all_activations = [[prepared_data["unified"][i]] for i in range(batch_size)]

        # Current hidden states for each image (these will be updated in-place references)
        current_unified = list(prepared_data["unified"])  # Shallow copy of list

        num_groups = self.layer_offloader.num_groups

        # Process each layer group with progress bar
        group_pbar = tqdm(
            range(num_groups),
            desc="  Forward groups",
            leave=False,
            position=1,
        )
        for group_idx in group_pbar:
            group_pbar.set_postfix({"group": f"{group_idx+1}/{num_groups}"})

            with self.profiler.section(f"layer_group_{group_idx}_load"):
                self.layer_offloader.load_group(group_idx)

            with self.profiler.section(f"layer_group_{group_idx}_forward"):
                # Process ALL images through this group with progress bar
                img_pbar = tqdm(
                    range(batch_size),
                    desc=f"    Group {group_idx+1} images",
                    leave=False,
                    position=2,
                )
                for img_idx in img_pbar:
                    unified = current_unified[img_idx].to(device, dtype=dtype)
                    # freqs_cis is complex - don't cast to bfloat16, just move to device
                    freqs_cis = prepared_data["unified_freqs_cis"][img_idx].to(device)
                    t_noisy = prepared_data["t_noisy"][img_idx].to(device, dtype=dtype)

                    layer_kwargs = {
                        "attn_mask": None,
                        "freqs_cis": freqs_cis,
                        "adaln_input": t_noisy,
                    }

                    with torch.no_grad():
                        unified = self.layer_offloader.forward_through_group(
                            group_idx, unified,
                            use_gradient_checkpointing=False,
                            **layer_kwargs,
                        )

                    # Store output on CPU in efficient dtype
                    # Only store once to save RAM (don't duplicate in current_unified)
                    unified_cpu = unified.to(cpu_dtype).cpu()
                    current_unified[img_idx] = unified_cpu
                    all_activations[img_idx].append(unified_cpu)

                    # Clean up GPU tensors
                    del unified, freqs_cis, t_noisy
                img_pbar.close()

            with self.profiler.section(f"layer_group_{group_idx}_offload"):
                self.layer_offloader.offload_group(group_idx)

            if self.verbose:
                self.profiler.log_memory(f"after_group_{group_idx}")
        group_pbar.close()

        return current_unified, all_activations

    def backward_batch_through_layers(
        self,
        prepared_data: Dict[str, Any],
        all_activations: List[List[torch.Tensor]],
        loss_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Backward through layer groups for all images efficiently.

        Key optimization: Load each layer group once and backward ALL images
        through it before moving to the next group.

        Args:
            prepared_data: Output from prepare_batch_data()
            all_activations: Output from forward_batch_through_layers()
            loss_scale: Scale factor for loss (for gradient accumulation)

        Returns:
            Total loss (scalar)
        """
        device = self.computation_device
        dtype = self.dtype
        cpu_dtype = self.activation_cpu_dtype
        dit = self.pipe.dit
        batch_size = prepared_data["batch_size"]

        if self.verbose:
            print(f"  Backward pass: {batch_size} images through {self.layer_offloader.num_groups} layer groups")
            self.profiler.log_memory("backward_start")

        num_groups = self.layer_offloader.num_groups

        # ============================================
        # Final layer forward + loss + backward
        # ============================================
        total_loss = 0.0
        grad_outputs = []  # Gradients w.r.t. final layer inputs

        with self.profiler.section("final_layer_loss"):
            self._move_dit_persistent_to_gpu()

            loss_pbar = tqdm(
                range(batch_size),
                desc="  Loss computation",
                leave=False,
                position=1,
            )
            for img_idx in loss_pbar:
                # Get final activation (output of last layer group)
                unified_final = all_activations[img_idx][-1].to(device, dtype=dtype)
                unified_final.requires_grad_(True)

                t_noisy = prepared_data["t_noisy"][img_idx].to(device, dtype=dtype)
                patch_metadata = prepared_data["patch_metadata"][img_idx]
                target = prepared_data["targets"][img_idx].to(device, dtype=dtype)

                # Final layer
                unified_out = dit.all_final_layer["2-1"](unified_final, t_noisy)

                # Unpatchify
                x_output = dit.unpatchify([unified_out[0]], patch_metadata.get("x_size"))[0]
                x_output = rearrange(x_output, "C B H W -> B C H W")
                model_output = -x_output

                # Flow matching loss
                loss = torch.nn.functional.mse_loss(model_output, target)
                scaled_loss = loss * loss_scale / batch_size  # Average over batch

                total_loss += loss.item() / batch_size
                loss_pbar.set_postfix({"loss": f"{total_loss:.4f}"})

                # Backward through final layer
                scaled_loss.backward()

                # Save gradient for this image in CPU dtype
                if unified_final.grad is not None:
                    grad_outputs.append(unified_final.grad.to(cpu_dtype).cpu())
                else:
                    raise RuntimeError(f"No gradient for image {img_idx}")

                # Clean up
                del unified_final, t_noisy, target, unified_out, x_output, model_output
            loss_pbar.close()

            # Accumulate gradients for persistent components
            for name, param in dit.named_parameters():
                if param.grad is not None and param.requires_grad:
                    if not name.startswith("layers."):
                        param_key = f"persistent.{name}"
                        grad_cpu = param.grad.detach().cpu()
                        if param_key in self.accumulated_grads:
                            self.accumulated_grads[param_key] += grad_cpu
                        else:
                            self.accumulated_grads[param_key] = grad_cpu
                        param.grad = None

        if self.verbose:
            self.profiler.log_memory("after_final_layer_backward")

        # ============================================
        # Backward through layer groups (reverse order)
        # ============================================
        # Process one image at a time through each layer group.
        # Use .clone() to ensure completely independent computation graphs.

        group_pbar = tqdm(
            list(reversed(range(num_groups))),
            desc="  Backward groups",
            leave=False,
            position=1,
        )
        for group_idx in group_pbar:
            group_pbar.set_postfix({"group": f"{num_groups - group_idx}/{num_groups}"})

            with self.profiler.section(f"layer_group_{group_idx}_bwd_load"):
                self.layer_offloader.load_group(group_idx)

            with self.profiler.section(f"layer_group_{group_idx}_backward"):
                new_grad_outputs = []

                img_pbar = tqdm(
                    range(batch_size),
                    desc=f"    Group {group_idx+1} backward",
                    leave=False,
                    position=2,
                )
                for img_idx in img_pbar:
                    # Clone to ensure independent graph (critical for correctness)
                    group_input = all_activations[img_idx][group_idx].clone().to(device, dtype=dtype)
                    group_input.requires_grad_(True)

                    freqs_cis = prepared_data["unified_freqs_cis"][img_idx].to(device)
                    t_noisy = prepared_data["t_noisy"][img_idx].to(device, dtype=dtype)

                    layer_kwargs = {
                        "attn_mask": None,
                        "freqs_cis": freqs_cis,
                        "adaln_input": t_noisy,
                    }

                    # Forward pass (creates fresh graph for this image)
                    group_output = self.layer_offloader.forward_through_group(
                        group_idx, group_input,
                        use_gradient_checkpointing=False,
                        **layer_kwargs,
                    )

                    # Backward pass
                    grad_output_gpu = grad_outputs[img_idx].to(device, dtype=dtype)
                    group_output.backward(grad_output_gpu)

                    # Save gradient for next group
                    if group_input.grad is not None:
                        new_grad_outputs.append(group_input.grad.to(cpu_dtype).cpu())
                    else:
                        new_grad_outputs.append(torch.zeros(
                            group_input.shape, dtype=cpu_dtype, device='cpu'
                        ))

                    # Clean up this image's tensors immediately
                    del group_input, group_output, freqs_cis, t_noisy, grad_output_gpu
                    torch.cuda.empty_cache()
                img_pbar.close()

                # Update grad_outputs for next layer group
                del grad_outputs
                grad_outputs = new_grad_outputs

                # Accumulate layer gradients (averaged over batch)
                for layer_idx, layer in enumerate(self.layer_offloader.groups[group_idx]):
                    layer_base_idx = group_idx * self.layer_offloader.layers_per_group + layer_idx
                    for name, param in layer.named_parameters():
                        if param.grad is not None:
                            param_key = f"layers.{layer_base_idx}.{name}"
                            grad_cpu = param.grad.detach().cpu()
                            if param_key in self.accumulated_grads:
                                self.accumulated_grads[param_key] += grad_cpu
                            else:
                                self.accumulated_grads[param_key] = grad_cpu
                            param.grad = None

            with self.profiler.section(f"layer_group_{group_idx}_bwd_offload"):
                self.layer_offloader.offload_group(group_idx)

            if self.verbose:
                self.profiler.log_memory(f"after_backward_group_{group_idx}")
        group_pbar.close()

        self._move_dit_persistent_to_cpu()

        return total_loss

    def process_image_batch(
        self,
        batch: List[Dict[str, Any]],
        loss_scale: float = 1.0,
    ) -> float:
        """
        Process a batch of images with optimized layer-group swapping.

        This is the main entry point for batch training. It:
        1. Prepares all data (text encoding, image encoding)
        2. Forwards all images through layer groups (one swap per group)
        3. Backwards all images through layer groups (one swap per group)

        Total swaps: 2 * num_groups (vs 2 * num_groups * batch_size without batching)

        Args:
            batch: List of dicts with 'image' and 'prompt' keys
            loss_scale: Scale for gradient accumulation

        Returns:
            Average loss over the batch
        """
        if self.verbose:
            self.profiler.log_memory("batch_processing_start")

        # Prepare all data
        with self.profiler.section("prepare_batch_data"):
            prepared_data = self.prepare_batch_data(batch)

        # Forward through layer groups
        with self.profiler.section("forward_through_layers"):
            final_outputs, all_activations = self.forward_batch_through_layers(prepared_data)

        # Backward through layer groups
        with self.profiler.section("backward_through_layers"):
            total_loss = self.backward_batch_through_layers(
                prepared_data, all_activations, loss_scale=loss_scale
            )

        # Clean up
        with self.profiler.section("cleanup"):
            del all_activations
            del final_outputs
            del prepared_data
            torch.cuda.empty_cache()
            gc.collect()

        if self.verbose:
            self.profiler.log_memory("batch_processing_end")

        return total_loss

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt using the text encoder"""
        device = self.computation_device

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        text_inputs = self.pipe.tokenizer(
            [prompt_text],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # Encode
        with torch.no_grad():
            # Handle FP8 weights - need to convert on the fly
            prompt_embeds = self.pipe.text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_masks,
                output_hidden_states=True,
            ).hidden_states[-2]

        # Extract non-padded embeddings
        embeddings = prompt_embeds[0][prompt_masks[0]]

        return embeddings

    def encode_image(self, image) -> torch.Tensor:
        """Encode an image using the VAE encoder"""
        device = self.computation_device

        # Preprocess image
        image_tensor = self.pipe.preprocess_image(image)
        image_tensor = image_tensor.to(device, dtype=self.dtype)

        with torch.no_grad():
            latents = self.pipe.vae_encoder(image_tensor)

        return latents

    def process_single_image(
        self,
        data: Dict[str, Any],
        loss_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single image through the full pipeline with layer group offloading.

        Returns a dict containing the loss and any other outputs.
        """
        device = self.computation_device
        dtype = self.dtype
        dit = self.pipe.dit

        # Extract data
        image = data["image"]
        prompt = data["prompt"]

        # Resize image to proper dimensions (divisible by 16, within max_pixels)
        image = self.image_resizer(image)
        height = image.size[1]
        width = image.size[0]

        if self.verbose:
            print(f"  Image size: {width}x{height}")

        # ============================================
        # Phase 1: Text encoding (text encoder on GPU)
        # ============================================
        if self.verbose:
            print("  Phase 1: Encoding text...")

        self._move_text_encoder_to_gpu()
        prompt_embeds = self.encode_prompt(prompt)
        self._move_text_encoder_to_cpu()

        # ============================================
        # Phase 2: Image encoding (VAE on GPU)
        # ============================================
        if self.verbose:
            print("  Phase 2: Encoding image...")

        self._move_vae_to_gpu()

        # Generate noise
        noise = self.pipe.generate_noise(
            (1, 16, height//8, width//8),
            rand_device="cpu",
            rand_torch_dtype=dtype,
        ).to(device)

        # Encode input image
        image_tensor = self.pipe.preprocess_image(image).to(device, dtype=dtype)
        with torch.no_grad():
            input_latents = self.pipe.vae_encoder(image_tensor)

        self._move_vae_to_cpu()

        # ============================================
        # Phase 3: DIT forward with layer groups
        # ============================================
        if self.verbose:
            print("  Phase 3: DIT forward pass...")

        # Move DIT persistent components to GPU
        self._move_dit_persistent_to_gpu()

        # Sample random timestep
        timestep = torch.randint(0, 1000, (1,), device=device)
        t = timestep / 1000.0

        # Create noisy latents (flow matching)
        latents = (1 - t.view(-1, 1, 1, 1)) * input_latents + t.view(-1, 1, 1, 1) * noise

        # Timestep embedding
        t_noisy = dit.t_embedder(1000 - timestep)

        # Patchify
        latents_reshaped = rearrange(latents, "B C H W -> C B H W")
        x, cap_feats, patch_metadata = dit.patchify_and_embed(
            [latents_reshaped], [prompt_embeds.to(device, dtype=dtype)]
        )
        x = x[0].to(dtype=dtype)
        cap_feats = cap_feats[0].to(dtype=dtype)

        # Embed and prepare x
        x = dit.all_x_embedder["2-1"](x)
        x[torch.cat(patch_metadata.get("x_pad_mask"))] = dit.x_pad_token.to(dtype=dtype, device=device)
        x_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata.get("x_pos_ids"), dim=0).to(device))
        x = rearrange(x, "L C -> 1 L C")
        x_freqs_cis = rearrange(x_freqs_cis, "L C -> 1 L C")

        # Noise refiner (small, on GPU)
        for layer in dit.noise_refiner:
            x = layer(x=x, attn_mask=None, freqs_cis=x_freqs_cis, adaln_input=t_noisy)

        # Cap embedder
        cap_feats = dit.cap_embedder(cap_feats)
        cap_feats[torch.cat(patch_metadata.get("cap_pad_mask"))] = dit.cap_pad_token.to(dtype=dtype, device=device)
        cap_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata.get("cap_pos_ids"), dim=0).to(device))
        cap_feats = rearrange(cap_feats, "L C -> 1 L C")
        cap_freqs_cis = rearrange(cap_freqs_cis, "L C -> 1 L C")

        # Context refiner
        for layer in dit.context_refiner:
            cap_feats = layer(x=cap_feats, attn_mask=None, freqs_cis=cap_freqs_cis)

        # Unified sequence
        unified = torch.cat([x, cap_feats], dim=1)
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)

        # ============================================
        # Phase 4: Main layers with MANUAL forward/backward
        # ============================================
        if self.verbose:
            print("  Phase 4: Main layers (layer-group training)...")

        # Store layer kwargs for backward pass
        layer_kwargs = {
            "attn_mask": None,
            "freqs_cis": unified_freqs_cis,
            "adaln_input": t_noisy,
        }

        # Forward through layer groups, saving activations
        # We'll do manual backward later
        activations = [unified.detach().cpu()]  # Save input to first group

        with torch.no_grad():
            for group_idx in range(self.layer_offloader.num_groups):
                self.layer_offloader.load_group(group_idx)

                unified = self.layer_offloader.forward_through_group(
                    group_idx, unified,
                    use_gradient_checkpointing=False,
                    **layer_kwargs,
                )

                # Save activation for backward (on CPU to save GPU memory)
                activations.append(unified.detach().cpu())

                self.layer_offloader.offload_group(group_idx)

                if self.verbose and group_idx == 0:
                    print_gpu_memory(f"    After group {group_idx}: ")

        # Move final output back to GPU for loss computation
        # This is the output of the last layer group, input to final layer
        unified_for_final = activations[-1].to(device, dtype=dtype)
        unified_for_final.requires_grad_(True)
        unified = unified_for_final  # Use this for final layer

        # ============================================
        # Phase 5: Final layer and loss
        # ============================================
        if self.verbose:
            print("  Phase 5: Final layer and loss...")

        # Final layer (keep on GPU, it's small)
        # unified_for_final has requires_grad=True
        unified_out = dit.all_final_layer["2-1"](unified_for_final, t_noisy)

        # Unpatchify
        x_output = dit.unpatchify([unified_out[0]], patch_metadata.get("x_size"))[0]
        x_output = rearrange(x_output, "C B H W -> B C H W")
        model_output = -x_output

        # Flow matching loss: predict the noise (velocity)
        # target = noise - input_latents (velocity from data to noise)
        target = noise - input_latents
        loss = torch.nn.functional.mse_loss(model_output, target)

        # Scale loss for gradient accumulation
        scaled_loss = loss * loss_scale

        # ============================================
        # Phase 6: Backward through final layer only
        # ============================================
        if self.verbose:
            print("  Phase 6: Backward through final layer...")

        # Backward through final layer to get gradient w.r.t. unified_for_final
        # Use retain_graph=False since we only need to backward once through this part
        scaled_loss.backward()

        # Accumulate gradients for persistent components (final layer, embedders, refiners)
        for name, param in dit.named_parameters():
            if param.grad is not None and param.requires_grad:
                # Only accumulate for non-layer params (persistent components)
                if not name.startswith("layers."):
                    param_key = f"persistent.{name}"
                    grad_cpu = param.grad.detach().cpu()
                    if param_key in self.accumulated_grads:
                        self.accumulated_grads[param_key] += grad_cpu
                    else:
                        self.accumulated_grads[param_key] = grad_cpu
                    param.grad = None

        # Get gradient of loss w.r.t. the output of the last layer group
        if unified_for_final.grad is not None:
            grad_output = unified_for_final.grad.detach().cpu()
        else:
            raise RuntimeError("No gradient computed for unified_for_final.")

        # ============================================
        # Phase 7: Manual backward through layer groups
        # ============================================
        if self.verbose:
            print("  Phase 7: Manual backward through layer groups...")

        # Detach layer_kwargs tensors since the original graph is freed
        # We need fresh tensors for the recomputed forward passes
        layer_kwargs_detached = {
            "attn_mask": None,
            "freqs_cis": unified_freqs_cis.detach(),
            "adaln_input": t_noisy.detach(),
        }

        # Manual backward through layer groups (reverse order)
        for group_idx in reversed(range(self.layer_offloader.num_groups)):
            self.layer_offloader.load_group(group_idx)

            # Get input activation for this group
            group_input = activations[group_idx].to(device, dtype=dtype)
            group_input.requires_grad_(True)

            # Recompute forward for this group (with detached kwargs)
            group_output = self.layer_offloader.forward_through_group(
                group_idx, group_input,
                use_gradient_checkpointing=False,
                **layer_kwargs_detached,
            )

            # Backward through this group
            grad_output_gpu = grad_output.to(device, dtype=dtype)
            group_output.backward(grad_output_gpu)

            # Save gradient for next group (becomes grad_output for previous group)
            grad_output = group_input.grad.detach().cpu() if group_input.grad is not None else None

            # IMPORTANT: Accumulate gradients to CPU storage before offloading
            # This preserves gradients across layer group offloading
            # Use full parameter names as keys (not ids, which change after device transfer)
            for layer_idx, layer in enumerate(self.layer_offloader.groups[group_idx]):
                layer_base_idx = group_idx * self.layer_offloader.layers_per_group + layer_idx
                for name, param in layer.named_parameters():
                    if param.grad is not None:
                        # Create a unique key using layer index and param name
                        param_key = f"layers.{layer_base_idx}.{name}"
                        grad_cpu = param.grad.detach().cpu()
                        if param_key in self.accumulated_grads:
                            self.accumulated_grads[param_key] += grad_cpu
                        else:
                            self.accumulated_grads[param_key] = grad_cpu
                        param.grad = None  # Clear to allow offloading

            self.layer_offloader.offload_group(group_idx)

            if self.verbose and group_idx == self.layer_offloader.num_groups - 1:
                print_gpu_memory(f"    After backward group {group_idx}: ")

        # Clear activations to free CPU memory
        del activations
        gc.collect()

        return {"loss": loss.detach(), "model_output": model_output.detach()}

    def forward(self, data: Dict[str, Any], loss_scale: float = 1.0) -> torch.Tensor:
        """Process a single data sample and return loss

        Args:
            data: Dict containing 'image' and 'prompt'
            loss_scale: Scale factor for loss (for gradient accumulation, use 1/accumulation_steps)
        """
        result = self.process_single_image(data, loss_scale=loss_scale)
        return result["loss"]

    def offload_dit_layers(self):
        """Offload all DIT layers to CPU after backward pass"""
        for group_idx in range(self.layer_offloader.num_groups):
            self.layer_offloader.offload_group(group_idx)
        self._move_dit_persistent_to_cpu()
        if self.verbose:
            print_gpu_memory("  After offloading DIT: ")

    def trainable_modules(self):
        """Return trainable parameters"""
        return filter(lambda p: p.requires_grad, self.pipe.dit.parameters())

    def is_training(self):
        """Check if DIT is in training mode"""
        return self.pipe.dit.training


def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image Layer Group Training")

    # Dataset args
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, default=None)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    # Model args
    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None)
    parser.add_argument("--model_base_path", type=str, default=None,
                        help="Custom base path for downloading/loading models (default: ./models). "
                             "Can also be set via DIFFSYNTH_MODEL_BASE_PATH environment variable.")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--trainable_models", type=str, default="dit")
    parser.add_argument("--fp8_models", type=str, default="text_encoder,vae")

    # Layer group args
    parser.add_argument("--num_layer_groups", type=int, default=6,
                        help="Number of layer groups (more groups = less GPU memory, more swaps)")
    parser.add_argument("--images_per_group_batch", type=int, default=4,
                        help="Images to process through each layer group before swapping (reduces swaps)")

    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=True)

    # AdamW optimizer args
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer-state-dtype", type=str, default="float32",
                        choices=["float16", "float32"],
                        help="Dtype for AdamW momentum/variance states on CPU (float32 recommended for stability)")
    parser.add_argument("--scale_lr_with_batch", action="store_true", default=False,
                        help="Scale learning rate linearly with images_per_group_batch (disabled by default)")

    # LR scheduler args
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="Learning rate scheduler type")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum LR as ratio of max LR (for cosine scheduler)")

    # Output args
    parser.add_argument("--output_path", type=str, default="./models/train/Z-Image_layer_group")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")

    # Resume args
    parser.add_argument("--continue_training", action="store_true", default=False,
                        help="Continue training from the latest checkpoint in output_path")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint (directory or training_state_*.pt file) to resume from")

    # Other
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable detailed profiling of GPU/CPU memory and timing")
    parser.add_argument("--profile_report_interval", type=int, default=10,
                        help="Print profiler report every N steps (only if --profile is enabled)")

    return parser.parse_args()


import gc  # Already in your imports, but ensure it's used

def save_training_state(
    output_path: str,
    global_step: int,
    epoch: int,
    model: 'ZImageLayerGroupTrainingModule',
    optimizer: CPUOffloadedAdamW,
    lr_scheduler: LRScheduler,
):
    """Save full training state for resuming with low memory usage, batched by layer group.

    Optimizer state files are saved as a single set (overwritten each save) since keeping
    multiple copies serves little purpose and wastes disk space. Model checkpoints are
    saved with step numbers to preserve training history.
    """
    # Ensure everything is offloaded to CPU
    model.offload_dit_layers()

    # Get layer group info from the offloader
    num_groups = model.layer_offloader.num_groups
    layers_per_group = model.layer_offloader.layers_per_group
    num_layers = model.layer_offloader.num_layers

    # Small dict for per-param steps (ints, negligible RAM)
    opt_steps = {k: optimizer.state[k]["step"] for k in optimizer.state}

    # Save optimizer states in batches (one per layer group) - single overwriting set
    for group_idx in range(num_groups):
        start_layer = group_idx * layers_per_group
        end_layer = min(start_layer + layers_per_group, num_layers)

        # Collect keys for this group (e.g., "layers.0.norm1.weight", etc.)
        group_keys = [
            k for k in optimizer.state
            if any(k.startswith(f"layers.{l}.") for l in range(start_layer, end_layer))
        ]

        if group_keys:
            flat_group = {}
            for k in group_keys:
                flat_group[f"{k}_exp_avg"] = optimizer.state[k]["exp_avg"]
                flat_group[f"{k}_exp_avg_sq"] = optimizer.state[k]["exp_avg_sq"]

            # No step number - overwrites each save
            opt_group_path = os.path.join(output_path, f"optimizer_group_{group_idx}.safetensors")
            save_file(flat_group, opt_group_path)
            print(f"Saved optimizer group {group_idx} to {opt_group_path}")

            del flat_group  # Release the dict immediately
            gc.collect()  # Force garbage collection to free any overhead

    # Save persistent params (small, separate file) - single overwriting file
    persistent_keys = [k for k in optimizer.state if k.startswith("persistent.")]
    if persistent_keys:
        flat_persistent = {}
        for k in persistent_keys:
            flat_persistent[f"{k}_exp_avg"] = optimizer.state[k]["exp_avg"]
            flat_persistent[f"{k}_exp_avg_sq"] = optimizer.state[k]["exp_avg_sq"]

        # No step number - overwrites each save
        opt_pers_path = os.path.join(output_path, f"optimizer_persistent.safetensors")
        save_file(flat_persistent, opt_pers_path)
        print(f"Saved optimizer persistent to {opt_pers_path}")

        del flat_persistent
        gc.collect()

    # Save small non-tensor state to .pt - single overwriting file
    small_state = {
        "global_step": global_step,
        "epoch": epoch,
        "optimizer": {
            "lr": optimizer.lr,
            "betas": optimizer.betas,
            "eps": optimizer.eps,
            "weight_decay": optimizer.weight_decay,
            "steps": opt_steps,  # Small dict of ints
        },
        "scheduler_state": lr_scheduler.state_dict(),
        "num_groups": num_groups,  # Store for loading
    }
    # Save as training_state_latest.pt (overwrites)
    state_path = os.path.join(output_path, "training_state_latest.pt")
    torch.save(small_state, state_path)
    print(f"Saved training state metadata to {state_path}")

    # Save model weights WITH step number (keep training history)
    model_state = {}
    for name, param in model.pipe.dit.named_parameters():
        model_state[name] = param.to("cpu")

    model_path = os.path.join(output_path, f"model_step_{global_step}.safetensors")
    save_file(model_state, model_path)
    print(f"Saved model to {model_path}")

    gc.collect()  # Final cleanup

def load_training_state(
    checkpoint_path: str,
    optimizer: CPUOffloadedAdamW,
    lr_scheduler: LRScheduler,
) -> Tuple[int, int]:
    """Load training state for resuming.

    Args:
        checkpoint_path: Either:
            - A directory containing training_state_latest.pt
            - A path to training_state_latest.pt directly
            - (Legacy) A path to training_state_step_N.pt
        optimizer: The optimizer to restore state into
        lr_scheduler: The LR scheduler to restore state into

    Returns:
        Tuple of (global_step, epoch)
    """
    # Handle directory path - look for training_state_latest.pt
    if os.path.isdir(checkpoint_path):
        state_file = os.path.join(checkpoint_path, "training_state_latest.pt")
        if not os.path.exists(state_file):
            raise FileNotFoundError(
                f"No training_state_latest.pt found in {checkpoint_path}. "
                "Make sure the directory contains a valid checkpoint."
            )
        checkpoint_path = state_file

    print(f"Loading training state from {checkpoint_path}...")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    small_state = torch.load(checkpoint_path, map_location="cpu")

    global_step = small_state["global_step"]

    # Reconstruct optimizer state
    opt_state = {}

    # Get num_groups from saved state (new format) or default to 6 (legacy)
    num_groups = small_state.get("num_groups", 6)

    # Load layer group batches - try new format first, then legacy
    for group_idx in range(num_groups):
        # New format: optimizer_group_N.safetensors (no step number)
        opt_group_path = os.path.join(checkpoint_dir, f"optimizer_group_{group_idx}.safetensors")

        # Legacy format: optimizer_group_N_step_XXX.safetensors
        if not os.path.exists(opt_group_path):
            legacy_path = os.path.join(checkpoint_dir, f"optimizer_group_{group_idx}_step_{global_step}.safetensors")
            if os.path.exists(legacy_path):
                opt_group_path = legacy_path

        if os.path.exists(opt_group_path):
            flat_group = load_file(opt_group_path)
            for key in [k for k in flat_group if k.endswith("_exp_avg")]:
                param_key = key.replace("_exp_avg", "")
                if param_key not in opt_state:
                    opt_state[param_key] = {}
                opt_state[param_key]["exp_avg"] = flat_group[key]
                opt_state[param_key]["exp_avg_sq"] = flat_group[f"{param_key}_exp_avg_sq"]
                opt_state[param_key]["step"] = small_state["optimizer"]["steps"][param_key]

    # Load persistent - try new format first, then legacy
    opt_pers_path = os.path.join(checkpoint_dir, "optimizer_persistent.safetensors")
    if not os.path.exists(opt_pers_path):
        legacy_path = os.path.join(checkpoint_dir, f"optimizer_persistent_step_{global_step}.safetensors")
        if os.path.exists(legacy_path):
            opt_pers_path = legacy_path

    if os.path.exists(opt_pers_path):
        flat_pers = load_file(opt_pers_path)
        for key in [k for k in flat_pers if k.endswith("_exp_avg")]:
            param_key = key.replace("_exp_avg", "")
            if param_key not in opt_state:
                opt_state[param_key] = {}
            opt_state[param_key]["exp_avg"] = flat_pers[key]
            opt_state[param_key]["exp_avg_sq"] = flat_pers[f"{param_key}_exp_avg_sq"]
            opt_state[param_key]["step"] = small_state["optimizer"]["steps"][param_key]

    # Full optimizer state dict
    full_opt_state = {
        "lr": small_state["optimizer"]["lr"],
        "betas": small_state["optimizer"]["betas"],
        "eps": small_state["optimizer"]["eps"],
        "weight_decay": small_state["optimizer"]["weight_decay"],
        "state": opt_state,
    }
    optimizer.load_state_dict(full_opt_state)

    lr_scheduler.load_state_dict(small_state["scheduler_state"])

    epoch = small_state["epoch"]

    print(f"Resumed from step {global_step}, epoch {epoch}")
    return global_step, epoch

def load_image_from_data(image_data, base_path: str):
    """Load an image from various input formats."""
    if isinstance(image_data, str):
        # If just filename, prepend the base path
        if not os.path.isabs(image_data) and not os.path.exists(image_data):
            image_data = os.path.join(base_path, image_data)
        return Image.open(image_data).convert("RGB")
    elif hasattr(image_data, 'convert'):
        # Already a PIL Image
        return image_data
    else:
        raise ValueError(f"Unexpected image type: {type(image_data)}")


def main():
    args = parse_args()

    # Set custom model base path if provided
    if args.model_base_path is not None:
        os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = args.model_base_path
        print(f"Using custom model base path: {args.model_base_path}")

    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("Z-Image Layer Group Training")
    print("=" * 60)

    # Optionally scale LR with effective batch size (disabled by default)
    if args.scale_lr_with_batch:
        baseline_batch = 1
        effective_multiplier = args.images_per_group_batch / baseline_batch
        args.learning_rate *= effective_multiplier
        print(f"Scaled LR to {args.learning_rate} for batch size {args.images_per_group_batch} (multiplier: {effective_multiplier:.1f}x)")
        # Also scale warmup (sqrt for stability)
        import math
        args.warmup_steps = int(args.warmup_steps * math.sqrt(effective_multiplier))
        print(f"Scaled warmup steps to {args.warmup_steps}")
    print(f"Layer groups: {args.num_layer_groups}")
    print(f"Images per group batch: {args.images_per_group_batch}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Max pixels: {args.max_pixels}")
    print(f"FP8 models: {args.fp8_models}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LR scheduler: {args.lr_scheduler} (warmup: {args.warmup_steps} steps)")
    print(f"AdamW: betas=({args.adam_beta1}, {args.adam_beta2}), eps={args.adam_eps}, wd={args.weight_decay}")
    print(f"Seed: {args.seed}")
    print(f"Profiling: {args.profile}")
    if args.continue_training:
        print(f"Continue training: will resume from {args.output_path} if checkpoint exists")
    elif args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 60)

    # Create dataset
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create training module
    model = ZImageLayerGroupTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        fp8_models=args.fp8_models,
        computation_device="cuda",
        num_layer_groups=args.num_layer_groups,
        images_per_group_batch=args.images_per_group_batch,
        max_pixels=args.max_pixels,
        verbose=args.verbose,
        profile=args.profile,
    )

    # Count trainable parameters
    trainable_params = list(model.trainable_modules())
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")

    # Calculate total training steps for scheduler
    steps_per_epoch = len(dataset) // args.images_per_group_batch
    total_training_steps = steps_per_epoch * args.num_epochs

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")

    # Create CPU-offloaded AdamW optimizer
    # Collect parameter names for tracking
    param_names = []
    for name, param in model.pipe.dit.named_parameters():
        if param.requires_grad:
            if name.startswith("layers."):
                param_names.append(name)
            else:
                param_names.append(f"persistent.{name}")

    # Parse optimizer state dtype
    state_dtype = torch.float16 if getattr(args, 'optimizer_state_dtype', 'float32') == "float16" else torch.float32

    optimizer = CPUOffloadedAdamW(
        param_groups=[{"param_names": param_names}],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        verbose=args.verbose,
        state_dtype=state_dtype,
    )

    # Create LR scheduler
    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=args.min_lr_ratio,
        scheduler_type=args.lr_scheduler,
    )

    print(f"Optimizer: CPUOffloadedAdamW (states on CPU RAM)")
    print(f"LR Scheduler: {args.lr_scheduler} with {args.warmup_steps} warmup steps")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    # Determine checkpoint path: --continue_training uses output_path, --resume_from_checkpoint uses explicit path
    checkpoint_path = None
    if args.continue_training:
        checkpoint_path = args.output_path  # Directory path - load_training_state handles this
    elif args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint

    if checkpoint_path:
        # Check if checkpoint exists before trying to load
        if os.path.isdir(checkpoint_path):
            state_file = os.path.join(checkpoint_path, "training_state_latest.pt")
            if not os.path.exists(state_file):
                if args.continue_training:
                    print(f"No checkpoint found in {checkpoint_path}, starting fresh training")
                    checkpoint_path = None
                else:
                    raise FileNotFoundError(f"No training_state_latest.pt found in {checkpoint_path}")

        if checkpoint_path:
            global_step, start_epoch = load_training_state(
                checkpoint_path,
                optimizer,
                lr_scheduler,
            )

    # Create model logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Calculate swap efficiency
    naive_swaps = args.images_per_group_batch * args.num_layer_groups * 2  # forward + backward
    optimized_swaps = args.num_layer_groups * 2  # forward + backward
    print(f"Swap reduction: {naive_swaps} -> {optimized_swaps} per batch ({naive_swaps / optimized_swaps:.1f}x improvement)")
    print("=" * 60)

    model.pipe.dit.train()

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Shuffle dataset indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Process in batches of images_per_group_batch
        num_batches = len(indices) // args.images_per_group_batch

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")

        for batch_idx in progress_bar:
            # Collect batch of images
            start_idx = batch_idx * args.images_per_group_batch
            end_idx = start_idx + args.images_per_group_batch
            batch_indices = indices[start_idx:end_idx]

            batch = []
            skip_batch = False

            for idx in batch_indices:
                sample = dataset[idx]
                try:
                    image = load_image_from_data(
                        sample["image"][0] if isinstance(sample["image"], list) else sample["image"],
                        args.dataset_base_path
                    )
                    prompt = sample["prompt"][0] if isinstance(sample["prompt"], list) else sample["prompt"]
                    batch.append({"image": image, "prompt": prompt})
                except Exception as e:
                    print(f"Warning: Failed to load image at index {idx}: {e}")
                    skip_batch = True
                    break

            if skip_batch or len(batch) == 0:
                continue

            try:
                # Process batch with optimized layer-group swapping
                # This does forward and backward for all images with minimal swaps
                loss_scale = 1.0 / args.gradient_accumulation_steps
                batch_loss = model.process_image_batch(batch, loss_scale=loss_scale)

                # Optimizer step after each batch (or after accumulation if gradient_accumulation_steps > 1)
                # For simplicity, we step after each batch here since batching already provides
                # effective gradient accumulation
                if args.gradient_accumulation_steps == 1 or (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Run optimizer step (loads groups one at a time)
                    model.run_optimizer_step(optimizer)

                    # Update learning rate
                    lr_scheduler.step()

                    global_step += 1

                    # Logging
                    current_lr = lr_scheduler.get_lr()
                    progress_bar.set_postfix({
                        "loss": f"{batch_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step
                    })

                    # Print profiler report at intervals
                    if args.profile and global_step % args.profile_report_interval == 0:
                        model.profiler.report()
                        model.profiler.reset()

                    # Save checkpoint
                    if global_step % args.save_steps == 0:
                        print(f"\nSaving checkpoint at step {global_step}...")
                        save_training_state(
                            args.output_path,
                            global_step,
                            epoch,
                            model,
                            optimizer,
                            lr_scheduler,
                        )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at batch {batch_idx}! Clearing cache and continuing...")
                    print(f"Consider reducing --images_per_group_batch (currently {args.images_per_group_batch})")
                    torch.cuda.empty_cache()
                    gc.collect()
                    model.accumulated_grads.clear()
                    continue
                else:
                    raise e

    print("\nTraining complete!")

    # Final profiler report
    if args.profile:
        print("\nFinal profiler report:")
        model.profiler.report()

    # Final save
    print("Saving final checkpoint...")
    save_training_state(
        args.output_path,
        global_step,
        args.num_epochs,
        model,
        optimizer,
        lr_scheduler,
    )
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
