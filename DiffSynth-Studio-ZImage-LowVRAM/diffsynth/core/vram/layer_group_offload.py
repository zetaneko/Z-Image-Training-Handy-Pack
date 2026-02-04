"""
Layer Group Offloading for Low-VRAM Training

This module implements batched layer-group training where:
1. Layers are split into groups (e.g., 6 groups of 5 layers each)
2. Each group is loaded to GPU, processes ALL images in batch, then offloaded
3. Boundary activations are stored on CPU between groups
4. This amortizes the CPU↔GPU transfer cost across many images

Memory profile (for 30-layer DIT with 6 groups):
- GPU: ~8-10GB (5 layers + gradients + activations)
- CPU RAM: ~4-6GB (boundary activations for batch)
- Swaps: 12 per batch (vs 60 per image with naive approach)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import gc


@dataclass
class LayerGroupConfig:
    """Configuration for layer group offloading"""
    num_groups: int = 6
    offload_device: str = "cpu"
    pin_memory: bool = True
    preserve_rng_state: bool = True
    verbose: bool = False


@dataclass
class BoundaryActivation:
    """Stores activation at boundary between layer groups"""
    tensor: torch.Tensor
    requires_grad: bool
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, offload_device: str = "cpu", pin_memory: bool = True):
        """Save tensor to CPU, optionally with pinned memory for faster transfers"""
        cpu_tensor = tensor.detach().cpu()
        if pin_memory and cpu_tensor.is_pinned() == False:
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except:
                pass  # pin_memory may fail on some systems
        return cls(
            tensor=cpu_tensor,
            requires_grad=tensor.requires_grad,
            device=tensor.device,
            dtype=tensor.dtype,
        )

    def restore(self, device: torch.device = None) -> torch.Tensor:
        """Restore tensor to original device"""
        target_device = device or self.device
        tensor = self.tensor.to(target_device, non_blocking=True)
        if self.requires_grad:
            tensor = tensor.requires_grad_(True)
        return tensor


class LayerGroup:
    """Manages a group of layers that are loaded/offloaded together"""

    def __init__(
        self,
        layers: nn.ModuleList,
        group_idx: int,
        offload_device: str = "cpu",
        computation_device: str = "cuda",
        computation_dtype: torch.dtype = torch.bfloat16,
    ):
        self.layers = layers
        self.group_idx = group_idx
        self.offload_device = offload_device
        self.computation_device = computation_device
        self.computation_dtype = computation_dtype
        self.is_loaded = False

    def load(self):
        """Load all layers in this group to GPU"""
        if self.is_loaded:
            return
        for layer in self.layers:
            layer.to(device=self.computation_device, dtype=self.computation_dtype)
        self.is_loaded = True

    def offload(self):
        """Offload all layers in this group to CPU"""
        if not self.is_loaded:
            return
        for layer in self.layers:
            layer.to(device=self.offload_device)
        self.is_loaded = False
        # Clear GPU cache after offload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through all layers in this group"""
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def __len__(self):
        return len(self.layers)

    def parameters(self):
        """Iterate over all parameters in this group"""
        for layer in self.layers:
            yield from layer.parameters()

    def named_parameters(self):
        """Iterate over all named parameters in this group"""
        for layer in self.layers:
            yield from layer.named_parameters()


class LayerGroupManager:
    """
    Manages splitting a model's layers into groups for memory-efficient training.

    This enables training large models on limited VRAM by:
    1. Only keeping one layer group on GPU at a time
    2. Processing entire batches through each group before swapping
    3. Storing boundary activations on CPU
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        config: LayerGroupConfig = None,
        computation_device: str = "cuda",
        computation_dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config or LayerGroupConfig()
        self.computation_device = computation_device
        self.computation_dtype = computation_dtype

        # Split layers into groups
        self.groups = self._create_groups(layers)
        self.num_groups = len(self.groups)

        # Initially offload all groups
        for group in self.groups:
            group.offload()

        if self.config.verbose:
            print(f"LayerGroupManager: Created {self.num_groups} groups from {len(layers)} layers")
            for i, group in enumerate(self.groups):
                print(f"  Group {i}: {len(group)} layers")

    def _create_groups(self, layers: nn.ModuleList) -> List[LayerGroup]:
        """Split layers into groups"""
        num_layers = len(layers)
        group_size = (num_layers + self.config.num_groups - 1) // self.config.num_groups

        groups = []
        for i in range(0, num_layers, group_size):
            group_layers = nn.ModuleList(list(layers[i:i + group_size]))
            group = LayerGroup(
                layers=group_layers,
                group_idx=len(groups),
                offload_device=self.config.offload_device,
                computation_device=self.computation_device,
                computation_dtype=self.computation_dtype,
            )
            groups.append(group)

        return groups

    @contextmanager
    def group_context(self, group_idx: int):
        """Context manager that loads a group and offloads it when done"""
        group = self.groups[group_idx]
        try:
            group.load()
            yield group
        finally:
            group.offload()

    def get_group(self, group_idx: int) -> LayerGroup:
        return self.groups[group_idx]


class BatchedLayerGroupForward(torch.autograd.Function):
    """
    Custom autograd function for batched layer group forward/backward.

    This handles:
    1. Forward: Process input through layer group, save boundary for backward
    2. Backward: Reload group, recompute forward (checkpointing), compute gradients
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        layer_group: LayerGroup,
        group_manager: LayerGroupManager,
        layer_kwargs: Dict[str, Any],
    ):
        ctx.layer_group = layer_group
        ctx.group_manager = group_manager
        ctx.layer_kwargs = layer_kwargs

        # Save input for backward (on CPU to save GPU memory)
        ctx.save_for_backward(input_tensor.detach().cpu())
        ctx.input_device = input_tensor.device
        ctx.input_dtype = input_tensor.dtype
        ctx.input_requires_grad = input_tensor.requires_grad

        # Forward pass (no grad to save memory, will recompute in backward)
        with torch.no_grad():
            output = layer_group.forward(input_tensor, **layer_kwargs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Restore input
        input_cpu, = ctx.saved_tensors
        input_tensor = input_cpu.to(ctx.input_device, dtype=ctx.input_dtype)
        if ctx.input_requires_grad:
            input_tensor = input_tensor.requires_grad_(True)

        layer_group = ctx.layer_group

        # Ensure group is loaded
        layer_group.load()

        # Recompute forward with grad enabled (gradient checkpointing style)
        with torch.enable_grad():
            output = layer_group.forward(input_tensor, **ctx.layer_kwargs)

        # Backward through this group
        if ctx.input_requires_grad:
            output.backward(grad_output)
            grad_input = input_tensor.grad
        else:
            grad_input = None

        return grad_input, None, None, None


class LayerGroupTrainer:
    """
    High-level trainer that implements batched layer group training.

    Usage:
        trainer = LayerGroupTrainer(model.dit, num_groups=6)

        for batch in dataloader:
            # Process entire batch through all layer groups
            loss = trainer.forward_backward(batch, loss_fn)
            optimizer.step()
    """

    def __init__(
        self,
        dit_model: nn.Module,
        num_groups: int = 6,
        computation_device: str = "cuda",
        computation_dtype: torch.dtype = torch.bfloat16,
        offload_device: str = "cpu",
        pin_memory: bool = True,
        verbose: bool = True,
    ):
        self.dit = dit_model
        self.computation_device = computation_device
        self.computation_dtype = computation_dtype
        self.verbose = verbose

        # Create config
        config = LayerGroupConfig(
            num_groups=num_groups,
            offload_device=offload_device,
            pin_memory=pin_memory,
            verbose=verbose,
        )

        # Create layer group manager for the main transformer layers
        if hasattr(dit_model, 'layers'):
            self.layer_manager = LayerGroupManager(
                layers=dit_model.layers,
                config=config,
                computation_device=computation_device,
                computation_dtype=computation_dtype,
            )
        else:
            raise ValueError("DIT model must have 'layers' attribute (nn.ModuleList of transformer blocks)")

        # Handle refiner layers separately (they're small, keep on GPU or manage separately)
        self.has_refiners = hasattr(dit_model, 'noise_refiner') and hasattr(dit_model, 'context_refiner')

        # Track components that should stay on GPU (small components)
        self.persistent_components = []
        for name in ['t_embedder', 'cap_embedder', 'rope_embedder', 'all_x_embedder',
                     'all_final_layer', 'x_pad_token', 'cap_pad_token']:
            if hasattr(dit_model, name):
                component = getattr(dit_model, name)
                if component is not None:
                    self.persistent_components.append((name, component))

        if verbose:
            print(f"LayerGroupTrainer initialized:")
            print(f"  - Main layers: {len(dit_model.layers)} layers in {num_groups} groups")
            print(f"  - Persistent components: {[name for name, _ in self.persistent_components]}")
            if self.has_refiners:
                print(f"  - Has refiner layers (will be managed separately)")

    def prepare_persistent_components(self):
        """Move small persistent components to GPU"""
        for name, component in self.persistent_components:
            if isinstance(component, nn.Module):
                component.to(self.computation_device, dtype=self.computation_dtype)
            elif isinstance(component, nn.Parameter):
                component.data = component.data.to(self.computation_device, dtype=self.computation_dtype)

    def offload_persistent_components(self):
        """Move persistent components to CPU (if needed for extreme memory savings)"""
        for name, component in self.persistent_components:
            if isinstance(component, nn.Module):
                component.to("cpu")
            elif isinstance(component, nn.Parameter):
                component.data = component.data.to("cpu")

    def forward_through_groups(
        self,
        x: torch.Tensor,
        layer_kwargs: Dict[str, Any] = None,
        store_boundaries: bool = True,
    ) -> Tuple[torch.Tensor, List[BoundaryActivation]]:
        """
        Forward pass through all layer groups.

        Args:
            x: Input tensor
            layer_kwargs: Keyword arguments to pass to each layer
            store_boundaries: Whether to store boundary activations (needed for backward)

        Returns:
            output: Final output tensor
            boundaries: List of boundary activations (empty if store_boundaries=False)
        """
        layer_kwargs = layer_kwargs or {}
        boundaries = []

        for group_idx in range(self.layer_manager.num_groups):
            # Store boundary activation before this group
            if store_boundaries and group_idx > 0:
                boundaries.append(BoundaryActivation.from_tensor(
                    x,
                    offload_device=self.layer_manager.config.offload_device,
                    pin_memory=self.layer_manager.config.pin_memory
                ))

            # Process through this group
            with self.layer_manager.group_context(group_idx) as group:
                x = group.forward(x, **layer_kwargs)

        return x, boundaries

    def forward_backward_through_groups(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable,
        layer_kwargs: Dict[str, Any] = None,
    ) -> torch.Tensor:
        """
        Combined forward and backward pass with layer group offloading.

        This is more memory efficient than separate forward/backward because
        we can discard activations after each group's backward pass.

        Args:
            x: Input tensor
            target: Target for loss computation
            loss_fn: Loss function that takes (output, target) -> loss
            layer_kwargs: Keyword arguments to pass to each layer

        Returns:
            loss: The computed loss value
        """
        layer_kwargs = layer_kwargs or {}
        num_groups = self.layer_manager.num_groups

        # Phase 1: Forward pass, storing boundary activations
        boundaries = [BoundaryActivation.from_tensor(x)]  # Store initial input

        with torch.no_grad():
            current = x
            for group_idx in range(num_groups):
                with self.layer_manager.group_context(group_idx) as group:
                    current = group.forward(current, **layer_kwargs)
                    if group_idx < num_groups - 1:
                        boundaries.append(BoundaryActivation.from_tensor(current))

        # Compute loss with the final output
        output = current
        loss = loss_fn(output, target)

        # Phase 2: Backward pass through groups in reverse
        grad_output = torch.autograd.grad(loss, output, retain_graph=False)[0]

        for group_idx in reversed(range(num_groups)):
            with self.layer_manager.group_context(group_idx) as group:
                # Restore input for this group
                group_input = boundaries[group_idx].restore(self.computation_device)
                group_input.requires_grad_(True)

                # Recompute forward
                with torch.enable_grad():
                    group_output = group.forward(group_input, **layer_kwargs)

                # Backward through this group
                group_output.backward(grad_output)

                # Gradient for previous group
                if group_idx > 0:
                    grad_output = group_input.grad

        return loss


class ZImageLayerGroupTrainer(LayerGroupTrainer):
    """
    Specialized trainer for Z-Image DIT model with layer group offloading.

    Handles the specific architecture:
    - noise_refiner (2 layers)
    - context_refiner (2 layers)
    - layers (30 main transformer layers) <- these are grouped
    - Various embedders
    """

    def __init__(
        self,
        dit_model: nn.Module,
        num_groups: int = 6,
        computation_device: str = "cuda",
        computation_dtype: torch.dtype = torch.bfloat16,
        refiner_on_gpu: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            dit_model=dit_model,
            num_groups=num_groups,
            computation_device=computation_device,
            computation_dtype=computation_dtype,
            verbose=verbose,
        )

        self.refiner_on_gpu = refiner_on_gpu

        # Create separate managers for refiners if they exist
        if self.has_refiners:
            if not refiner_on_gpu:
                # Treat refiners as additional groups
                self.noise_refiner_group = LayerGroup(
                    layers=dit_model.noise_refiner,
                    group_idx=-2,
                    offload_device="cpu",
                    computation_device=computation_device,
                    computation_dtype=computation_dtype,
                )
                self.context_refiner_group = LayerGroup(
                    layers=dit_model.context_refiner,
                    group_idx=-1,
                    offload_device="cpu",
                    computation_device=computation_device,
                    computation_dtype=computation_dtype,
                )
            else:
                # Keep refiners on GPU (they're small)
                dit_model.noise_refiner.to(computation_device, dtype=computation_dtype)
                dit_model.context_refiner.to(computation_device, dtype=computation_dtype)

    def forward_z_image(
        self,
        x: torch.Tensor,
        cap_feats: torch.Tensor,
        t_noisy: torch.Tensor,
        freqs_cis: torch.Tensor,
        cap_freqs_cis: torch.Tensor,
        use_gradient_checkpointing: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through Z-Image DIT with layer group offloading.

        This mirrors the structure of model_fn_z_image_turbo but with
        layer group management.
        """
        dit = self.dit

        # Ensure persistent components are on GPU
        self.prepare_persistent_components()

        # Noise refiner (small, keep on GPU or load temporarily)
        if self.has_refiners:
            if self.refiner_on_gpu:
                for layer in dit.noise_refiner:
                    x = layer(x=x, attn_mask=None, freqs_cis=freqs_cis, adaln_input=t_noisy)
            else:
                self.noise_refiner_group.load()
                for layer in self.noise_refiner_group.layers:
                    x = layer(x=x, attn_mask=None, freqs_cis=freqs_cis, adaln_input=t_noisy)
                self.noise_refiner_group.offload()

        # Context refiner
        if self.has_refiners:
            if self.refiner_on_gpu:
                for layer in dit.context_refiner:
                    cap_feats = layer(x=cap_feats, attn_mask=None, freqs_cis=cap_freqs_cis)
            else:
                self.context_refiner_group.load()
                for layer in self.context_refiner_group.layers:
                    cap_feats = layer(x=cap_feats, attn_mask=None, freqs_cis=cap_freqs_cis)
                self.context_refiner_group.offload()

        # Unified sequence
        unified = torch.cat([x, cap_feats], dim=1)
        unified_freqs_cis = torch.cat([freqs_cis, cap_freqs_cis], dim=1)

        # Main layers with group offloading
        layer_kwargs = {
            "attn_mask": None,
            "freqs_cis": unified_freqs_cis,
            "adaln_input": t_noisy,
        }

        unified, _ = self.forward_through_groups(
            unified,
            layer_kwargs=layer_kwargs,
            store_boundaries=self.training,
        )

        return unified

    @property
    def training(self):
        return self.dit.training


def create_layer_group_trainer(
    pipe,
    num_groups: int = 6,
    verbose: bool = True,
) -> ZImageLayerGroupTrainer:
    """
    Factory function to create a layer group trainer from a Z-Image pipeline.

    Usage:
        from diffsynth.core.vram.layer_group_offload import create_layer_group_trainer

        trainer = create_layer_group_trainer(pipe, num_groups=6)
    """
    if not hasattr(pipe, 'dit') or pipe.dit is None:
        raise ValueError("Pipeline must have a 'dit' model")

    device = str(pipe.device)
    dtype = pipe.torch_dtype

    return ZImageLayerGroupTrainer(
        dit_model=pipe.dit,
        num_groups=num_groups,
        computation_device=device,
        computation_dtype=dtype,
        verbose=verbose,
    )
