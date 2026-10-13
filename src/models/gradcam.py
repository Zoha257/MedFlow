from __future__ import annotations

import torch
from torch import nn


class GradCAM:
    """Compute Grad-CAM maps for a convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_inputs: tuple[torch.Tensor | None, ...],
        grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        self.gradients = grad_outputs[0]

    def __call__(self, inputs: torch.Tensor, target_class: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        score = logits[:, target_class].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        minimum = cam.amin(dim=(2, 3), keepdim=True)
        maximum = cam.amax(dim=(2, 3), keepdim=True)
        return (cam - minimum) / (maximum - minimum).clamp_min(1e-8)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
