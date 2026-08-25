"""Differentiable component-aligned image evidence extraction."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def directions_in_normalized_frame(directions: torch.Tensor,
                                   log_aspect: torch.Tensor) -> torch.Tensor:
    """Convert pixel-frame directions to normalized image-coordinate directions."""
    aspect = log_aspect.exp().reshape(-1, 1, 1)
    converted = torch.stack([directions[..., 0] / aspect[..., 0],
                             directions[..., 1]], dim=-1)
    return converted / converted.norm(dim=-1, keepdim=True).clamp_min(1e-7)


def extract_component_patches(images: torch.Tensor, centers: torch.Tensor,
                              directions: torch.Tensor, log_aspect: torch.Tensor,
                              output_size: int = 32, half_width: float = .13,
                              half_length: float = .28) -> torch.Tensor:
    """Return axis-centered patches with every predicted axis aligned vertically.

    Args:
        images: ``(B,C,H,W)`` square-resized images.
        centers: ``(B,K,2)`` normalized x/y component centers.
        directions: ``(B,K,2)`` unit directions in original pixel coordinates.
        log_aspect: ``(B,1)`` log(original width / original height).
    Returns:
        Tensor shaped ``(B,K,C,output_size,output_size)``.
    """
    if images.ndim != 4 or centers.ndim != 3 or directions.shape != centers.shape:
        raise ValueError("expected images (B,C,H,W) and centers/directions (B,K,2)")
    batch, components, _ = centers.shape
    if len(images) != batch or log_aspect.shape[0] != batch:
        raise ValueError("batch dimensions must agree")
    axis = directions_in_normalized_frame(directions, log_aspect)
    perpendicular = torch.stack([axis[..., 1], -axis[..., 0]], dim=-1)
    coordinates = torch.linspace(-1, 1, output_size, device=images.device,
                                 dtype=images.dtype)
    local_y, local_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    offsets = (local_x[None, None, ..., None] * half_width *
               perpendicular[:, :, None, None, :] +
               local_y[None, None, ..., None] * half_length *
               axis[:, :, None, None, :])
    grid_normalized = centers[:, :, None, None, :] + offsets
    grid = grid_normalized.mul(2).sub(1).reshape(
        batch * components, output_size, output_size, 2)
    repeated_images = images[:, None].expand(-1, components, -1, -1, -1).reshape(
        batch * components, images.shape[1], images.shape[2], images.shape[3])
    patches = F.grid_sample(repeated_images, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
    return patches.reshape(batch, components, images.shape[1], output_size, output_size)
