import torch

from geomed_copilot.component_patches import (
    directions_in_normalized_frame,
    extract_component_patches,
)


def test_pixel_direction_is_converted_for_anisotropic_image():
    direction = torch.tensor([[[1.0, 1.0]]])
    converted = directions_in_normalized_frame(direction, torch.tensor([[0.69314718]]))
    expected = torch.tensor([.5, 1.0]); expected = expected / expected.norm()
    assert torch.allclose(converted[0, 0], expected, atol=1e-5)


def test_axis_center_maps_to_patch_center():
    image = torch.zeros(1, 1, 65, 65)
    image[0, 0, 32, 32] = 1
    patch = extract_component_patches(
        image, torch.tensor([[[.5, .5]]]), torch.tensor([[[0., 1.]]]),
        torch.zeros(1, 1), output_size=33)
    maximum = patch[0, 0, 0].argmax()
    assert divmod(maximum.item(), 33) == (16, 16)


def test_horizontal_axis_is_rotated_to_vertical_patch_evidence():
    horizontal = torch.zeros(1, 1, 65, 65)
    horizontal[0, 0, 32, 12:53] = 1
    patch = extract_component_patches(
        horizontal, torch.tensor([[[.5, .5]]]), torch.tensor([[[1., 0.]]]),
        torch.zeros(1, 1), output_size=33)[0, 0, 0]
    vertical_energy = patch[:, 16].sum()
    horizontal_energy = patch[16].sum()
    assert vertical_energy > 3 * horizontal_energy
