from __future__ import annotations

import math

import torch

from spark.data import energy_reference_for_z
from spark.predict import grid, reconstruct, uncertainty
from spark.utils import inverse_transform, reconstruct_from_samples


def test_population_uncertainty():
    energy = torch.linspace(0, 2, 5).expand(2, -1)
    spectra = torch.arange(8, dtype=torch.float32)[:, None, None].expand(8, 2, 5)
    areas = torch.arange(8, dtype=torch.float32)[:, None].expand(8, 2)
    metrics = uncertainty(energy, spectra, areas)
    expected = torch.full((2,), math.sqrt(5.25))
    torch.testing.assert_close(metrics["unc_mean_std"], expected)
    torch.testing.assert_close(metrics["unc_rms_std"], expected)
    torch.testing.assert_close(metrics["unc_max_std"], expected)
    torch.testing.assert_close(metrics["unc_integrated_var"], torch.full((2,), 10.5))
    torch.testing.assert_close(metrics["area_unc_std"], expected)
    identical = uncertainty(energy, torch.ones_like(spectra), torch.ones_like(areas))
    assert torch.count_nonzero(identical["unc_rms_std"]) == 0


def test_sample_area_scaling():
    energy = torch.linspace(-10, 10, 10001)
    mu = torch.exp(-energy.square() / 2) / math.sqrt(2 * math.pi)
    samples, area = inverse_transform(energy, 7 * mu, n_samples=512)
    shape, unit_area = inverse_transform(energy, mu, n_samples=512)
    torch.testing.assert_close(samples, shape, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(area, 7 * unit_area)
    reconstructed = reconstruct_from_samples(energy, samples, sigma=0.3, areas=area)
    torch.testing.assert_close(torch.trapezoid(reconstructed, energy), area)
    assert bool((samples[1:] >= samples[:-1]).all())


def test_element_energy_scaling():
    centers, scales = energy_reference_for_z(torch.tensor([27, 78]))
    samples = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    expected = torch.tensor([[7710.0, 7745.0, 7780.0], [11550.0, 11585.0, 11620.0]])
    torch.testing.assert_close(samples * scales + centers, expected)


def test_batched_reconstruction():
    samples = torch.linspace(7710, 7780, 512).expand(8, 10, -1).clone()
    samples += torch.arange(8)[:, None, None] * 0.1
    areas = torch.arange(1, 9, dtype=torch.float32)[:, None].expand(8, 10)
    energy = grid(samples)
    result = reconstruct(energy, samples, areas)
    torch.testing.assert_close(energy[:, 0], torch.full((10,), 7710.0))
    torch.testing.assert_close(energy[:, -1], torch.full((10,), 7779.7))
    direct = reconstruct_from_samples(energy[0], samples[3, 9], sigma=0.3,
        areas=areas[3, 9], adaptive_broadening=True, sigma_scale=1.0, sigma_min=0.3, sigma_max=1.0)
    torch.testing.assert_close(result[3, 9], direct)
