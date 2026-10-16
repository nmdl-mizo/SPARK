from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch


########################################################
##### 1D empirical distribution
########################################################


def inverse_transform(
    energy_grid: torch.Tensor,
    densities: torch.Tensor,
    n_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample continuous energy points per site via inverse-transform sampling,
    and return corresponding spectral areas.

    Parameters:
    -----------
    energy_grid : Tensor, shape (M,) or (sites, M)
        Energy grid points.
    densities : Tensor, shape (M,) or (sites, M)
        DOS densities for each site.
    n_samples : int or None
        Number of samples per site (defaults to M).

    Returns:
    --------
    samples_all : Tensor, shape (sites, n_samples) or (n_samples,)
        Deterministic samples at CDF quantiles for each site.
    areas : Tensor, shape (sites,) or scalar
        Spectral area (trapezoidal integral) per site, for later scaling.
    """
    single_site = False
    if energy_grid.ndim == 1:
        energy_grid = energy_grid.unsqueeze(0)
        densities = densities.unsqueeze(0)
        single_site = True

    if energy_grid.shape != densities.shape:
        raise ValueError(
            f"energy_grid shape {tuple(energy_grid.shape)} must match "
            f"densities shape {tuple(densities.shape)}"
        )

    n_sites, n_grid = densities.shape
    if n_samples is None:
        n_samples = n_grid

    densities = torch.clamp(densities, min=0.0)
    area_zero = torch.zeros(n_sites, dtype=energy_grid.dtype, device=energy_grid.device)
    areas = torch.trapezoid(densities, energy_grid, dim=1)

    samples_list = []
    quantiles = (
        (torch.arange(n_samples, device=energy_grid.device, dtype=energy_grid.dtype) + 0.5)
        / n_samples
    )

    for site_idx in range(n_sites):
        site_energy = energy_grid[site_idx]
        site_density = densities[site_idx]
        density_sum = site_density.sum()

        if density_sum <= 0:
            midpoint = 0.5 * (site_energy[0] + site_energy[-1])
            samples = torch.full(
                (n_samples,),
                midpoint,
                dtype=site_energy.dtype,
                device=site_energy.device,
            )
            samples_list.append(samples)
            areas[site_idx] = area_zero[site_idx]
            continue

        probabilities = site_density / density_sum
        cdf = torch.cumsum(probabilities, dim=0)
        # cdf[-1] = 1.0

        idx = torch.searchsorted(cdf, quantiles, right=False)
        idx = torch.clamp(idx, max=n_grid - 1)
        left_idx = torch.clamp(idx - 1, min=0)

        cdf_left = torch.where(idx > 0, cdf[left_idx], torch.zeros_like(quantiles))
        cdf_right = cdf[idx]
        energy_left = site_energy[left_idx]
        energy_right = site_energy[idx]

        denom = torch.clamp(cdf_right - cdf_left, min=torch.finfo(site_energy.dtype).eps)
        weight = torch.where(
            idx > 0,
            (quantiles - cdf_left) / denom,
            torch.zeros_like(quantiles),
        )
        samples = energy_left + weight * (energy_right - energy_left)
        samples_list.append(samples)

    samples_all = torch.stack(samples_list, dim=0)
    if single_site:
        return samples_all[0], areas[0]
    return samples_all, areas


def _compute_adaptive_sigmas(
    samples: torch.Tensor,
    sigma_scale: float,
    sigma_min: float,
    sigma_max: Optional[float],
) -> torch.Tensor:
    sorted_samples, _ = torch.sort(samples, dim=1)
    n_sites, n_samp = sorted_samples.shape

    if n_samp == 1:
        local_spacing = torch.full_like(sorted_samples, sigma_min)
    else:
        left_spacing = sorted_samples[:, 1:] - sorted_samples[:, :-1]
        right_spacing = left_spacing.clone()

        local_spacing = torch.empty_like(sorted_samples)
        local_spacing[:, 0] = right_spacing[:, 0]
        local_spacing[:, -1] = left_spacing[:, -1]

        if n_samp > 2:
            local_spacing[:, 1:-1] = 0.5 * (left_spacing[:, :-1] + right_spacing[:, 1:])

    eps = torch.finfo(samples.dtype).eps
    sigmas = sigma_scale * torch.clamp(local_spacing, min=max(sigma_min, eps))
    sigmas = torch.clamp(sigmas, min=sigma_min)
    if sigma_max is not None:
        sigmas = torch.clamp(sigmas, max=sigma_max)
    return sigmas


def compute_adaptive_sigmas(
    samples: torch.Tensor,
    sigma_scale: float,
    sigma_min: float,
    sigma_max: Optional[float],
) -> torch.Tensor:
    return _compute_adaptive_sigmas(
        samples=samples,
        sigma_scale=sigma_scale,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def reconstruct_from_samples(
    energy_grid: torch.Tensor,
    samples: torch.Tensor,
    sigma: float,
    areas: Optional[torch.Tensor] = None,
    adaptive_broadening: bool = False,
    sigma_scale: float = 1.0,
    sigma_min: float = 1e-3,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Reconstruct DOS from sample points per site using Gaussian KDE (PyTorch version),
    scaling to provided spectral areas.

    When adaptive broadening is enabled, each sample uses its own Gaussian width
    derived from the local sample spacing.
    """
    single_site = False
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
        single_site = True

    if energy_grid.dim() == 1:
        energy_grid = energy_grid.unsqueeze(0).expand(samples.size(0), -1)

    n_sites, n_grid = energy_grid.size()

    samples_exp = samples.unsqueeze(-1)
    grid_exp = energy_grid.view(n_sites, 1, n_grid)

    diff = grid_exp - samples_exp
    if adaptive_broadening:
        sigma_values = _compute_adaptive_sigmas(
            samples=samples,
            sigma_scale=sigma_scale,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    else:
        sigma_values = torch.full_like(samples, sigma)

    sigma_exp = sigma_values.unsqueeze(-1)
    gauss = torch.exp(-(diff.pow(2)) / (2 * sigma_exp.pow(2)))
    gauss = gauss / (sigma_exp * math.sqrt(2 * math.pi))

    spectrum = gauss.mean(dim=1)

    if areas is not None:
        if areas.dim() == 0:
            areas = areas.repeat(n_sites)
        spectrum = spectrum * areas.view(n_sites, 1)

    if single_site:
        return spectrum[0]
    return spectrum


def reconstruct_from_samples_torch(
    energy_grid: torch.Tensor,
    samples: torch.Tensor,
    sigma: float,
    areas: Optional[torch.Tensor] = None,
    adaptive_broadening: bool = False,
    sigma_scale: float = 1.0,
    sigma_min: float = 1e-3,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    return reconstruct_from_samples(
        energy_grid=energy_grid,
        samples=samples,
        sigma=sigma,
        areas=areas,
        adaptive_broadening=adaptive_broadening,
        sigma_scale=sigma_scale,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def interpolate_spectrum(
    raw_energy: torch.Tensor,
    raw_mu: torch.Tensor,
    *,
    target_grid: Optional[torch.Tensor] = None,
    n_points: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_energy = raw_energy.to(dtype=dtype, device=device)
    raw_mu = raw_mu.to(dtype=dtype, device=device)

    if target_grid is None:
        if n_points is None:
            n_points = 3 * raw_energy.numel()
        energy_grid = torch.linspace(raw_energy.min(), raw_energy.max(), n_points, dtype=dtype, device=device)
    else:
        energy_grid = target_grid.to(dtype=dtype, device=device)

    upper_idx = torch.searchsorted(raw_energy, energy_grid, right=False)
    upper_idx = torch.clamp(upper_idx, min=1, max=raw_energy.numel() - 1)
    lower_idx = upper_idx - 1

    lower_energy = raw_energy[lower_idx]
    upper_energy = raw_energy[upper_idx]
    lower_mu = raw_mu[lower_idx]
    upper_mu = raw_mu[upper_idx]

    weight = (energy_grid - lower_energy) / torch.clamp(
        upper_energy - lower_energy,
        min=torch.finfo(dtype).eps,
    )
    interpolated_mu = lower_mu + weight * (upper_mu - lower_mu)

    return energy_grid, interpolated_mu


def process_xmu_dat(
    xmu_path: str | Path,
    *,
    target_grid: Optional[torch.Tensor] = None,
    n_points: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read xmu.dat, extract FEFF energy/mu columns, interpolate them to a
    uniform energy grid, and return the tensors expected by inverse_transform.

    Returns:
        interpolated_energy_grid, interpolated_mu
    """
    raw_energy_values = []
    raw_mu_values = []

    with Path(xmu_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            try:
                raw_energy_values.append(float(parts[0]))
                raw_mu_values.append(float(parts[3]))
            except ValueError:
                continue

    if not raw_energy_values:
        raise ValueError(f"No valid xmu.dat rows found in {xmu_path}")

    raw_energy = torch.tensor(raw_energy_values, dtype=dtype, device=device)
    raw_mu = torch.tensor(raw_mu_values, dtype=dtype, device=device)
    return interpolate_spectrum(
        raw_energy,
        raw_mu,
        target_grid=target_grid,
        n_points=n_points,
        dtype=dtype,
        device=device,
    )
