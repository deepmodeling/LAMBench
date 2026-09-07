import logging
import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.units import fs

from lambench.models.ase_models import ASEModel
from lambench.tasks.calculator.nve_md.nve_md_data import TEST_DATA


def run_md_nve_simulation(
    model: ASEModel,
    num_steps: int,
    timestep: float,
    temperature_K: int,
    test_data: list[Atoms] = TEST_DATA,
) -> dict[str, float]:
    """
    This function runs NVE simulations for a list of test systems using the given model.
    """
    results = {}
    for atoms in test_data:
        result = nve_simulation_single(
            atoms,
            model.calc,
            num_steps=num_steps,
            timestep=timestep,
            temperature_K=temperature_K,
        )
        results[str(atoms.symbols)] = result
        logging.info(f"Simulation completed for system {atoms.symbols!s}: {result}")
    return results


def nve_simulation_single(
    atoms: Atoms,
    calculator: Calculator,
    num_steps: int,
    timestep: float,
    temperature_K: int,
):
    """
    Run an NVE simulation using VelocityVerlet and return performance metrics.

    Parameters:
        atoms: ASE Atoms objects for simulation.
        calculator: ASE calculator to use for the simulation.
        num_steps (int): Number of steps to run.
        timestep (float): Time step in fs.
        temperature_K (int): Temperature in Kelvin.

    Returns:
        dict: A dictionary containing:
            - 'simulation_time': Time taken for the simulation (s).
            - 'steps': Total steps completed (int).
            - 'momenta_diff': Average momenta difference (AMU \u00b7 \u00c5/fs).
            - 'slope': Energy drift per step (eV/atom/ps).
    """
    LOG_INTERVAL = max(1, num_steps // 100)
    WARMUP_STEPS = int(0.2 * num_steps)
    WARMUP_STEPS = (WARMUP_STEPS // LOG_INTERVAL) * LOG_INTERVAL

    atoms.calc = calculator
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(0)
    )
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = VelocityVerlet(atoms, timestep * fs)

    # Track energies and steps
    energies = []

    def log_energy(a=atoms):
        energies.append(a.get_total_energy())
        if energies[-1] > 1e10:
            # To allow for early stopping in case of divergence
            raise RuntimeError

    dyn.attach(log_energy, interval=LOG_INTERVAL)

    # Measure performance
    start_time = time.time()
    try:
        dyn.run(num_steps)
    except Exception as e:
        logging.error(f"Simulation crashed after {dyn.nsteps} steps: {e}")
    end_time = time.time()

    # Compute metrics
    simulation_time = end_time - start_time

    slope = None
    # Perform linear fit on energies using np.linalg.lstsq
    if energies:
        warmup_idx = WARMUP_STEPS // LOG_INTERVAL
        if warmup_idx < len(energies) and len(energies) - warmup_idx > 1:
            steps_after_warmup = (
                np.arange(0, len(energies) - warmup_idx) * LOG_INTERVAL + WARMUP_STEPS
            )
            times = steps_after_warmup * timestep * fs
            A = np.vstack([times, np.ones(len(times))]).T
            energies_after_warmup = energies[warmup_idx:]
            slope, _ = np.linalg.lstsq(A, energies_after_warmup, rcond=None)[0]
    try:
        momenta_diff = np.linalg.norm(atoms.get_momenta().sum(axis=0))
    except Exception:
        momenta_diff = None
    return {
        "simulation_time": simulation_time,  # Simulation efficiency, s
        "steps": dyn.nsteps,  # Simulation stability
        "momenta_diff": momenta_diff,  # Momentum conservation, AMU · Å/fs
        "slope": np.abs(1000 * slope / len(atoms))
        if slope is not None
        else None,  # Energy drift, eV/atom/ps
    }
