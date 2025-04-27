import logging
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import dpdata
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE
from tqdm import tqdm

from lambench.models.basemodel import BaseLargeAtomModel

# pyright: reportMissingImports=false


class ASEModel(BaseLargeAtomModel):
    """
    A specialized atomic simulation model that extends BaseLargeAtomModel to provide
    a unified interface for ASE-compatible calculators. This class dynamically selects
    and instantiates the appropriate calculator based on the model family attribute and
    facilitates various simulation tasks including energy, force, and virial evaluations,
    as well as structural relaxation.

    Attributes:
        calc (Calculator): A cached property that initializes and returns an ASE Calculator
            instance based on the model family. Depending on self.model_family, it creates:
                - MACE Calculator using mace_mp (for "MACE"),
                - ORB Calculator via ORBCalculator (for "ORB"),
                - SevenNet Calculator (for "SevenNet"),
                - OCPCalculator (for "EquiformerV2"),
                - MatterSimCalculator (for "MatterSim"),
                - DP calculator (for "DP").
            Note: one should implement the corresponding calculator classes when adding new models to the benchmark.

    Methods:
        evaluate(task) -> Optional[dict[str, float]]:
            Evaluates a given computational task. The method supports:
                - Direct prediction tasks (using DirectPredictTask) by resetting pytorch dtype
                  and calling run_ase_dptest.
                - Calculator-based simulation tasks (using CalculatorTask) such as:
                    - "nve_md": runs an NVE molecular dynamics simulation.
                    - "phonon_mdr": runs a phonon simulation.
                    - "inference_efficiency": runs an inference efficiency test.
            Note: one should implement the corresponding task methods when adding new tasks to the benchmark.

        run_ase_dptest(calc: Calculator, test_data: Path) -> dict:
            A static method that processes test data by iterating over atomic systems and frames.
            It calculates energy, force, and virial properties, handling potential errors during
            energy computation and logging any failures. It returns a dictionary containing the
            mean absolute error (MAE) and root mean square error (RMSE) for energy (both overall and
            per atom), and, if available, for force and virial terms.

        run_ase_relaxation(atoms: Atoms, calc: Calculator, fmax: float = 5e-3, steps: int = 500,
                           fix_symmetry: bool = True, relax_cell: bool = True) -> Optional[Atoms]:
            A static method that relaxes an atomic structure using the FIRE optimizer. It optionally
            applies symmetry constraints and cell relaxation. In case of an exception during the
            relaxation process, the method logs the error and returns None.

    Usage:
        The ASEModel class is designed to abstract the complexity involved in setting up diverse
        atomic simulation tasks. It enables simulation and evaluation workflows by automatically
        selecting the correct ASE calculator based on the model's attributes and by providing
        utility methods to run direct prediction tests and relaxation simulations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def calc(self, head=None) -> Calculator:
        """ASE Calculator with the model loaded."""
        calculator_dispatch = {
            "MACE": self._init_mace_calculator,
            "ORB": self._init_orb_calculator,
            "SevenNet": self._init_sevennet_calculator,
            "Equiformer": self._init_equiformer_calculator,
            "MatterSim": self._init_mattersim_calculator,
            "DP": self._init_dp_calculator,
            "GRACE": self._init_grace_calculator,
            "PET-MAD": self._init_petmad_calculator,
        }

        if self.model_family not in calculator_dispatch:
            raise ValueError(f"Model {self.model_name} is not supported by ASEModel")

        return calculator_dispatch[self.model_family]()

    def _init_mace_calculator(self) -> Calculator:
        from mace.calculators import mace_mp

        return mace_mp(
            model=self.model_name.split("_")[-1],
            device="cuda",
            default_dtype="float64",
        )

    def _init_orb_calculator(self) -> Calculator:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.ORB_PRETRAINED_MODELS[self.model_name.replace("_", "-")](
            device="cuda"
        )
        return ORBCalculator(orbff, device="cuda")

    def _init_sevennet_calculator(self) -> Calculator:
        from sevenn.sevennet_calculator import SevenNetCalculator

        model_config = {"model": self.model_name, "device": "cuda"}
        if self.model_name == "7net-mf-ompa":
            model_config["modal"] = "mpa"
        return SevenNetCalculator(**model_config)

    def _init_equiformer_calculator(self) -> Calculator:
        from fairchem.core import OCPCalculator

        return OCPCalculator(
            checkpoint_path=self.model_path,
            cpu=False,
        )

    def _init_mattersim_calculator(self) -> Calculator:
        from mattersim.forcefield import MatterSimCalculator

        return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")

    def _init_dp_calculator(self) -> Calculator:
        from deepmd.calculator import DP

        return DP(
            model=self.model_path,
            head="MP_traj_v024_alldata_mixu",
        )

    def _init_grace_calculator(self) -> Calculator:
        from tensorpotential.calculator import grace_fm

        return grace_fm(
            self.model_name,
            pad_neighbors_fraction=0.05,
            pad_atoms_number=2,
            min_dist=0.5,
        )

    def _init_petmad_calculator(self) -> Calculator:
        from pet_mad.calculator import PETMADCalculator

        return PETMADCalculator(checkpoint_path=str(self.model_path), device="cuda")

    def evaluate(
        self, task
    ) -> dict[str, dict[str, float]] | dict[str, dict[str, dict[str, float]]]:
        # DirectPredictTask | CalculatorTask
        from lambench.tasks.calculator.calculator_tasks import CalculatorTask
        from lambench.tasks.direct.direct_tasks import DirectPredictTask

        if isinstance(task, DirectPredictTask):
            # Reset the default dtype to float32 to avoid type mismatch
            import torch

            torch.set_default_dtype(torch.float32)
            return self.run_ase_dptest(self.calc, task.test_data)
        elif isinstance(task, CalculatorTask):
            if task.task_name == "nve_md":
                from lambench.tasks.calculator.nve_md.nve_md import (
                    run_md_nve_simulation,
                )

                num_steps = task.calculator_params.get("num_steps", 1000)
                timestep = task.calculator_params.get("timestep", 1.0)
                temperature_K = task.calculator_params.get("temperature_K", 300)
                return {
                    "metrics": run_md_nve_simulation(
                        self, num_steps, timestep, temperature_K
                    )
                }
            elif task.task_name == "phonon_mdr":
                from lambench.tasks.calculator.phonon.phonon import (
                    run_phonon_simulation,
                )

                assert task.test_data is not None
                task.workdir.mkdir(exist_ok=True)
                distance = task.calculator_params.get("distance", 0.01)
                return {
                    "metrics": run_phonon_simulation(
                        self, task.test_data, distance, task.workdir
                    )
                }
            elif task.task_name == "inference_efficiency":
                from lambench.tasks.calculator.inference_efficiency.inference_efficiency import (
                    run_inference,
                )

                assert task.test_data is not None
                warmup_ratio = task.calculator_params.get("warmup_ratio", 0.2)
                return {"metrics": run_inference(self, task.test_data, warmup_ratio)}
            elif task.task_name == "torsionnet":
                from lambench.tasks.calculator.torsionnet.torsionnet import (
                    run_torsionnet,
                )

                assert task.test_data is not None
                return {"metrics": run_torsionnet(self, task.test_data)}
            else:
                raise NotImplementedError(f"Task {task.task_name} is not implemented.")

        else:
            raise NotImplementedError(
                f"Task {task.task_name} is not implemented for ASEModel."
            )

    @staticmethod
    def run_ase_dptest(calc: Calculator, test_data: Path) -> dict:
        energy_err = []
        energy_pre = []
        energy_lab = []
        atom_num = []
        energy_err_per_atom = []
        force_err = []
        virial_err = []
        virial_err_per_atom = []
        max_ele_num = 120
        failed_structures = []
        failed_tolereance = 10
        systems = [i.parent for i in test_data.rglob("type_map.raw")]
        assert systems, f"No systems found in the test data {test_data}."
        mix_type = any(systems[0].rglob("real_atom_types.npy"))

        for filepth in tqdm(systems, desc="Systems"):
            if mix_type:
                sys = dpdata.MultiSystems()
                sys.load_systems_from_file(filepth, fmt="deepmd/npy/mixed")
            else:
                sys = dpdata.LabeledSystem(filepth, fmt="deepmd/npy")
            for ls in tqdm(sys, desc="Set", leave=False):
                for frame in tqdm(ls, desc="Frames", leave=False):
                    atoms: Atoms = frame.to_ase_structure()[0]
                    atoms.calc = calc

                    # Energy
                    try:
                        energy_predict = np.array(atoms.get_potential_energy())
                        if not np.isfinite(energy_predict):
                            raise ValueError("Energy prediction is non-finite.")
                    except (ValueError, RuntimeError):
                        file = Path(
                            f"failed_structures/{calc.name}/{atoms.symbols}.cif"
                        )
                        file.parent.mkdir(parents=True, exist_ok=True)
                        write(file, atoms)
                        logging.error(
                            f"Error in energy prediction; CIF file saved as {file}."
                        )
                        failed_structures.append(atoms.symbols)
                        if len(failed_structures) > failed_tolereance:
                            logging.error(f"Failed structures: {failed_structures}")
                            raise RuntimeError("Too many failures; aborting.")
                        continue  # skip this frame
                    energy_pre.append(energy_predict)
                    energy_lab.append(frame.data["energies"])
                    energy_err.append(energy_predict - frame.data["energies"])
                    energy_err_per_atom.append(energy_err[-1] / len(atoms))
                    atomic_numbers = atoms.get_atomic_numbers()
                    atom_num.append(np.bincount(atomic_numbers, minlength=max_ele_num))

                    # Force
                    try:
                        force_pred = atoms.get_forces()
                        force_err.append(
                            frame.data["forces"].squeeze(0) - np.array(force_pred)
                        )
                    except KeyError as _:  # no force in the data
                        pass

                    # Virial
                    try:
                        stress = atoms.get_stress()
                        virial_tensor = (
                            -np.array(
                                [
                                    [stress[0], stress[5], stress[4]],
                                    [stress[5], stress[1], stress[3]],
                                    [stress[4], stress[3], stress[2]],
                                ]
                            )
                            * atoms.get_volume()
                        )
                        virial_err.append(frame.data["virials"] - virial_tensor)
                        virial_err_per_atom.append(
                            virial_err[-1] / force_err[-1].shape[0]
                        )
                    except (
                        NotImplementedError,  # atoms.get_stress() for eqv2
                        ValueError,  # atoms.get_volume()
                        KeyError,  # frame.data["virials"]
                    ) as _:  # no virial in the data
                        pass

        if failed_structures:
            logging.error(f"Failed structures: {failed_structures}")
        atom_num = np.array(atom_num)
        energy_err = np.array(energy_err)
        energy_pre = np.array(energy_pre)
        energy_lab = np.array(energy_lab)
        shift_bias, _, _, _ = np.linalg.lstsq(atom_num, energy_err, rcond=1e-10)
        unbiased_energy = (
            energy_pre
            - (atom_num @ shift_bias.reshape(max_ele_num, -1)).reshape(-1)
            - energy_lab.squeeze()
        )
        unbiased_energy_err_per_a = unbiased_energy / atom_num.sum(-1)

        res = {
            "energy_mae": [np.mean(np.abs(np.stack(unbiased_energy)))],
            "energy_rmse": [np.sqrt(np.mean(np.square(unbiased_energy)))],
            "energy_mae_natoms": [np.mean(np.abs(np.stack(unbiased_energy_err_per_a)))],
            "energy_rmse_natoms": [
                np.sqrt(np.mean(np.square(unbiased_energy_err_per_a)))
            ],
        }
        if force_err:
            res.update(
                {
                    "force_mae": [np.mean(np.abs(np.concatenate(force_err)))],
                    "force_rmse": [
                        np.sqrt(np.mean(np.square(np.concatenate(force_err))))
                    ],
                }
            )
        if virial_err_per_atom:
            res.update(
                {
                    "virial_mae": [np.mean(np.abs(np.stack(virial_err)))],
                    "virial_rmse": [np.sqrt(np.mean(np.square(np.stack(virial_err))))],
                    "virial_mae_natoms": [
                        np.mean(np.abs(np.stack(virial_err_per_atom)))
                    ],
                    "virial_rmse_natoms": [
                        np.sqrt(np.mean(np.square(np.stack(virial_err_per_atom))))
                    ],
                }
            )
        return res

    @staticmethod
    def run_ase_relaxation(
        atoms: Atoms,
        calc: Calculator,
        fmax: float = 5e-3,
        steps: int = 500,
        fix_symmetry: bool = True,
        relax_cell: bool = True,
        observer: Optional[Callable] = None,
    ) -> Optional[Atoms]:
        atoms.calc = calc
        if fix_symmetry:
            atoms.set_constraint(FixSymmetry(atoms))
        if relax_cell:
            atoms = FrechetCellFilter(atoms)
        opt = FIRE(atoms, trajectory=None, logfile=None)
        if observer:
            opt.insert_observer(observer, atoms=opt.atoms)
        try:
            opt.run(fmax=fmax, steps=steps)
        except Exception as e:
            logging.error(f"Relaxation failed: {e}")
            return None
        if relax_cell:
            atoms = atoms.atoms
        return atoms
