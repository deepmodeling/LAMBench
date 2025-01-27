from functools import cached_property
import logging
from pathlib import Path
from typing import Optional

import ase
import dpdata
import numpy as np
from ase.calculators.calculator import Calculator
from ase.io import write
from tqdm import tqdm

from lambench.models.basemodel import BaseLargeAtomModel


class ASEModel(BaseLargeAtomModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def calc(self, head=None) -> Calculator:
        """ASE Calculator with the model loaded."""

        if self.model_family == "MACE":
            from mace.calculators import mace_mp

            return mace_mp(model="medium", device="cuda", default_dtype="float64")
        elif self.model_family == "ORB":
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            orbff = pretrained.orb_v2(device="cuda")  # orb-v2-20241011.ckpt
            return ORBCalculator(orbff, device="cuda")
        elif self.model_family == "SevenNet":
            from sevenn.sevennet_calculator import SevenNetCalculator

            return SevenNetCalculator("7net-0_11July2024", device="cuda")
        elif self.model_family == "EquiformerV2":
            from fairchem.core import OCPCalculator

            return OCPCalculator(
                checkpoint_path=self.model_path,
                # Model retrieved from https://huggingface.co/fairchem/OMAT24#model-checkpoints with agreement with the license
                # NOTE: check the list of public model at https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/pretrained_models.yml
                # Uncomment the following lines to use one:
                # model_name="EquiformerV2-153M-S2EF-OC20-All+MD",
                # local_cache=str(Path.home().joinpath(".cache")),
                cpu=False,
            )
        elif self.model_family == "MatterSim":
            from mattersim.forcefield import MatterSimCalculator

            return MatterSimCalculator(
                load_path="MatterSim-v1.0.0-5M.pth", device="cuda"
            )
        elif self.model_family == "DP":
            from deepmd.calculator import DP

            return DP(
                model=self.model_path,
                head="MP_traj_v024_alldata_mixu",
            )
        else:
            raise ValueError(f"Model {self.model_name} is not supported by ASEModel")

    def evaluate(self, task) -> Optional[dict[str, float]]:
        from lambench.tasks.calculator.calculator_tasks import CalculatorTask
        from lambench.tasks.direct.direct_tasks import DirectPredictTask

        if isinstance(task, DirectPredictTask):
            # Reset the default dtype to float32 to avoid type mismatch
            import torch

            torch.set_default_dtype(torch.float32)
            return self.run_ase_dptest(self.calc, task.test_data)
        elif isinstance(task, CalculatorTask):
            if task.task_name == "nve_md":
                from lambench.tasks.calculator.nve_md import run_md_nve_simulation

                num_steps = task.calculator_params.get("num_steps", 1000)
                timestep = task.calculator_params.get("timestep", 1.0)
                temperature_K = task.calculator_params.get("temperature_K", 300)
                return {
                    "metrics": run_md_nve_simulation(
                        self, num_steps, timestep, temperature_K
                    )
                }
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
            for ls in tqdm(sys, desc="Set", leave=False):  # type: ignore
                for frame in tqdm(ls, desc="Frames", leave=False):
                    atoms: ase.Atoms = frame.to_ase_structure()[0]  # type: ignore
                    atoms.calc = calc

                    # Energy
                    try:
                        energy_predict = np.array(atoms.get_potential_energy())
                        if not np.isfinite(energy_predict):
                            raise ValueError("Energy prediction is non-finite.")
                    except (ValueError, RuntimeError):
                        file_name = f"failed_structures/{calc.name}/{atoms.symbols}.cif"
                        write(file_name, atoms)
                        logging.error(
                            f"Error in energy prediction; CIF file saved as {file_name}."
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
                        stress_tensor = (
                            -np.array(
                                [
                                    [stress[0], stress[5], stress[4]],
                                    [stress[5], stress[1], stress[3]],
                                    [stress[4], stress[3], stress[2]],
                                ]
                            )
                            * atoms.get_volume()
                        )
                        virial_err.append(frame.data["virials"] - stress_tensor)
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
            "energy_mae": [np.mean(np.abs(np.stack(unbiased_energy)))],  # type: ignore
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
