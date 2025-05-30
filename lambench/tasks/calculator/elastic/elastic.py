# ruff: noqa: E402
"""
This module has been modified from MatCalc
https://github.com/materialsvirtuallab/matcalc/blob/main/src/matcalc/_elasticity.py

https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE

BSD 3-Clause License

Copyright (c) 2023, Materials Virtual Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
The test data is obtained from the following paper:

de Jong, M., Chen, W., Angsten, T. et al. Charting the complete elastic properties of inorganic crystalline compounds.
Sci Data 2, 150009 (2015). https://doi.org/10.1038/sdata.2015.9

"""

import json
from ase.io import read
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO
from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.analysis.elasticity.elastic import get_strain_state_dict
from pymatgen.io.ase import AseAtomsAdaptor
from lambench.models.ase_models import ASEModel
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import logging

EV_A3_TO_GPA = 160.21766208  # eV/Å³ to GPa


def run_inference(
    model: ASEModel,
    test_data: Path,
    fmax: float,
    max_steps: int,
) -> dict[str, float]:
    with open(test_data, "r") as f:
        data = json.load(f)

    TOTAL_STRUCTURES = len(data)
    SUCCESS_STRUCTURES = 0
    g_vrh_label = []
    k_vrh_label = []
    g_vrh_pred = []
    k_vrh_pred = []

    for idx, atom_info in enumerate(data):
        try:
            g_vrh_pred, k_vrh_pred = get_elastic_for_one(
                model, atom_info, fmax, max_steps
            )
            g_vrh_pred.append(g_vrh_pred)
            k_vrh_pred.append(k_vrh_pred)
            g_vrh_label.append(atom_info["G_VRH"])
            k_vrh_label.append(atom_info["K_VRH"])
            SUCCESS_STRUCTURES += 1
        except Exception as e:
            logging.error(f"Error processing structure {idx}: {e}")

    results = {
        "success_rate": SUCCESS_STRUCTURES / TOTAL_STRUCTURES,
        "MAE_G_VRH": mean_absolute_error(np.array(g_vrh_label), np.array(g_vrh_pred)),
        "MAE_K_VRH": mean_absolute_error(np.array(k_vrh_label), np.array(k_vrh_pred)),
    }
    return results


def get_elastic_for_one(
    model: ASEModel, atom_info: dict, fmax: float, max_steps: int
) -> dict[str, float]:
    """
    Calculate the elastic properties for one structure.
    """
    atoms = read(StringIO(atom_info["poscar"]), format="vasp")

    relaxed_atoms = model.run_ase_relaxation(
        atoms=atoms,
        calc=model.calc,
        fmax=fmax,
        steps=max_steps,
        fix_symmetry=False,
        relax_cell=True,
    )
    structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
    deformed_structure_set = DeformedStructureSet(
        structure,
        np.linspace(-0.01, 0.01, 4),
        np.linspace(-0.06, 0.06, 4),
    )
    stresses = []
    for deformed_structure in deformed_structure_set:
        atoms = deformed_structure.to_ase_atoms()
        atoms.calc = model.calc
        stresses.append(atoms.get_stress(voigt=False))

    strains = [
        Strain.from_deformation(deformation)
        for deformation in deformed_structure_set.deformations
    ]
    eq_stress = relaxed_atoms.get_stress(voigt=False)
    elastic_tensor = get_elastic_tensor_from_strains(
        strains=strains,
        stresses=stresses,
        eq_stress=eq_stress,
    )
    return elastic_tensor.g_vrh * EV_A3_TO_GPA, elastic_tensor.k_vrh * EV_A3_TO_GPA


def get_elastic_tensor_from_strains(
    self,
    strains: ArrayLike,
    stresses: ArrayLike,
    eq_stress: ArrayLike = None,
    tol: float = 1e-7,
) -> ElasticTensor:
    """
    Compute the elastic tensor from given strain and stress data using least-squares
    fitting.

    This function calculates the elastic constants from strain-stress relations,
    using a least-squares fitting procedure for each independent component of stress
    and strain tensor pairs. An optional equivalent stress array can be supplied.
    Residuals from the fitting process are accumulated and returned alongside the
    elastic tensor. The elastic tensor is zeroed according to the given tolerance.
    """

    strain_states = [tuple(ss) for ss in np.eye(6)]
    ss_dict = get_strain_state_dict(
        strains, stresses, eq_stress=eq_stress, add_eq=self.use_equilibrium
    )
    c_ij = np.zeros((6, 6))
    for ii in range(6):
        strain = ss_dict[strain_states[ii]]["strains"]
        stress = ss_dict[strain_states[ii]]["stresses"]
        for jj in range(6):
            fit = np.polyfit(strain[:, ii], stress[:, jj], 1, full=True)
            c_ij[ii, jj] = fit[0][0]
    elastic_tensor = ElasticTensor.from_voigt(c_ij)
    return elastic_tensor.zeroed(tol)
