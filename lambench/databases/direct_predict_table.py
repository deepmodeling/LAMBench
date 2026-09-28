import numpy as np
from sqlalchemy import Column, Float

from lambench.databases.base_table import BaseRecord


class DirectPredictRecord(BaseRecord):
    __tablename__ = "direct_predict"
    # Results are stored in eV, not meV
    energy_rmse = Column(Float)
    energy_mae = Column(Float)
    energy_rmse_natoms = Column(Float)
    energy_mae_natoms = Column(Float)
    force_rmse = Column(Float)
    force_mae = Column(Float)
    virial_rmse = Column(Float)
    virial_mae = Column(Float)
    virial_rmse_natoms = Column(Float)
    virial_mae_natoms = Column(Float)

    def to_dict(self, ev_to_mev: bool = False) -> dict:
        output = {
            "energy_rmse": self.energy_rmse,
            "energy_mae": self.energy_mae,
            "energy_rmse_natoms": self.energy_rmse_natoms,
            "energy_mae_natoms": self.energy_mae_natoms,
            "force_rmse": self.force_rmse,
            "force_mae": self.force_mae,
            "virial_rmse": self.virial_rmse,
            "virial_mae": self.virial_mae,
            "virial_rmse_natoms": self.virial_rmse_natoms,
            "virial_mae_natoms": self.virial_mae_natoms,
        }
        if ev_to_mev:
            output = {
                k: np.round(v * 1000, 1) if v is not None else None
                for k, v in output.items()
            }  # Convert to meV
        return output
