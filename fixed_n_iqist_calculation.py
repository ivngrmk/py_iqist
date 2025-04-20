from typing import Self

from .iqist_calculation import iQISTCalculation
import numpy as np
import os

class FixedNiQISTCalculation(iQISTCalculation):
    @classmethod
    def default_parameters_structure(cls) -> dict:
        structure = super().default_parameters_structure()
        structure["qx"] = (None, float, ["Q_x"])
        structure["qy"] = (None, float, ["Q_y"])
        structure["occ"] = (None, float, ["occ"])
        return structure

    @property
    def label(self) -> str:
        return super().label + f"occ{self.parameters["occ"]:.4f}qx{self.parameters["qx"]:.4f}qy{self.parameters["qy"]:.4f}"

    def setup(self, calculation_dir: str):
        with open(os.path.join(calculation_dir,"log.txt"),'w') as log_f:
            log_f.write("\n")
        return super().setup(calculation_dir)
    
    @classmethod
    def from_calculation(cls, calc_dirn: str, *kargs, **kwargs) -> Self:
        calc = super().from_calculation(calc_dirn, *kargs, **kwargs)

        conv_fn = os.path.join(calc_dirn, "solver.conv.dat")
        if os.path.isfile(conv_fn):
            with open(conv_fn, 'r') as conv_f:
                header = conv_f.readline()
                column_labels = header.strip().split(sep=',')
                if column_labels[-1].strip() == "mune":
                    conv_data = np.loadtxt(conv_fn, skiprows=1)
                    mu = conv_data[-1,-1]
                    calc.parameters["mu"] = mu

        return calc