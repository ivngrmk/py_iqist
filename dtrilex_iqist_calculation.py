from typing import Self
import os

from .iqist_calculation import iQISTCalculation
import numpy as np
    
class DTRILEXQISTCalculation(iQISTCalculation):
    @classmethod
    def default_parameters_structure(cls) -> dict:
        structure = super().default_parameters_structure()
        # Does not watching at mune parameter at all. Leaving it with default value from the template.
        del structure["mu"]
        structure["n"] = (None, float, ["n"])
        return structure

    @classmethod
    def from_calculation(cls, calc_dirn: str) -> Self:
        iqist_calculation = super().from_calculation(calc_dirn)

        # Using last iteration for loading data
        mu_fn = os.path.join(calc_dirn,"solver.mu.dat")
        if os.path.isfile(mu_fn):
            iqist_calculation.mu = iqist_calculation.get_mu_data(mu_fn)
        return iqist_calculation

    def get_mu_data(self, mu_f_path : str) -> float:
        data = np.loadtxt(mu_f_path)
        return data[-1,-1]
    
    def __init__(self, parameters: dict, bin_path: str, template_path: str) -> None:
        super().__init__(parameters, bin_path, template_path)

        self.mu = None

    def setup(self, calculation_dir: str):
        super().setup(calculation_dir)

        if not self.mu is None:
            f_str = "\n"
            f_str += f"1 {self.mu:10.6f}\n"
        with open(os.path.join(calculation_dir,"solver.mu.in"),'w') as mu_f:
            mu_f.write(f_str)
