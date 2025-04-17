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