import unittest
import os

import numpy as np

from py_iqist.fixed_n_iqist_calculation import FixedNiQISTCalculation

class TestCalculator(unittest.TestCase):
    SETUP_DIR = "./calc_setup"
    TEMPLATE_F_PATH = "./template_solver.ctqmc.in"
    EXAMPLE_DIR_PATH = "./calc_example"
    BIN_F_PATH = "./test_bin.sh"

    def setUp(self) -> None:
       for fn in os.listdir(self.SETUP_DIR):
            os.remove(os.path.join(self.SETUP_DIR,fn))

    def test_creation(self):
        iqist_calc = FixedNiQISTCalculation(parameters={
        "uu" : 8.0,
        "beta" : 10.0,
        "tt" : 0.0,
        "mu" : 4.0,
        "qx" : np.pi,
        "qy" : np.pi,
        "occ" : 0.995,
        }, template_path="./template_solver.ctqmc.in")
        self.assertTrue(not iqist_calc is None)
        self.assertEqual(iqist_calc.parameters["uu"], 8.0)
        self.assertEqual(iqist_calc.parameters["beta"], 10.0)
        self.assertEqual(iqist_calc.parameters["tt"], 0.0)
        self.assertEqual(iqist_calc.parameters["mu"], 4.0)
        self.assertEqual(iqist_calc.parameters["qx"], np.pi)
        self.assertEqual(iqist_calc.parameters["qy"], np.pi)

    def test_creation_from_calc(self):
       iqist_calc = FixedNiQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH)
       self.assertTrue(not iqist_calc is None)

    def test_change_parameters(self):
        iqist_calc = FixedNiQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH)
        iqist_calc.change_parameters({"beta" : 40.0})
        self.assertEqual(iqist_calc.parameters["beta"], 40.0)
        self.assertEqual(iqist_calc.parameters["qx"], 2.0943951)

    def test_setup(self):
        iqist_calc = FixedNiQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH)
        iqist_calc.setup(self.SETUP_DIR)

    def test_run(self):
        iqist_calc = FixedNiQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH, bin_path=self.BIN_F_PATH)
        iqist_calc.setup(self.SETUP_DIR)
        iqist_calc.run(sbatch=False, cpus_per_task = 1, nodes = 1)
        self.assertTrue(os.path.isfile(os.path.join(self.SETUP_DIR,"run.sh")))

if __name__ == "__main__":
  unittest.main()