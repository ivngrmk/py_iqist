import unittest
import os
from py_iqist.iqist_calculation import iQISTCalculation
from simple_slurm import Slurm

class TestCalculator(unittest.TestCase):
    SETUP_DIR = "./calc_setup"
    TEMPLATE_F_PATH = "./template_solver.ctqmc.in"
    EXAMPLE_DIR_PATH = "./calc_example"
    BIN_F_PATH = "./test_bin.sh"

    def setUp(self) -> None:
       for fn in os.listdir(self.SETUP_DIR):
            os.remove(os.path.join(self.SETUP_DIR,fn))

    def test_creation(self):
        iqist_calc = iQISTCalculation(parameters={
        "uu" : 8.0,
        "beta" : 10.0,
        "tt" : 0.0,
        "mu" : 4.0,
        }, template_path="./template_solver.ctqmc.in")
        self.assertTrue(not iqist_calc is None)
        self.assertEqual(iqist_calc.parameters["uu"], 8.0)
        self.assertEqual(iqist_calc.parameters["beta"], 10.0)
        self.assertEqual(iqist_calc.parameters["tt"], 0.0)
        self.assertEqual(iqist_calc.parameters["mu"], 4.0)

    def test_creation_from_calc(self):
       iqist_calc = iQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH)
       self.assertTrue(not iqist_calc is None)

    def test_change_parameters(self):
        iqist_calc = iQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH)
        iqist_calc.change_parameters({"beta" : 40.0})
        self.assertEqual(iqist_calc.parameters["beta"], 40.0)

    def test_setup(self):
        iqist_calc = iQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH)
        iqist_calc.setup(self.SETUP_DIR)

    def test_run(self):
        iqist_calc = iQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH, bin_path=self.BIN_F_PATH)
        iqist_calc.setup(self.SETUP_DIR)
        iqist_calc.run(sbatch=False, cpus_per_task = 1, nodes = 1)
        self.assertTrue(os.path.isfile(os.path.join(self.SETUP_DIR,"run.sh")))

    def test_dep_run(self):
        iqist_calc = iQISTCalculation.from_calculation(self.EXAMPLE_DIR_PATH, template_path=self.TEMPLATE_F_PATH, bin_path=self.BIN_F_PATH)
        iqist_calc.setup(self.SETUP_DIR)
        job_id = iqist_calc.run(sbatch=True, cpus_per_task = 1, nodes = 1)
        
        self.assertTrue(isinstance(job_id,int))

        dep_job_id = iqist_calc.run(sbatch=True, cpus_per_task = 1, nodes = 1, dependency = job_id)
        self.assertTrue(isinstance(dep_job_id,int) and dep_job_id > job_id)

if __name__ == "__main__":
  unittest.main()