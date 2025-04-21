import enum
import os
from typing import Optional, Self, Tuple
import re
import abc

from triqs.gf import MeshImFreq
from triqs.gf.gf import Gf

# following change should be done
# export SQUEUE_FORMAT='"%i","%j","%t","%M","%L","%D","%C","%m","%b","%R"'
# in order to sucsessfully use this module
# this is taken as a default squeue_format from simple_slurm
os.environ["SQUEUE_FORMAT"] = '"%i","%j","%t","%M","%L","%D","%C","%m","%b","%R"'
from simple_slurm import Slurm

import numpy as np

class InputSolverFile(enum.StrEnum):
    CTQMC = "solver.ctqmc.in"
    NMAT = "solver.nmat.in"
    CH_W = "solver.ch_w.in"
    HYB = "solver.hyb.in"
    KTAU = "solver.ktau.in"
    PIB = "solver.pib.in"
    SGM = "solver.sgm.in"

class iQISTCalculation():
    # Method and attributes to redefine by user #
    @property
    def label(self) -> str:
        """Should return label of the calculation used for SLURM job_name.
        """
        return f"U{self.parameters["uu"]:.2f}beta{self.parameters["uu"]:.2f}tt{self.parameters["tt"]:.2f}"
    
    @classmethod
    def default_parameters_structure(cls) -> dict:
        """All parameters specified here should present in upper case in the template file as a keywords, prefixed with space and followed by \\n.
            qx -> QX, nbfrq -> NBFRQ, etc.
        """
        return {
            # parameter_name, default_value, type, tuple of labels in solver.ctqmc.in (corresponding keywords is uppercase of parameter_name)
            'niter'       : ( 10,       int,   ["niter",]),
            'tt'          : (None,      float, ["t1",]),
            'uu'          : (None,      float, ["U", "Uc"]),
            'mu'          : (None,      float, ["mune",]),
            'beta'        : (None,      float, ["beta",]),
            'alpha'       : (0.7,       float, ["alpha,"]),
            # DMFT specific parameters
            'mfreq'       : (8193,      int,   ["mfreq",]),
            'ntime'       : (8192,      int,   ["ntime",]),
            'nbfrq_local' : (128,       int,   ["nbfrq",]),
            # MonteCarlo parameters
            'nflip'       : (10000,     int,   ["nflip",]),
            'ntherm'      : (10000000,  int,   ["ntherm",]),
            'nsweep'      : (100000000, int,   ["nsweep",]),
            'nwrite'      : (10000000,  int,   ["nwrite",]),
            'nclean'      : (100000,    int,   ["nclean",]),
            'nmonte'      : (100,       int,   ["nmonte",]),
            'ncarlo'      : (100,       int,   ["ncarlo",]),
            # vertex calculation
            'nffrq'       : (1,         int,   ["nffrq1",]),
            'nbfrq'       : (1,         int,   ["nbfrq1",]),
        }
    
    @classmethod
    def default_parameters(cls) -> dict:
        parameters = {}
        parameters_structure = cls.default_parameters_structure()
        for parameter_key in parameters_structure:
            parameters[parameter_key] = parameters_structure[parameter_key][0]
        return parameters

    @staticmethod
    def separate_parameters_line(line : str) -> list:
        return line.replace(' ','').strip().split(':')
    
    @classmethod
    def parse_parameters_line(cls, line : str, parameters : dict) -> str:
        structure = cls.default_parameters_structure()
        assert set(parameters.keys()) == set(structure)
        words = cls.separate_parameters_line(line)
        if len(words) != 2:
            return line
        for parameter_key in parameters:
            if words[0] in structure[parameter_key][2]:
                parameters[parameter_key] = structure[parameter_key][1](words[-1])
        return line

    def generate_input_file(self, file_type : InputSolverFile, data : np.ndarray, beta : float, data_err : Optional[np.ndarray] = None) -> str:
        file_str = None
        if file_type is InputSolverFile.CH_W:
            file_str = ""
            if data_err is None:
                data_err = 0.0*data
            if len(data.shape) != 3 or data.shape[1] != data.shape[2]:
                raise RuntimeError("Wrong shape of data :", data.shape)
            if data.shape != data_err.shape:
                raise RuntimeError("Missmatch of data and data_err shapes :", data.shape, data_err.shape)
            nbfrq = data.shape[0]
            nspin = data.shape[1]
            for sigma2 in range(nspin):
                for sigma1 in range(nspin):
                    flvr1 = sigma1 + 1
                    flvr2 = sigma2 + 1
                    file_str += f"# flvr:     {flvr1}  flvr:     {flvr2}\n"
                    for n in range(nbfrq):
                        file_str += f"{2*np.pi*n/beta:12.6f}{data[n,sigma1,sigma2]:12.6f}{data_err[n,sigma1,sigma2]:12.6f}\n"
                    file_str += "\n"
                    file_str += "\n"
            file_str += " Total:\n"
            total_data = data.sum(axis=(1,2))
            total_data_err = data_err.sum(axis=(1,2))
            for n in range(nbfrq):
                file_str += f"{2*np.pi*n/beta:12.6f}{total_data[n]:12.6f}{total_data_err[n]:12.6f}\n"
            file_str += "\n"
            file_str += "\n"
        if file_type is InputSolverFile.KTAU:
            file_str = ""
            file_str += " \n"
            ntime = data.shape[0]
            if len(data.shape) != 1:
                raise RuntimeError
            taus = np.linspace(0.0,beta,ntime,endpoint=True)
            for tau, value in zip(taus, data):
                file_str += f"{tau:24.14f}{value.real:24.14f}{value.imag:24.14f}\n"
        if file_type is InputSolverFile.PIB:
            file_str = ""
            nbfrq = data.shape[0]
            if len(data.shape) != 1:
                raise RuntimeError
            for n in range(nbfrq):
                file_str += f"{0:6d}{2*np.pi/beta*n:16.8f}{data[n].real:16.8f}{data[n].imag:16.8f}{0.0:16.8f}{0.0:16.8f}\n"
            file_str += "\n"
            file_str += "\n"
        if file_type is InputSolverFile.NMAT:
            file_str = ""
            if data_err is None:
                data_err = 0.0*data
            if data.shape[0] != 2 or len(data.shape) != 1:
                raise RuntimeError("Wrong shape of data :", data.shape)
            file_str += "#   < n_i >   data:\n"
            for sigma in range(2):
                file_str += f"{sigma+1:6d}{data[sigma]:12.6f}{data_err[sigma]:12.6f}\n"
            file_str += f"   sup{data[0]:12.6f}{data_err[0]:12.6f}\n"
            file_str += f"   sdn{data[1]:12.6f}{data_err[1]:12.6f}\n"
            file_str += f"   sum{data.sum():12.6f}{data_err.sum():12.6f}\n"
            # Does not support pair-wise correlator
            file_str += "# < n_i n_j > data:\n"
            for sigma1 in range(2):
                for sigma2 in range(2):
                    file_str += f"{sigma1+1:6d}{sigma2+1:6d}{0.0:12.6f}{0.0:12.6f}\n"
        if file_type is InputSolverFile.SGM or file_type is InputSolverFile.HYB:
            file_str = ""
            if data_err is None:
                data_err = 0.0*data
            if len(data.shape) != 2 or data.shape != data_err.shape or data.shape[1] != 2:
                raise RuntimeError("Wrong shape of data :", data.shape)
            mfreq = data.shape[0]
            for sigma in range(2):
                for n in range(mfreq):
                    data_v = data[n][sigma]
                    data_v_err = data_err[n][sigma]
                    file_str += f"{sigma+1:6d}{(2*n+1)*np.pi/beta:16.8f}{np.real(data_v):16.8f}{np.imag(data_v):16.8f}{data_v_err.real:16.8f}{data_v_err.imag:16.8f}\n"
                file_str += "\n"
                file_str += "\n"
        if file_type is InputSolverFile.CTQMC:
            if not self.check_template():
                raise RuntimeError("Wrong template.")

            file_str = ""
            with open(self.template_path,'r') as f:
                file_str = f.read()
            for key in self.parameters:
                file_str = file_str.replace(key.upper(), str(self.parameters[key]))

        return file_str
    
    def generate_input_data(self, file_type : InputSolverFile) -> Tuple[np.ndarray,np.ndarray]:
        if file_type is InputSolverFile.CH_W:
            data = np.zeros((self.parameters["nbfrq"],2,2),dtype=float)
        if file_type is InputSolverFile.KTAU:
            data = np.zeros(self.parameters["ntime"], dtype=complex)
        if file_type is InputSolverFile.PIB:
            data = np.zeros(self.parameters["nbfrq_local"], dtype=complex)
        if file_type is InputSolverFile.NMAT:
            data = np.array(self.occ_data,dtype=float)
        if file_type is InputSolverFile.SGM:
            data = self.sgm_data
        if file_type is InputSolverFile.HYB:
            data = self.hyb_data
        if file_type is InputSolverFile.CTQMC:
            data = None
        # Default data is without error
        if isinstance(data, np.ndarray):
            return data, 0.0*data
        else:
            return data, None

    def setup_initial_data(self) -> None:
        mfreq = self.parameters["mfreq"]
        U = self.parameters["uu"]
        occ = self.parameters["occ"]

        self.occ_data = (2.0*occ/3.0,1.0*occ/3.0)
        nup = self.occ_data[0]
        ndn = self.occ_data[1]
        self.sgm_data = np.array(np.concatenate((U*ndn*np.ones(mfreq)[:,None],U*nup*np.ones(mfreq)[:,None]),axis=1),dtype=complex)
        self.hyb_data = self.sgm_data.copy()
    
    #----------------------------------------------------------#

    def f_mesh(self):
        beta = self.parameters["beta"]
        mfreq = self.parameters["mfreq"]
        return (2*np.arange(mfreq)+1)*np.pi/beta
    
    def gf_to_triqs(self, data :  np.ndarray, name : Optional[str] = None):
        iw_mesh = MeshImFreq(beta=self.parameters["beta"], S='Fermion', n_iw=self.parameters["mfreq"])
        # only fermionic is assumed
        gf_data = np.concatenate((np.conjugate(data[::-1]),data),axis=0)
        if not name is None:
            triqs_gf = Gf(mesh = iw_mesh, data = gf_data, name = name)
        else:
            triqs_gf = Gf(mesh = iw_mesh, data = gf_data)
        return triqs_gf
    
    def sgm_to_triqs(self) -> Gf:
        if not self.sgm_data is None:
            return self.gf_to_triqs(self.sgm_data, name="iqist_self_energy")
        else:
            raise RuntimeError("Self energy data is None.")
    
    def grn_to_triqs(self) -> Gf:
        if not self.grn_data is None:
            return self.gf_to_triqs(self.grn_data, name="iqist_self_energy")
        else:
            raise RuntimeError("Green's function data is None.")

    def hyb_to_triqs(self) -> Gf:
        if not self.hyb_data is None:
            return self.gf_to_triqs(self.hyb_data, name="iqist_self_energy")
        else:
            raise RuntimeError("Hybridization function data is None.")


    def __init__(self, parameters : dict, template_path : str, bin_path : Optional[str] = None) -> None:
        self.bin_path = bin_path
        self.template_path = template_path

        self.parameters : dict = self.default_parameters()
        for parameter_key in parameters:
            self.parameters[parameter_key] = parameters[parameter_key]
        for parameter_key in self.parameters:
            if self.parameters[parameter_key] is None:
                raise ValueError(f"Parameter {parameter_key} remained None.")
            
        self.grn_data : Optional[np.ndarray] = None
        self.sgm_data : Optional[np.ndarray] = None
        self.hyb_data : Optional[np.ndarray] = None
        self.occ_data : Optional[np.ndarray] = None

        self.calculation_dir : str = None

    def check_template(self):
        template_path = self.template_path
        with open(template_path, 'r') as template_f:
            template = template_f.read()
        for key in self.default_parameters():
            if not (key.upper() + "\n" in template):
                return False
        return True

    def change_parameters(self, new_parameters : dict) -> None:
        for key in new_parameters:
            if key in self.parameters:
                self.parameters[key] = new_parameters[key]
            else:
                raise KeyError

    def write_input_file(self, dir_path : str, file_type : InputSolverFile) -> None:
        data, data_err = self.generate_input_data(file_type=file_type)
        with open(os.path.join(dir_path,str(file_type)), 'w') as f:
            f.write(self.generate_input_file(data = data, data_err = data_err, beta = self.parameters["beta"], file_type = file_type))

    def setup(self, calculation_dir : str):
        self.calculation_dir = calculation_dir
        for input_type in InputSolverFile:
            self.write_input_file(self.calculation_dir, input_type)

    def run(self, *args, sbatch : bool = False, **kwargs) -> Optional[int]:
        """ If sbatch is False, then the script is only created but the job is not launched.
        """
        if not os.path.isfile(self.bin_path):
            raise RuntimeError("Binary file not found.")
        if self.calculation_dir is None:
            raise RuntimeError("Calculation dir was not specified.")
        job_name = self.label
        slurm = self.generate_slurm(*args, job_name=job_name, **kwargs)
        batch_script = str(slurm)
        run_f_path = os.path.join(self.calculation_dir,'run.sh')
        with open(run_f_path,'w') as f:
            f.write(batch_script)
        if not os.path.isfile(run_f_path):
            raise RuntimeError("Could not create run.sh .")
        if sbatch:
            job_id = slurm.sbatch()
            return job_id
        
    def generate_slurm(self, cpus_per_task : int, nodes : int, job_name : str, partition = "mpi", output = '1.o' ,dependency : Optional[int] = None) -> Slurm:
        if not dependency is None:
            slurm = Slurm(job_name = job_name, cpus_per_task = cpus_per_task, nodes = nodes, partition = partition, output = output, dependency=dict(afterok=dependency))
        else:
            slurm = Slurm(job_name = job_name, cpus_per_task = cpus_per_task, nodes = nodes, partition = partition, output = output)
        slurm.add_cmd("module purge")
        slurm.add_cmd("module load mpi/impi-5.0.3")
        slurm.add_cmd("module load intel/mkl-11.2.3")
        slurm.add_cmd(f"{self.bin_path}")
        return slurm

    @classmethod
    def from_calculation(cls, calc_dirn : str, *kargs, **kwargs) -> Self:
        # Loading parameters from calculation
        parameters = cls.default_parameters()
        with open(os.path.join(calc_dirn,InputSolverFile.CTQMC),'r') as f:
            for line in f:
                _ = cls.parse_parameters_line(line, parameters)
        # Creating iQISTCalculation instance
        iqist_calculation = cls(parameters, *kargs, **kwargs)

        # Reading  data from dat files corresponding to last computed iteration
        nmat_pattern = r'solver\.nmat\s*(\d+)\.dat'
        nmat_f_path = cls.get_last_file_by_pattern(calc_dirn, nmat_pattern)
        sgm_pattern = r'solver\.sgm\s*(\d+)\.dat'
        sgm_f_path = cls.get_last_file_by_pattern(calc_dirn, sgm_pattern)
        hyb_pattern = r'solver\.hyb\s*(\d+)\.dat'
        hyb_f_path = cls.get_last_file_by_pattern(calc_dirn, hyb_pattern)
        grn_f_path = os.path.join(calc_dirn,"solver.grn.dat")

        if os.path.isfile(nmat_f_path):
            iqist_calculation.occ_data    = iqist_calculation.get_occ_data(nmat_f_path)
        if os.path.isfile(sgm_f_path):
            iqist_calculation.sgm_data, _ = iqist_calculation.get_gf_data(sgm_f_path)
        if os.path.isfile(hyb_f_path):
            iqist_calculation.hyb_data, _ = iqist_calculation.get_gf_data(hyb_f_path)
        if os.path.isfile(grn_f_path):
            iqist_calculation.grn_data, _ = iqist_calculation.get_gf_data(grn_f_path)

        return iqist_calculation
    
    @staticmethod
    def get_occ_data(nmat_fn : str) -> np.ndarray:
        with open(nmat_fn, 'r') as file:
            for _ in range(1):
                _ = file.readline()
            nup = float(file.readline().split()[1])
            ndn = float(file.readline().split()[1])
            return np.array((nup,ndn),dtype=float)

    @staticmethod
    def get_last_file_by_pattern(dir_path : str, pattern : str) -> str:
        def extract_it_number(fn, pattern):
            match = re.search(pattern, fn)
            if match:
                return int(match.group(1))
            else:
                return None
        fns = []
        for fn in os.listdir(dir_path):
            match = re.search(pattern, fn)
            if match:
                fns.append(fn)
        fns = sorted(fns, key=lambda fn : extract_it_number(fn, pattern))
        return os.path.join(dir_path,fns[-1])
    
    @staticmethod
    def get_gf_data(gf_fn : str) -> Tuple[np.ndarray, np.ndarray]:
        with open(gf_fn, 'r') as file:
            lines = file.readlines()
        mfreq = (len(lines) - 2*2) // 2

        gf_data = np.zeros((mfreq,2),dtype=complex)
        gf_data_err = np.zeros((mfreq,2),dtype=complex)
        with open(gf_fn, 'r') as file:
            for sigma in range(2):
                for i in range(mfreq):
                    line = file.readline()
                    words = line.split()
                    v = complex(float(words[2]), float(words[3]))
                    v_rel_error = abs(
                        complex(float(words[4]), float(words[5])))/abs(v)
                    gf_data[i, sigma] = v
                    gf_data_err[i, sigma] = v_rel_error
                file.readline()
                file.readline()
        return gf_data, gf_data_err

    def __str__(self) -> str:
        msg = self.label + "\n"
        for key in self.parameters:
            msg += f"{key} : {self.parameters[key]}\n"
        return msg