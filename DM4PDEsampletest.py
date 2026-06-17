import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
import matplotlib.pyplot as plt
import copy
import argparse

from scripts.generate_poisson import *
from scripts.generate_helmholtz import *

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description='Param')
parser.add_argument('--pdetype', type=str, default='poisson', help='PDEs to solve.')
parser.add_argument('--evaltype', type=str, default='obseval', help='Type of eval.')
parser.add_argument('--pdeloss', type=str, default=True, help='whether use pde loss to guide.')
parser.add_argument('--problem', type=str, default='forward', help='forward or inverse problem to solve.')
args = parser.parse_args()


def test_poisson(s, obs_schedule, problem = 'forward'):
    """
    Test the influence of different observation sample numbers.
    Args:
        s (int): img resolution.
        obs_schedule (list): different observation sample numbers to test.
        problem (str, optional): forward or inverse problem to solve. Defaults to 'forward'.
    """
    
    # basic configs
    poisson_config={
        "data": {
            "name": 'Poisson',
            "datapath": '/data0/zhangxf/testPDEdata/poisson/testpoisson_5000-64-64_1.mat',
            "offset": 0,
            "obs_size": 500
        },
        "test": {
            "pre-trained": '/data0/zhangxf/PDEdata/models/pretrained-multi/00001--uncond-ddpmpp-edm-gpus3-batch60-fp32/network-snapshot-012024.pkl',
            "iterations": 2000
        },
        "generate": {
            "seed": 0,
            "device": 'cuda:7',
            "batch_size": 1,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "rho": 7,
            "zeta_obs_a": 800,
            "zeta_obs_u": 4000,
            "zeta_pde": 1,
            "guide": True,
            "full": False
        },
        "output": {
            "file_path": "",
            "save": False,
            "return": True
        }
    }
    
    print("Eval Poisson ...")
    
    # Global True
    poisson_testdata = scipy.io.loadmat('/data0/zhangxf/testPDEdata/poisson/testpoisson_5000-64-64_1.mat')
    poisson_testdata['f_data'].shape, poisson_testdata['phi_data'].shape
    
    plt.figure(figsize=(120, 12))
    
    plt.subplot(2, 11, 1)
    plt.imshow(poisson_testdata['f_data'][poisson_config['data']['offset'], :, :], cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.title('Poisson f True')
    plt.subplot(2, 11, 12)
    plt.imshow(poisson_testdata['phi_data'][poisson_config['data']['offset'], :, :], cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.title(r'Poisson $\phi$ True')
    
    if problem == "forward":
        print("Eval Forward Problems...\n")
        for i in tqdm.tqdm(range(10)):
            nobs = obs_schedule[i]
            if nobs == 1.0:
                poisson_config["generate"]["full"] = True
            else:
                poisson_config["generate"]["full"] = False
                poisson_config["data"]["obs_size"] = round(nobs * (s * s))
                
            # Forward problems: Sample with obs_loss (just f) and pde_loss guide
            poisson_config_forward = copy.deepcopy(poisson_config)
            poisson_config_forward['generate']['zeta_obs_u'] = 0
            poisson_f_sample_guide_forward, poisson_phi_sample_guide_forward, poisson_f_index_forward, poisson_phi_index_forward = generate_poisson(poisson_config_forward)
            
            plt.subplot(2, 11, 1+i+1)
            plt.imshow(poisson_f_sample_guide_forward[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title(f'DM4PDE {nobs} Obs Guide')
            plt.subplot(2, 11, 12+i+1)
            plt.imshow(poisson_phi_sample_guide_forward[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title("")
            
        plt.savefig("/home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_forward.svg", format="svg")
        print(f"End Eval. Results save to /home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_forward.svg \n")
        
    elif problem == "inverse":
        print("Eval Inverse Problems...\n")
        for i in tqdm.tqdm(range(10)):
            nobs = obs_schedule[i]
            if nobs == 1.0:
                poisson_config["generate"]["full"] = True
            else:
                poisson_config["generate"]["full"] = False
                poisson_config["data"]["obs_size"] = round(nobs * (s * s))
                
            # Inverse problems: Sample with obs_loss (just phi) and pde_loss guide
            poisson_config_inverse = copy.deepcopy(poisson_config)
            poisson_config_inverse['generate']['guide'] = True
            poisson_config_inverse['generate']['zeta_obs_a'] = 0
            poisson_f_sample_guide_inverse, poisson_phi_sample_guide_inverse, poisson_f_index_inverse, poisson_phi_index_inverse = generate_poisson(poisson_config_inverse)
            
            plt.subplot(2, 11, 1+i+1)
            plt.imshow(poisson_f_sample_guide_inverse[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title(f'DM4PDE {nobs} Obs Guide')
            plt.subplot(2, 11, 12+i+1)
            plt.imshow(poisson_phi_sample_guide_inverse[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title("")
        
        plt.savefig("/home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_inverse.svg", format="svg")
        print(f"End Eval. Results save to /home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_inverse.svg \n")

def test_helmholtz1(s, obs_schedule, problem = 'forward'):
    """Test the influence of different observation sample numbers.
    
    Args:
        problem (str, optional): forward or inverse problem to solve. Defaults to 'forward'.
    """
    
    # basic configs.
    helmholtz_config={
        "data": {
            "name": 'Helmholtz',
            "datapath": '/data0/zhangxf/testPDEdata/helmholtz/testhelmholtz(k=1)_5000-64-64_1.mat',
            "offset": 0,
            "obs_size": 500,
            "pdeparam": 1
        },
        "test": {
            "pre-trained": '/data0/zhangxf/PDEdata/models/pretrained-multi/00001--uncond-ddpmpp-edm-gpus3-batch60-fp32/network-snapshot-012024.pkl',
            "iterations": 2000
        },
        "generate": {
            "seed": 0,
            "device": 'cuda:7',
            "batch_size": 1,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "rho": 7,
            "zeta_obs_a": 80,
            "zeta_obs_u": 600,
            "zeta_pde": 1,
            "guide": False,
            "full": False
        },
        "output": {
            "file_path": "",
            "save": False,
            "return": True
        }
    }
    
    print("Eval Helmholtz ...")

    # Global True
    helmholtz_testdata = scipy.io.loadmat('/data0/zhangxf/testPDEdata/helmholtz/testhelmholtz(k=1)_5000-64-64_1.mat')
    helmholtz_testdata['f_data'].shape, helmholtz_testdata['psi_data'].shape
    
    plt.figure(figsize=(200, 12))
    plt.subplot(2, 11, 1)
    plt.imshow(helmholtz_testdata['f_data'][helmholtz_config['data']['offset'], :, :], cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.title('helmholtz f True')
    plt.subplot(2, 11, 12)
    plt.imshow(helmholtz_testdata['psi_data'][helmholtz_config['data']['offset'], :, :], cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.title(r'helmholtz $\psi$ True')
    
    if problem == "forward":
        print("Eval Forward Problems...\n")
        for i in tqdm.tqdm(range(10)):
            nobs = obs_schedule[i]
            if nobs == 1.0:
                helmholtz_config["generate"]["full"] = True
            else:
                helmholtz_config["generate"]["full"] = False
                helmholtz_config["data"]["obs_size"] = round(nobs * (s * s))
                
            # Forward problems: Sample with obs_loss (just f) and pde_loss guide
            helmholtz_config_forward = copy.deepcopy(helmholtz_config)
            helmholtz_config_forward['generate']['guide'] = True
            helmholtz_config_forward['generate']['zeta_obs_u'] = 0
            helmholtz_f_sample_guide_forward, helmholtz_psi_sample_guide_forward, helmholtz_f_index_forward, helmholtz_psi_index_forward = generate_helmholtz(helmholtz_config_forward)
            
            plt.subplot(2, 11, 1+i+1)
            plt.imshow(helmholtz_f_sample_guide_forward[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title(f'DM4PDE {nobs} Obs Guide')
            plt.subplot(2, 11, 12+i+1)
            plt.imshow(helmholtz_psi_sample_guide_forward[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title("")
            
        plt.savefig("/home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_forward.svg", format="svg")
        print(f"End Eval. Results save to /home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_forward.svg \n")
        
    elif problem == "inverse":
        print("Eval Inverse Problems...\n")
        for i in tqdm.tqdm(range(10)):
            nobs = obs_schedule[i]
            if nobs == 1.0:
                helmholtz_config["generate"]["full"] = True
            else:
                helmholtz_config["generate"]["full"] = False
                helmholtz_config["data"]["obs_size"] = round(nobs * (s * s))
                
            # Inverse problems: Sample with obs_loss (just psi) and pde_loss guide
            helmholtz_config_inverse = copy.deepcopy(helmholtz_config)
            helmholtz_config_inverse['generate']['guide'] = True
            helmholtz_config_inverse['generate']['zeta_obs_a'] = 0
            helmholtz_f_sample_guide_inverse, helmholtz_psi_sample_guide_inverse, helmholtz_f_index_inverse, helmholtz_psi_index_inverse = generate_helmholtz(helmholtz_config_inverse)
            
            plt.subplot(2, 11, 1+i+1)
            plt.imshow(helmholtz_f_sample_guide_inverse[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title(f'DM4PDE {nobs} Obs Guide')
            plt.subplot(2, 11, 12+i+1)
            plt.imshow(helmholtz_psi_sample_guide_inverse[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title("")
        
        plt.savefig("/home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_inverse.svg", format="svg")
        print(f"End Eval. Results save to /home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_inverse.svg \n")
        
def test_helmholtz2(s, k_schedule, obs_schedule, pdeloss = True, problem = 'forward'):
    """Helmholtz_test_2: Test the performance on different helmholtz equations (set different k)

    Args:
        s (int): img resolution.
        k_schedule (list): different observation sample numbers to test.
        obs_schedule (list): different observation sample numbers to test.
        pdeloss (bool): whether use pde loss to guide. Defaults to True.
        problem (str, optional): forward or inverse problem to solve. Defaults to 'forward'.
    """
    
    # basic configs.
    helmholtz_config={
        "data": {
            "name": 'Helmholtz',
            "datapath": f'/data0/zhangxf/testPDEdata/helmholtz/testhelmholtz(k=1)_5000-{s}-{s}_1.mat',
            "offset": 0,
            "obs_size": 500,
            "pdeparam": 1
        },
        "test": {
            "pre-trained": '/data0/zhangxf/PDEdata/models/pretrained-multi/00001--uncond-ddpmpp-edm-gpus3-batch60-fp32/network-snapshot-012024.pkl',
            "iterations": 2000
        },
        "generate": {
            "seed": 0,
            "device": 'cuda:7',
            "batch_size": 1,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "rho": 7,
            "zeta_obs_a": 80,
            "zeta_obs_u": 600,
            "zeta_pde": 10,
            "guide": True,
            "full": False
        },
        "output": {
            "file_path": "",
            "save": True,
            "return": True
        }
    }
    
    if pdeloss == False:
        helmholtz_config['generate']['zeta_pde'] = 0
        
    if problem == 'forward':
        helmholtz_config['generate']['zeta_obs_u'] = 0
    elif problem == 'inverse':
        helmholtz_config['generate']['zeta_obs_a'] = 0
    
    plt.figure(figsize=(100, 32))
    
    for i in range(8):
        k = k_schedule[i]
        helmholtz_config['data']['datapath'] = f'/data0/zhangxf/testPDEdata/helmholtz/testhelmholtz(k={k})_5000-{s}-{s}_1.mat'
        helmholtz_config['data']['pdeparam'] = k
        helmholtz_config['output']['file_path'] = f'/data0/zhangxf/testPDEdata/testsamples/testhelmholtz(k={k})_results.mat'
        
        # Global True
        helmholtz_testdata = scipy.io.loadmat(helmholtz_config['data']['datapath'])
        
        if i == 0:
            plt.subplot(4, 9, 1)
            plt.imshow(helmholtz_testdata['f_data'][helmholtz_config['data']['offset'], :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title('helmholtz f True')
            
        plt.subplot(4, 9, 1+i+1)
        plt.imshow(helmholtz_testdata['psi_data'][helmholtz_config['data']['offset'], :, :], cmap='plasma', interpolation='nearest')
        plt.colorbar()
        plt.title(r'helmholtz $\psi$ True')
        
        for j in range(3):
            n = obs_schedule[j]
            print(f"Schedule: k = {k}, n = {n}.")
            helmholtz_config_test = copy.deepcopy(helmholtz_config)
            if n == 1.0:
                helmholtz_config_test['generate']['full'] = True
            else:
                helmholtz_config_test['data']['obs_size'] = round(n * (s * s))
                
            helmholtz_f_sample_guide, helmholtz_psi_sample_guide, helmholtz_f_index, helmholtz_psi_index = generate_helmholtz(helmholtz_config_test)
            
            if i == 0:
                plt.subplot(4, 9, 1+(j+1)*9)
                plt.imshow(helmholtz_f_sample_guide[0, 0, :, :], cmap='plasma', interpolation='nearest')
                plt.colorbar()
                plt.title(fr"DM4PDE f, k = {k}, {n} Obs.")
            
            plt.subplot(4, 9, 1+(j+1)*9+i+1)
            plt.imshow(helmholtz_psi_sample_guide[0, 0, :, :], cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.title(fr"DM4PDE $\psi$, k = {k}, {n} Obs.")
        
    plt.tight_layout()
            
    plt.savefig(f"/home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_helmholtz(k={k},{problem},{pdeloss}).svg", format="svg")
    plt.savefig(f"/home/zjinzxf2025/C01Python/DiffusionPDE/eval/DM4PDE_helmholtz(k={k},{problem},{pdeloss}).pdf", format="pdf")
    


if __name__ == "__main__":
    
    s = 64
    obs_schedule = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_schedule = [1, 2, 3, 5, 10, 20]
    
    if args.pdetype == "poisson":
        test_poisson(s, obs_schedule, args.problem)
    elif args.pdetype == "helmholtz":
        if args.evaltype == "obseval":
            test_helmholtz1(s, obs_schedule, args.problem)
        elif args.evaltype == "parameval":
            test_helmholtz2(s, k_schedule, [0.1, 0.5, 1.0], args.pdeloss, args.problem)
    else:
        print("None. \n")