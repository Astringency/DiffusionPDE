import numpy as np
import os
import scipy
import h5py
import argparse

parser = argparse.ArgumentParser(description='Param')
parser.add_argument('--type', type=str, default='darcy', help='PDE type')
parser.add_argument('--round', type=str, default=1, help='Dataset num')
parser.add_argument('--n', type=str, default=10000, help='Dataset size')
parser.add_argument('--s', type=str, default=128, help='Dataset resolution')
parser.add_argument('--mixed', type=bool, default=False, help='If or not multi-datasets merger')
parser.add_argument('--save', type=str, default="/data0/zhangxf/PDEdata/multi-merged/", help='If or not multi-datasets merger')
args = parser.parse_args()

pde_type = args.type
R = int(args.round)
N = int(args.n)
S = int(args.s)

data_base_path = "/data0/zhangxf/PDEdata/"

if args.mixed:
    output_base_path = args.save
else:
    output_base_path = f"/data0/zhangxf/PDEdata/{pde_type}-merged/"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

# Load raw training data from .mat files

if pde_type != "helmholtzk":
    for j in range(1, R+1):
        print(f"Processing {pde_type} file {j}...")
        if pde_type == "darcy":
            file_path = f'{data_base_path}{pde_type}/{pde_type}_{N}-{S}-{S}_{j}.mat'
            with h5py.File(file_path, 'r') as file:
                a = file['thresh_a_data'][:] * 0.2 - 1.5
                u = file['thresh_p_data'][:] * 115 - 0.9
            a = a.transpose(0, 1, 2)
            u = u.transpose(0, 1, 2)
        elif pde_type == "poisson":
            file_path = f'{data_base_path}{pde_type}/{pde_type}_{N}-{S}-{S}_{j}.mat'
            f = scipy.io.loadmat(file_path)['f_data'] / 2.5
            phi = scipy.io.loadmat(file_path)['phi_data'] * 36.5
            f = f.transpose(1, 2, 0)
            phi = phi.transpose(1, 2, 0)
        elif pde_type == "nsnonbounded":
            file_path = f'{data_base_path}{pde_type}/ns_{N}-{S}-{S}-10_{j}.mat'
            with h5py.File(file_path, 'r') as file:
                a = file['a'][:] / 1.6
                u = file["u"][:, :, :, -1] / 1.6
            a = a.transpose(1, 2, 0)
            u = u.transpose(1, 2, 0)
        elif pde_type == "burger":
            file_path = f'{data_base_path}{pde_type}/{pde_type}_{N}-{S}-{S}_{j}.mat'
            output = scipy.io.loadmat(file_path)['output'] / 1.415
            output = output.reshape(S, S, -1)
        elif pde_type == "helmholtz":
            file_path = f'{data_base_path}{pde_type}/{pde_type}_{N}-{S}-{S}_{j}.mat'
            f = scipy.io.loadmat(file_path)['f_data'] / 2.15
            psi = scipy.io.loadmat(file_path)['psi_data'] / 0.028
            f = f.transpose(1, 2, 0)
            psi = psi.transpose(1, 2, 0)
        else:
            raise NotImplementedError(f"Unsupported dataset {pde_type}")
            
        # Iterate over each index i
        for i in range(N):
            # Load the files
            if pde_type == "darcy":
                f1_transformed = a[:, :, i]
                f2_transformed = u[:, :, i]
                combined = np.stack((f1_transformed, f2_transformed), axis=-1)
            elif pde_type == "poisson":
                f1_transformed = f[:, :, i]
                f2_transformed = phi[:, :, i]
                combined = np.stack((f1_transformed, f2_transformed), axis=-1)
            elif pde_type == "nsnonbounded":
                f1_transformed = a[:, :, i]
                f2_transformed = u[:, :, i]
                combined = np.stack((f1_transformed, f2_transformed), axis=-1)
            elif pde_type == "burger":
                f_transformed = output[:, :, i]
                combined = np.expand_dims(f_transformed, axis=2)
            elif pde_type == "helmholtz":
                f1_transformed = f[:, :, i]
                f2_transformed = psi[:, :, i]
                combined = np.stack((f1_transformed, f2_transformed), axis=-1)
            else:
                raise NotImplementedError(f"Unsupported dataset {pde_type}")
            
            # Save the combined array to a new .npy file
            output_file_path = output_base_path + f"merge{pde_type}_{i+(j-1)*N}.npy"
            np.save(output_file_path, combined)
            
            if pde_type == 'burger':
                assert combined.shape == (S, S, 1)
            else:
                assert combined.shape == (S, S, 2)
            
            if i % 500 == 0:
                print(f"Saved combined array for index {i} to {output_file_path}")
                print("Min:", combined.min(), "Max:", combined.max())
elif pde_type  == "helmholtzk":
    for k in [1, 2, 3, 4, 5]:
        print(f"helmholtz k = {k}.")
        file_path = f'{data_base_path}helmholtz/helmholtz_{N}-{S}-{S}_k{k}.mat'
        f = scipy.io.loadmat(file_path)['f_data']
        psi = scipy.io.loadmat(file_path)['psi_data']
        
        # f = (f - np.mean(f)) / np.std(f)
        # psi = (psi - np.mean(psi)) / np.std(psi)
        
        # f = (f - np.min(f)) / (np.max(f) - np.min(f))
        # psi = (psi - np.min(psi)) / (np.max(psi) - np.min(psi))
        
        f = f.transpose(1, 2, 0)
        psi = psi.transpose(1, 2, 0)
            
        # Iterate over each index i
        for i in range(N):
            f1_transformed = f[:, :, i]
            f2_transformed = psi[:, :, i]
            combined = np.stack((f1_transformed, f2_transformed), axis=-1)
            
            # Save the combined array to a new .npy file
            output_file_path = output_base_path + f"mergehelmholtz_{i}_k{k}.npy"
            np.save(output_file_path, combined)
            
            if i % 500 == 0:
                print(f"Saved combined array for k = {k} index {i} to {output_file_path}")
                print("Min:", combined.min(), "Max:", combined.max())

print("Finished processing all files.")