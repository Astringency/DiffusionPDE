import tqdm
import time
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
import h5py


def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


def get_reaction_diffusion_loss(a, u, a_GT, u_GT, a_mask, u_mask, pde_params=None, device=torch.device('cuda')):
    pde_params = pde_params or {}
    S = u.size(-1)
    a_GT = a_GT.view(1, 2, S, S)
    u_GT = u_GT.view(1, 2, S, S)
    a = a.view(1, 2, S, S)
    u = u.view(1, 2, S, S)

    # obs loss
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask

    # pde loss
    T = pde_params.get('T', 5)
    k = pde_params.get('k', 5e-3)
    D_u = pde_params.get('D_u', pde_params.get('Du', 1e-3))
    D_v = pde_params.get('D_v', pde_params.get('Dv', 5e-3))
    h = 1 / (S - 1)

    # Extract components: [batch, channel, H, W]
    # a: initial state (t=0), u: state at time T
    a_u = a[:, 0:1, :, :]   # activator at t=0
    a_v = a[:, 1:2, :, :]   # inhibitor at t=0
    u_u = u[:, 0:1, :, :]   # activator at t=T
    u_v = u[:, 1:2, :, :]   # inhibitor at t=T

    
    # Time derivatives (forward difference)
    u_t = (u_u - a_u) / T   # ∂u/∂t
    v_t = (u_v - a_v) / T   # ∂v/∂t

    # Laplacian with periodic boundary conditions
    def laplacian(field):
        # field shape: [batch, 1, H, W]
        # Roll operations for periodic boundaries
        top = torch.roll(field, shifts=1, dims=2)
        bottom = torch.roll(field, shifts=-1, dims=2)
        left = torch.roll(field, shifts=1, dims=3)
        right = torch.roll(field, shifts=-1, dims=3)
        
        # Central difference approximation
        return (top + bottom + left + right - 4 * field) / (h ** 2)

    # Compute Laplacians for both fields at time T
    lap_u = laplacian(u_u)  # ∇²u
    lap_v = laplacian(u_v)  # ∇²v

    # Reaction terms (Fitzhugh-Nagumo)
    R_u = u_u - u_u**3 - k - u_v  # R_u(u,v)
    R_v = u_u - u_v               # R_v(u,v)

    # PDE residuals
    res_u = u_t - (D_u * lap_u + R_u)  # ∂u/∂t - (D_u∇²u + R_u)
    res_v = v_t - (D_v * lap_v + R_v)  # ∂v/∂t - (D_v∇²v + R_v)

    # Combined PDE loss (mean squared error)
    pde_loss = (res_u**2 + res_v**2).mean()
    
    return pde_loss, observation_loss_a, observation_loss_u


def load_reaction_diffusion_sample(datapath, offset, device):
    with h5py.File(datapath, 'r') as file:
        sample_keys = sorted(
            key for key in file.keys()
            if isinstance(file[key], h5py.Group) and 'data' in file[key]
        )
        if sample_keys:
            sample = file[sample_keys[offset]]
            arr = np.asarray(sample['data'])
            init_idx = 50 if datapath.endswith('2D_diff-react_NA_NA.h5') and arr.shape[0] > 50 else 0
            coef = np.stack([arr[init_idx, :, :, 0], arr[init_idx, :, :, 1]], axis=0)
            sol = np.stack([arr[-1, :, :, 0], arr[-1, :, :, 1]], axis=0)
            pde_params = reaction_diffusion_params(file, sample)
        elif 'u' in file and 'v' in file:
            u_data = np.asarray(file['u'])
            v_data = np.asarray(file['v'])
            coef = np.stack([reaction_diffusion_frame(u_data, offset, 0), reaction_diffusion_frame(v_data, offset, 0)], axis=0)
            sol = np.stack([reaction_diffusion_frame(u_data, offset, -1), reaction_diffusion_frame(v_data, offset, -1)], axis=0)
            pde_params = reaction_diffusion_params(file, None)
        else:
            raise KeyError(f"{datapath} must contain sample groups with data or root u/v datasets")

    coef = torch.tensor(coef, dtype=torch.float64, device=device).unsqueeze(0)
    sol = torch.tensor(sol, dtype=torch.float64, device=device).unsqueeze(0)
    return coef, sol, pde_params


def reaction_diffusion_frame(arr, offset, time_index):
    sample = np.asarray(arr[offset])
    if sample.ndim == 2:
        return sample
    if sample.ndim == 3:
        if sample.shape[0] == sample.shape[1]:
            return sample[:, :, time_index]
        if sample.shape[1] == sample.shape[2]:
            return sample[time_index, :, :]
    raise ValueError(f"Cannot infer reaction_diffusion frame from shape={sample.shape}")


def reaction_diffusion_params(file, sample):
    params = {}
    for name in ('T', 'total_time', 'D_u', 'Du', 'D_v', 'Dv', 'k'):
        value = attr_value(file, sample, name)
        if value is not None:
            params[name] = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    if 'total_time' in params and 'T' not in params:
        params['T'] = params['total_time']
    if 'Du' in params and 'D_u' not in params:
        params['D_u'] = params['Du']
    if 'Dv' in params and 'D_v' not in params:
        params['D_v'] = params['Dv']
    return params


def attr_value(file, sample, name):
    if sample is not None and name in sample.attrs:
        return sample.attrs[name]
    if name in file.attrs:
        return file.attrs[name]
    return None

def generate_reaction_diffusion(config):
    """Generate Reaction Diffusion equation."""
    setup_file_name = f"{config['data']['offset']}_obs({config['data']['obs_size']})_zeta({config['generate']['zeta_obs_a']},{config['generate']['zeta_obs_u']},{config['generate']['zeta_pde']})_step({config['test']['iterations']})_"

    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    obs_size = config['data']['obs_size']

    a_GT, u_GT, pde_params = load_reaction_diffusion_sample(datapath, offset, device)
    
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    
    sigma_min = config['generate']['sigma_min']
    sigma_max = config['generate']['sigma_max']
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    num_steps = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    rho = config['generate']['rho']
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    if config['generate']['full']:
        known_index_a = torch.ones((net.img_resolution, net.img_resolution), dtype=torch.float32, device=device)
        known_index_u = torch.ones((net.img_resolution, net.img_resolution), dtype=torch.float32, device=device)
    else:
        known_index_a = random_index(obs_size, net.img_resolution, seed=1, device=device)
        known_index_u = random_index(obs_size, net.img_resolution, seed=0, device=device)
    
    ############################ Sample the data ############################
    time_start = time.time()
    loss = {'global_a': [], 'global_u': []}

    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Scale the data back
        a_N = x_N[:,0:2,:,:]
        u_N = x_N[:,2:4,:,:]
        a_N = (a_N*1.6).to(torch.float64)
        u_N = (u_N*1.6).to(torch.float64)
        
        # Compute the loss
        pde_loss, observation_loss_a, observation_loss_u = get_reaction_diffusion_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, pde_params, device=device)
        L_pde = torch.norm(pde_loss, 2)/(net.img_resolution * net.img_resolution)
        L_obs_a = torch.norm(observation_loss_a, 2)#/obs_size
        L_obs_u = torch.norm(observation_loss_u, 2)#/obs_size
        grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
        zeta_obs_a = config['generate']['zeta_obs_a']
        zeta_obs_u = config['generate']['zeta_obs_u']
        zeta_pde = config['generate']['zeta_pde']
        if i <= 0.8 * num_steps:
            x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
        else:
            x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde

        a_eval = x_next[:,0:2,:,:]
        u_eval = x_next[:,2:4,:,:]
        a_eval = (a_eval*1.6).to(torch.float64)
        u_eval = (u_eval*1.6).to(torch.float64)
        re_a_eval = torch.norm(a_eval - a_GT, 2) / torch.norm(a_GT, 2)
        re_u_eval = torch.norm(u_eval - u_GT, 2) / torch.norm(u_GT, 2)

        loss['global_a'].append(re_a_eval.item())
        loss['global_u'].append(re_u_eval.item())

    time_end = time.time()
    time_eval = time_end - time_start

    ############################ Save the data ############################
    x_final = x_next
    a_final = x_final[:,0:2,:,:]
    u_final = x_final[:,2:4,:,:]
    a_final = (a_final*1.6).to(torch.float64)
    u_final = (u_final*1.6).to(torch.float64)
    if config['generate']['guide']:
        relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
        relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
        print(f'Relative error of a: {relative_error_a}')
        print(f'Relative error of u: {relative_error_u}')
        
    a_final = a_final.detach().cpu().numpy()
    u_final = u_final.detach().cpu().numpy()
    
    if config['output']['save']:
        with open(f"{config['output']['file_path']}/{config['generate']['problem']}/{config['data']['name']}_{setup_file_name}_results.pkl", 'wb') as f:
            pickle.dump({
                'obs_index': {'known_index_a': known_index_a, 'known_index_u': known_index_u},
                'coef_final': a_final,
                'sol_final': u_final,
                'loss': loss,
                'time': time_eval
                }, f)
    
    if config['output']['return']:
        print('Done.')
        return a_final, u_final, known_index_a, known_index_u
