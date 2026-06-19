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

# from data.transform import PDEtransform

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

def generate_shallow_water_loss(a, u, a_GT, u_GT, a_mask, u_mask, resolution, device=torch.device('cuda')):
    "Generate observation loss"
    a_GT = a_GT.view(1, a.shape[1], resolution, resolution)
    u_GT = u_GT.view(1, u.shape[1], resolution, resolution)
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask

    # pde_loss = torch.tensor(0, dtype=torch.float32, device=device)
    
    # return pde_loss, observation_loss_a, observation_loss_u
    return observation_loss_a, observation_loss_u

def generate_shallow_water(config):
    """Generate non-bounded NS equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    obs_size = config['data']['obs_size']
    
    data = {}
    with h5py.File(datapath, "r") as f:
        for k in list(f.keys()):
            group = f[k]['data'] # type: ignore
            h0 = np.expand_dims(group['h'][0, :, :, 0], axis = 0) # type: ignore
            h = np.expand_dims(group['h'][-1, :, :, 0], axis = 0) # type: ignore
            if 'hu' in group and 'hv' in group:
                hu0 = np.expand_dims(group['hu'][0, :, :, 0], axis = 0) # type: ignore
                hu = np.expand_dims(group['hu'][-1, :, :, 0], axis = 0) # type: ignore
                hv0 = np.expand_dims(group['hv'][0, :, :, 0], axis = 0) # type: ignore
                hv = np.expand_dims(group['hv'][-1, :, :, 0], axis = 0) # type: ignore
                data[k] = np.stack([h0, hu0, hv0, h, hu, hv], axis = 1)
            else:
                data[k] = np.stack([h0, h], axis = 1)
    data = torch.tensor(np.concatenate(list(data.values()), axis=0)).to(torch.float32)

    state_channels = data.shape[1] // 2
    a_GT = data[offset, :state_channels, :, :]
    a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
    u_GT = data[offset, state_channels:, :, :]
    u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    if net.img_channels // 2 < state_channels:
        state_channels = net.img_channels // 2
        a_GT = a_GT[:state_channels]
        u_GT = u_GT[:state_channels]
    
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
        a_N = x_N[:,:state_channels,:,:]
        u_N = x_N[:,state_channels:state_channels * 2,:,:]
        a_N = a_N.to(torch.float64)
        u_N = u_N.to(torch.float64)
        
        # Compute the loss
        if config['generate']['guide']:
            # pde_loss, observation_loss_a, observation_loss_u = generate_shallow_water_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, net.img_resolution, device=device)
            observation_loss_a, observation_loss_u = generate_shallow_water_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, net.img_resolution, device=device)
            # L_pde = torch.norm(pde_loss, 2)/(net.img_resolution * net.img_resolution)
            L_obs_a = torch.norm(observation_loss_a, 2)/obs_size
            L_obs_u = torch.norm(observation_loss_u, 2)/obs_size
            grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
            # grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
            zeta_obs_a = config['generate']['zeta_obs_a']
            zeta_obs_u = config['generate']['zeta_obs_u']
            zeta_pde = config['generate']['zeta_pde']
            if i <= 0.8 * num_steps:
                x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
            else:
                x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) # - zeta_pde * grad_x_cur_pde
        else:
            x_next = x_next

        a_eval = x_next[:,:state_channels,:,:]
        u_eval = x_next[:,state_channels:state_channels * 2,:,:]
        a_eval = a_eval.to(torch.float64)
        u_eval = u_eval.to(torch.float64)
        re_a_eval = torch.norm(a_eval - a_GT, 2) / torch.norm(a_GT, 2)
        re_u_eval = torch.norm(u_eval - u_GT, 2) / torch.norm(u_GT, 2)

        loss['global_a'].append(re_a_eval.item())
        loss['global_u'].append(re_u_eval.item())
    
    time_end = time.time()
    time_eval = time_end - time_start
    
    ############################ Save the data ############################
    x_final = x_next
    a_final = x_final[:,:state_channels,:,:]
    u_final = x_final[:,state_channels:state_channels * 2,:,:]
    a_final = a_final.to(torch.float64)
    u_final = u_final.to(torch.float64)

    if config['generate']['guide']:
        relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
        relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
        print(f'Relative error of a: {relative_error_a}')
        print(f'Relative error of u: {relative_error_u}')
        
    a_final = a_final.detach().cpu().numpy()
    u_final = u_final.detach().cpu().numpy()

    # Save and return the results
    if config['output']['save']:
        # Save results
        with open(f"{config['output']['file_path']}/{config['generate']['problem']}/{config['data']['name']}_{offset}_results.pkl", 'wb') as f:
            pickle.dump({
                'obs_index': {'known_index_a': known_index_a, 'known_index_u': known_index_u},
                'coef_final': a_final,
                'sol_final': u_final,
                'loss': loss,
                'time': time_eval
                }, f)
    else:
        print("User declare no save.")

    if config['output']['return']:
        print('Done.')
        return a_final, u_final, known_index_a, known_index_u

    # if config['output']['return']:
    #     print("Done.")
    #     return {
    #             'obs_index': {'known_index_a': known_index_a, 'known_index_u': known_index_u},
    #             'coef_final': a_final,
    #             'sol_final': u_final,
    #             'loss': loss,
    #             'time': time_eval
    #         }
    # else:
    #     print("Done.")
    
