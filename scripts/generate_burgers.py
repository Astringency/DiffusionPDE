import tqdm
import time
import pickle
from pathlib import Path
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
from .burgers_observations import make_burgers_mask

def get_burger_loss(u, u_GT, mask):
    """Return the loss of the Burgers' equation and the observation loss."""
    # Use the generated sample as the source of truth for device placement.
    # This keeps the finite-difference kernels, ground truth, and sensor mask on
    # the same GPU even when the configured device is not the default cuda:0.
    device = u.device
    u_GT = u_GT.to(device=device, dtype=u.dtype)
    mask = mask.to(device=device, dtype=u.dtype)
    u = u.view(1, 1, 128, 128)
    u_GT = u_GT.view(1, 1, 128, 128)
    deriv_t = torch.tensor([[-1], [0], [1]], dtype=u.dtype, device=device).view(1, 1, 3, 1) / 2
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=u.dtype, device=device).view(1, 1, 1, 3) / 2
    u_t = F.conv2d(u, deriv_t, padding=(1, 0)) 
    u_x = F.conv2d(u, deriv_x, padding=(0, 1)) 
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))

    pde_loss = u_t + u * u_x - 0.01 * u_xx
    pde_loss = pde_loss.squeeze()
    observation_loss = u - u_GT
    observation_loss = observation_loss.squeeze()
    observation_loss = observation_loss * mask
    return pde_loss, observation_loss

def generate_burgers(config):
    """Generate Burgers' equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
    ground_truth = data['output'][offset, :, :]
    ground_truth = torch.tensor(ground_truth, dtype=torch.float64, device=device)
    if ground_truth.shape != (128, 128):
        raise ValueError('Burgers requires a complete (time, space) = (128, 128) trajectory')
    
    batch_size = config['generate']['batch_size']
    if batch_size != 1:
        raise ValueError('Burgers uses generate.batch_size=1; use --batch for multiple inputs')
    if config['test']['iterations'] < 2:
        raise ValueError('Burgers EDM sampling requires at least 2 iterations')
    seed = config['generate']['seed'] + (offset if config['generate'].get('seed_per_sample', False) else 0)
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device).eval().requires_grad_(False)
    
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
    selected_index = make_burgers_mask(config['data'], ground_truth.shape, device=device)
    
    ############################ Sample the data ############################
    time_start = time.time()
    loss = []

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
        x_N = (x_N * 1.415).to(torch.float64)
        
        # Compute the loss
        pde_loss, observation_loss = get_burger_loss(x_N, ground_truth, selected_index)
        L_pde = torch.norm(pde_loss, 2)/(128*128)
        # Preserve the paper's native guidance normalization for BOTH layouts,
        # including random-500; changing this denominator changes the sampler.
        L_obs = torch.norm(observation_loss, 2)/(128*5)
        grad_x_cur_obs = torch.autograd.grad(outputs=L_obs, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
        zeta_obs = config['generate']['zeta_obs']
        zeta_pde = config['generate']['zeta_pde']
        if i <= 0.8 * num_steps:
            x_next = x_next - zeta_obs * grad_x_cur_obs
        else:
            x_next = x_next - zeta_obs / 10 * grad_x_cur_obs - zeta_pde * grad_x_cur_pde

        x_eval = (x_next * 1.415).to(torch.float64)
        relative_error_eval = torch.norm(x_eval - ground_truth, 2)/torch.norm(ground_truth, 2)

        loss.append(relative_error_eval.item())
    
    time_end = time.time()
    time_eval = time_end - time_start
    
    ############################ Save the data ############################
    x_final = (x_next * 1.415).to(torch.float64)

    if config['generate']['guide']:
        relative_error = torch.norm(x_final - ground_truth, 2)/torch.norm(ground_truth, 2)
        print(f'Relative error of : {relative_error}')

    x_final = x_final.to('cpu').detach().numpy()

    # Save and return the results
    if config['output']['save']:
        # Save results
        Path(config['output']['file_path'], config['generate']['problem']).mkdir(parents=True, exist_ok=True)
        with open(f"{config['output']['file_path']}/{config['generate']['problem']}/{config['data']['name']}_{offset}_results.pkl", 'wb') as f:
            pickle.dump({
                'obs_index': {'known_sensor': selected_index.detach().cpu()},
                'sensor_mode': config['data'].get('sensor_mode', 'sensor_columns'),
                'num_observations': int(selected_index.sum().item()),
                'sample_seed': seed,
                'mask_seed': config['data'].get('mask_seed', 1) if config['data'].get('sensor_mode', 'sensor_columns') != 'sensor_columns' else 0,
                'num_steps': num_steps,
                'x_final': x_final,
                'loss': loss,
                'time': time_eval
                }, f)
    else:
        print("User declare no save.")

    if config['output']['return']:
        print('Done.')
        return x_final, selected_index
