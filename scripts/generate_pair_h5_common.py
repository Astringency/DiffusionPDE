import time
import pickle
import numpy as np
import torch
import h5py
import tqdm
import torch.nn.functional as F


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


def generate_pair_h5_pde(config, pde, coef_channels, sol_channels, param_specs, loss_func):
    """Generate endpoint-pair HDF5 PDE samples with DiffusionPDE's EDM sampler."""
    setup_file_name = f"{config['data']['offset']}_obs({config['data']['obs_size']})_zeta({config['generate']['zeta_obs_a']},{config['generate']['zeta_obs_u']},{config['generate']['zeta_pde']})_step({config['test']['iterations']})_"

    ############################ Load data and network ############################
    datapath = config['data'].get('datapath', config['data'].get('data_path'))
    offset = config['data']['offset']
    device = config['generate']['device']
    obs_size = config['data']['obs_size']

    coef_GT, sol_GT, pde_params = load_pair_h5_sample(
        datapath,
        offset,
        coef_channels,
        sol_channels,
        param_specs,
        config.get('data', {}),
        device,
    )

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

        coef_N, sol_N = split_pair_state(x_N, coef_channels, sol_channels)

        # Compute the loss
        if config['generate']['guide']:
            pde_loss, observation_loss_a, observation_loss_u = loss_func(
                coef_N,
                sol_N,
                coef_GT,
                sol_GT,
                known_index_a,
                known_index_u,
                pde_params,
            )
            L_pde = torch.norm(pde_loss, 2)/(net.img_resolution * net.img_resolution)
            L_obs_a = torch.norm(observation_loss_a, 2)/obs_size
            L_obs_u = torch.norm(observation_loss_u, 2)/obs_size
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
        else:
            x_next = x_next

        coef_eval, sol_eval = split_pair_state(x_next, coef_channels, sol_channels)
        re_a_eval = torch.norm(coef_eval - coef_GT, 2) / torch.norm(coef_GT, 2)
        re_u_eval = torch.norm(sol_eval - sol_GT, 2) / torch.norm(sol_GT, 2)

        loss['global_a'].append(re_a_eval.item())
        loss['global_u'].append(re_u_eval.item())

    time_end = time.time()
    time_eval = time_end - time_start

    ############################ Save the data ############################
    coef_final, sol_final = split_pair_state(x_next, coef_channels, sol_channels)

    if config['generate']['guide']:
        relative_error_a = torch.norm(coef_final - coef_GT, 2) / torch.norm(coef_GT, 2)
        relative_error_u = torch.norm(sol_final - sol_GT, 2) / torch.norm(sol_GT, 2)
        print(f'Relative error of a: {relative_error_a}')
        print(f'Relative error of u: {relative_error_u}')

    coef_final = coef_final.detach().cpu().numpy()
    sol_final = sol_final.detach().cpu().numpy()

    # Save and return the results
    if config['output']['save']:
        with open(f"{config['output']['file_path']}/{config['generate']['problem']}/{config['data']['name']}_{setup_file_name}_results.pkl", 'wb') as f:
            pickle.dump({
                'obs_index': {'known_index_a': known_index_a, 'known_index_u': known_index_u},
                'coef_final': coef_final,
                'sol_final': sol_final,
                'loss': loss,
                'time': time_eval
                }, f)
    else:
        print("User declare no save.")

    if config['output']['return']:
        print('Done.')
        return coef_final, sol_final, known_index_a, known_index_u


def load_pair_h5_sample(datapath, offset, coef_channels, sol_channels, param_specs, data_config, device):
    with h5py.File(datapath, 'r') as file:
        input_data = np.asarray(file['input_data'][offset:offset + 1], dtype=np.float32)
        output_data = np.asarray(file['output_data'][offset:offset + 1], dtype=np.float32)
        pde_params = {}
        for name, (default, aliases) in param_specs.items():
            value = read_scalar_param(file, data_config, name, aliases, offset, default)
            if value is not None:
                pde_params[name] = torch.tensor([value], dtype=torch.float64, device=device)

    if input_data.shape[1] != coef_channels:
        raise ValueError(f"Expected {coef_channels} input channels, got {input_data.shape[1]}")
    if output_data.shape[1] != sol_channels:
        raise ValueError(f"Expected {sol_channels} output channels, got {output_data.shape[1]}")

    coef_GT = torch.tensor(input_data, dtype=torch.float64, device=device)
    sol_GT = torch.tensor(output_data, dtype=torch.float64, device=device)
    return coef_GT, sol_GT, pde_params


def read_scalar_param(file, data_config, name, aliases, offset, default):
    for key in (name, *aliases):
        if key in data_config:
            return float(data_config[key])
    for key in (name, *aliases):
        if key in file:
            dataset = file[key]
            values = np.asarray(dataset[()] if dataset.shape == () else dataset[:], dtype=np.float64)
            if values.ndim == 0:
                return float(values)
            if values.shape[0] > offset:
                return float(values[offset].reshape(-1)[0])
            if values.shape[0] == 1:
                return float(values.reshape(-1)[0])
        if key in file.attrs:
            values = np.asarray(file.attrs[key], dtype=np.float64)
            return float(values.reshape(-1)[0])
    return default


def split_pair_state(x, coef_channels, sol_channels):
    expected = coef_channels + sol_channels
    if x.shape[1] != expected:
        raise ValueError(f"Expected {expected} channels, got {x.shape[1]}")
    coef = x[:, :coef_channels, :, :]
    sol = x[:, coef_channels:coef_channels + sol_channels, :, :]
    return coef, sol


def get_heat_loss(a, u, a_GT, u_GT, a_mask, u_mask, pde_params):
    observation_loss_a = (a - a_GT) * a_mask
    observation_loss_u = (u - u_GT) * u_mask
    alpha = param_field(pde_params, 'alpha', u, default=1.0)
    time_scale = time_scale_field(pde_params, u)
    u_mid = 0.5 * (a + u)
    pde_loss = (u - a) / time_scale - alpha * laplacian(u_mid)
    return zero_boundary(pde_loss), observation_loss_a, observation_loss_u


def get_wave_loss(a, u, a_GT, u_GT, a_mask, u_mask, pde_params):
    observation_loss_a = (a - a_GT) * a_mask
    observation_loss_u = (u - u_GT) * u_mask
    c = param_field(pde_params, 'c', u[:, 0:1], default=1.0)
    u0, v0 = a[:, 0:1], a[:, 1:2]
    u_t, v_t = u[:, 0:1], u[:, 1:2]
    time_scale = time_scale_field(pde_params, u_t)
    u_mid = 0.5 * (u0 + u_t)
    v_mid = 0.5 * (v0 + v_t)
    res_u = (u_t - u0) / time_scale - v_mid
    res_v = (v_t - v0) / time_scale - (c**2) * laplacian(u_mid)
    pde_loss = torch.cat([zero_boundary(res_u), zero_boundary(res_v)], dim=1)
    return pde_loss, observation_loss_a, observation_loss_u


def get_advection_diffusion_loss(a, u, a_GT, u_GT, a_mask, u_mask, pde_params):
    observation_loss_a = (a - a_GT) * a_mask
    observation_loss_u = (u - u_GT) * u_mask
    bx = param_field(pde_params, 'b_x', u, default=0.0)
    by = param_field(pde_params, 'b_y', u, default=0.0)
    kappa = param_field(pde_params, 'kappa', u, default=1.0)
    time_scale = time_scale_field(pde_params, u)
    u_mid = 0.5 * (a + u)
    pde_loss = (u - a) / time_scale + bx * dx(u_mid) + by * dy(u_mid) - kappa * laplacian(u_mid)
    return zero_boundary(pde_loss), observation_loss_a, observation_loss_u


def get_steady_heat_conduction_loss(a, u, a_GT, u_GT, a_mask, u_mask, pde_params):
    observation_loss_a = (a - a_GT) * a_mask
    observation_loss_u = (u - u_GT) * u_mask
    conductivity = (1.0 + 0.05 * (u - 298.0)).clamp_min(0.1)
    u_d = param_field(pde_params, 'u_D', u, default=298.0)
    pde_loss = steady_heat_residual_with_boundary(u, conductivity, a[:, :1], u_d)
    return pde_loss, observation_loss_a, observation_loss_u


def param_field(params, name, reference, default):
    if name not in params:
        return torch.full((reference.shape[0], 1, 1, 1), float(default), dtype=reference.dtype, device=reference.device)
    value = torch.as_tensor(params[name], dtype=reference.dtype, device=reference.device).reshape(-1)
    if value.numel() == 1:
        value = value.repeat(reference.shape[0])
    if value.numel() != reference.shape[0]:
        raise ValueError(f"PDE parameter {name!r} must have batch length {reference.shape[0]}, got {tuple(value.shape)}")
    return value.view(reference.shape[0], 1, 1, 1)


def time_scale_field(params, reference, default=1.0):
    for name in ('T', 'total_time', 'dt'):
        if name in params:
            return param_field(params, name, reference, default=1.0)
    return torch.full((reference.shape[0], 1, 1, 1), float(default), dtype=reference.dtype, device=reference.device)


def laplacian(u):
    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    padded = F.pad(u, (1, 1, 1, 1), 'constant', 0)
    return (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4.0 * u
    ) / (h**2)


def zero_boundary(x):
    y = x.clone()
    if y.shape[-1] > 1 and y.shape[-2] > 1:
        y[..., 0, :] = 0
        y[..., -1, :] = 0
        y[..., :, 0] = 0
        y[..., :, -1] = 0
    return y


def dx(f):
    h = 1.0 / max(int(f.shape[-1]) - 1, 1)
    padded = F.pad(f, (1, 1, 0, 0), mode='replicate')
    return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * h)


def dy(f):
    h = 1.0 / max(int(f.shape[-2]) - 1, 1)
    padded = F.pad(f, (0, 0, 1, 1), mode='replicate')
    return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * h)


def steady_heat_residual_with_boundary(u, conductivity, source, u_d):
    residual = nonlinear_heat_conduction_residual(u, conductivity, source)
    h_size = int(u.shape[-2])
    w_size = int(u.shape[-1])
    if h_size <= 0 or w_size <= 0:
        return residual
    dx_val = 1.0 / max(w_size - 1, 1)
    dy_val = 1.0 / max(h_size - 1, 1)
    u_d_row = u_d[..., 0, 0].unsqueeze(-1)
    residual[..., 0, :] = u[..., 0, :] - u_d_row
    if h_size > 1:
        residual[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy_val
    if w_size > 1 and h_size > 2:
        residual[..., 1:-1, 0] = (u[..., 1:-1, 0] - u[..., 1:-1, 1]) / dx_val
        residual[..., 1:-1, -1] = (u[..., 1:-1, -1] - u[..., 1:-1, -2]) / dx_val
    return residual


def nonlinear_heat_conduction_residual(u, conductivity, source):
    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    inv_h2 = 1.0 / (h**2)
    residual = torch.zeros_like(u[:, :1])
    if u.shape[-2] <= 2 or u.shape[-1] <= 2:
        return residual
    center_u = u[:, :, 1:-1, 1:-1]
    center_l = conductivity[:, :, 1:-1, 1:-1]
    accum = torch.zeros_like(center_u)
    for neighbor_u, neighbor_l in (
        (u[:, :, :-2, 1:-1], conductivity[:, :, :-2, 1:-1]),
        (u[:, :, 2:, 1:-1], conductivity[:, :, 2:, 1:-1]),
        (u[:, :, 1:-1, :-2], conductivity[:, :, 1:-1, :-2]),
        (u[:, :, 1:-1, 2:], conductivity[:, :, 1:-1, 2:]),
    ):
        face = 0.5 * (center_l + neighbor_l) * inv_h2
        accum = accum + face * (center_u - neighbor_u)
    residual[:, :, 1:-1, 1:-1] = accum - source[:, :, 1:-1, 1:-1]
    return residual
