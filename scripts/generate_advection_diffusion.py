from scripts.generate_pair_h5_common import generate_pair_h5_pde, get_advection_diffusion_loss


def generate_advection_diffusion(config):
    """Generate Advection-Diffusion equation."""
    param_specs = {
        'b_x': (0.0, ()),
        'b_y': (0.0, ()),
        'kappa': (1.0, ()),
        'T': (None, ()),
        'total_time': (None, ()),
        'dt': (None, ()),
    }
    return generate_pair_h5_pde(config, 'advection_diffusion', 1, 1, param_specs, get_advection_diffusion_loss)
