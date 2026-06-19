from scripts.generate_pair_h5_common import generate_pair_h5_pde, get_steady_heat_conduction_loss


def generate_steady_heat_conduction(config):
    """Generate Steady Heat Conduction equation."""
    param_specs = {
        'u_D': (298.0, ()),
    }
    return generate_pair_h5_pde(config, 'steady_heat_conduction', 1, 1, param_specs, get_steady_heat_conduction_loss)
