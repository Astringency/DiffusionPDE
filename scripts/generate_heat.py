from scripts.generate_pair_h5_common import generate_pair_h5_pde, get_heat_loss


def generate_heat(config):
    """Generate Heat equation."""
    param_specs = {
        'alpha': (1.0, ('fixed_alpha',)),
        'T': (None, ()),
        'total_time': (None, ()),
        'dt': (None, ()),
    }
    return generate_pair_h5_pde(config, 'heat', 1, 1, param_specs, get_heat_loss)
