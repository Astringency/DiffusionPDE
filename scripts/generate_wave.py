from scripts.generate_pair_h5_common import generate_pair_h5_pde, get_wave_loss


def generate_wave(config):
    """Generate Wave equation."""
    param_specs = {
        'c': (1.0, ('fixed_c',)),
        'T': (None, ()),
        'total_time': (None, ()),
        'dt': (None, ()),
    }
    return generate_pair_h5_pde(config, 'wave', 2, 2, param_specs, get_wave_loss)
