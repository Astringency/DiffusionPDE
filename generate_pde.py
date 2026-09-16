import yaml
from argparse import ArgumentParser
from scripts import generate_burgers, generate_darcy, generate_poisson, generate_helmholtz, generate_ns_nonbounded

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--problem', type=str, default = "both", help='Problem to solve')
    parser.add_argument('--batch', type=int, default = 1, help='Sample size')
    parser.add_argument('--start_offset', type=int, default = 0, help='First dataset offset to sample')
    parser.add_argument('--step_size', type=int, default = 2000, help='step size')
    options = parser.parse_args()
    config_path = options.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']
    config['generate']['problem'] = options.problem

    if options.problem == 'forward' and name != "Burgers":
        config['generate']['zeta_obs_u'] = 0
    elif options.problem == 'inverse' and name != "Burgers":
        config['generate']['zeta_obs_a'] = 0
    else:
        pass

    config['test']['iterations'] = options.step_size

    batch = options.batch
    start_offset = options.start_offset

    for i in range(start_offset, start_offset + batch):
        config['data']['offset'] = i

        if name == 'Burgers':
            print('Solving Burgers equation...')
            generate_burgers(config)
        elif name == 'Darcy':
            print('Solving Darcy Flow equation...')
            generate_darcy(config)
        elif name == 'Poisson':
            print('Solving Poisson equation...')
            generate_poisson(config)
        elif name == 'Helmholtz':
            print('Solving Helmholtz equation...')
            generate_helmholtz(config)
        elif name == 'NS-NonBounded':
            print('Solving non-bounded NS equation...')
            generate_ns_nonbounded(config)
        else:
            print('PDE not found')
            exit(1)
