import yaml
from argparse import ArgumentParser
from scripts import generate_burgers, generate_darcy, generate_poisson, generate_helmholtz, generate_ns_nonbounded, generate_ns_bounded, generate_shallow_water, generate_reaction_diffusion, generate_heat, generate_wave, generate_advection_diffusion, generate_steady_heat_conduction

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--problem', type=str, default = "both", help='Problem to solve')
    parser.add_argument('--batch', type=int, default = 1, help='Sample size')
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

    for i in range(batch):
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
        elif name == 'NS-Bounded':
            print('Solving bounded NS equation...')
            generate_ns_bounded(config)
        elif name == 'Shallow-water':
            print('Solving Shallow Water equation...')
            generate_shallow_water(config)
        elif name == 'Reaction-Diffusion' or name == 'reaction_diffusion':
            print('Solving Reaction Diffusion equation...')
            generate_reaction_diffusion(config)
        elif name == 'Heat':
            print('Solving Heat equation...')
            generate_heat(config)
        elif name == 'Wave':
            print('Solving Wave equation...')
            generate_wave(config)
        elif name == 'Advection-Diffusion':
            print('Solving Advection-Diffusion equation...')
            generate_advection_diffusion(config)
        elif name == 'Steady-Heat-Conduction':
            print('Solving Steady Heat Conduction equation...')
            generate_steady_heat_conduction(config)
        else:
            print('PDE not found')
            exit(1)
