"""
Predefined wind scenarios for the sailing challenge.
Each wind scenario defines a starting wind configuration.
"""

TRAINING_1 = {
    'wind_init_params': {
        'base_speed': 10.0,
        'base_max_rotation_angle_degree': 10,
        'pattern' : ( # 3x3 winds (not restricted to 8 directions)
            ((1, 1), (0, -1), (0, -1)),
            ((1, 1), (0, -1), (0, -1)),
            ((1, -1), (-0.55, 1), (-0.55, 1)),
        )
    },  # below are the param of the Gaussian distribution from which noise is sampled at each time-step
    'wind_evol_params' : {
        'mean_rotation_angle_degree': 3,
        'std_rotation_angle_degree': 0.8,
    }
}

TRAINING_2 = {
    'wind_init_params': {
        'base_speed': 10.0,
        'base_max_rotation_angle_degree': 10,
        'pattern' : ( # 3x3 winds (not restricted to 8 directions)
            ((0, -1), (0, -1), (1, 1)),
            ((0, -1), (0, -1), (1, 1)),
            ((-0.55, 1), (-0.55, 1),(1, -1)),
        )
    },  # below are the param of the Gaussian distribution from which noise is sampled at each time-step
    'wind_evol_params' : {
        'mean_rotation_angle_degree': 3,
        'std_rotation_angle_degree': 0.8,
    }
}

TRAINING_3 = {
    'wind_init_params': {
        'base_speed': 10.0,
        'base_max_rotation_angle_degree': 10,
        'pattern' : ( # 3x3 winds (not restricted to 8 directions)
            ((1, 1), (0, 1), (-1, 1)),
            ((1, 1), (0, 1), (1, 1)),
            ((-1, 1), (0, 1),(1, 1)),
        )
    },  # below are the param of the Gaussian distribution from which noise is sampled at each time-step
    'wind_evol_params' : {
        'mean_rotation_angle_degree': 3,
        'std_rotation_angle_degree': 0.8,
    }
}

# Dictionary mapping wind scenario names to their parameters
WIND_SCENARIOS = {
    'training_1': TRAINING_1,
    'training_2': TRAINING_2,
    'training_3': TRAINING_3,
}

def get_wind_scenario(name):
    """
    Get the parameters for a specific wind scenario.

    Args:
        name: String, one of ['training_1', 'training_2', 'training_3']

    Returns:
        Dictionary containing wind_init_params and wind_evol_params

    Note:
        To create a static environment (no wind evolution), use:
        env = SailingEnv(**get_wind_scenario(name), static_wind=True)

        To create a dynamic environment (default):
        env = SailingEnv(**get_wind_scenario(name))
    """
    if name not in WIND_SCENARIOS:
        raise ValueError(f"Unknown wind scenario '{name}'. Available wind scenarios: {list(WIND_SCENARIOS.keys())}")
    return WIND_SCENARIOS[name]
