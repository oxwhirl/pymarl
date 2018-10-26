# map parameter registry
map_param_registry = {
    "m5v5_c_far": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 60,
        "shield": False,
        "agent_race": 'Terran',
        "bot_race": 'Terran',
        "map_type": 'scm',
        "micro_battles": True},
    "dragoons_zealots": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "shield": False,
        "agent_race": 'Protoss',
        "bot_race": 'Protoss',
        "map_type": 'scm',
        "micro_battles": True},
    }

def get_map_params(map_name):
    return map_param_registry[map_name]

def map_present(map_name):
    return map_name in map_param_registry
