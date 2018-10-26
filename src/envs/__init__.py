from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    return env(**kwargs)


REGISTRY = {}

try:
    from .starcraft1 import StarCraft1Env
    REGISTRY["sc1"] = partial(env_fn,
                              env=StarCraft1Env)
except Exception as e:
    print(e)

REGISTRY["sc2"] = partial(env_fn,
                          env=StarCraft2Env)

