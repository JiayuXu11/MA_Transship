"""Tools for HARL."""

from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv





def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args, async_tf=True):
    """Make env for training."""
    def get_env_fn(rank):
        def init_env():
            if env_name == "multi_lt_transship":
                from harl.envs.multi_lt_transship.transship_multi_lt import (
                    MultiLtTransshipEnv,
                )
                env = MultiLtTransshipEnv(env_args)
            elif env_name == "multi_lt_transship_mechanism":
                from harl.envs.multi_lt_transship.transship_multi_lt_mechanism import (
                    MultiLtTransshipMechanismEnv,
                )
                env = MultiLtTransshipMechanismEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if (n_threads == 1) or (not async_tf):
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args, async_tf=True):
    """Make env for evaluation."""
    def get_env_fn(rank):
        def init_env():
            if env_name == "multi_lt_transship":
                from harl.envs.multi_lt_transship.transship_multi_lt import (
                    MultiLtTransshipEnv,
                )
                env = MultiLtTransshipEnv(env_args)
            elif env_name == "multi_lt_transship_mechanism":
                from harl.envs.multi_lt_transship.transship_multi_lt_mechanism import (
                    MultiLtTransshipMechanismEnv,
                )
                env = MultiLtTransshipMechanismEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if (n_threads == 1) or (not async_tf):
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs

    if env_name == "multi_lt_transship":
        from harl.envs.multi_lt_transship.transship_multi_lt import (
            MultiLtTransshipEnv,
        )
        env = MultiLtTransshipEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "multi_lt_transship_mechanism":
        from harl.envs.multi_lt_transship.transship_multi_lt_mechanism import (
            MultiLtTransshipMechanismEnv,
        )
        env = MultiLtTransshipMechanismEnv(env_args)
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError

    return env, manual_render, manual_expand_dims, manual_delay, env_num


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "multi_lt_transship":
        return envs.n_agents
    elif env == "multi_lt_transship_mechanism":
        return envs.n_agents
    else:
        print("Can not support the " + env + "environment.")
        raise NotImplementedError
