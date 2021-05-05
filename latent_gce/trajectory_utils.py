import numpy as np
import cv2
import gym
import ray


def collect_trajectories(env, params):
    total_steps = params['total_steps']
    steps_per_episode = params['steps_per_episode']
    max_action = np.array(params['max_action'])
    mode = params['mode']

    states = []
    actions = []
    episode_step_count = 0
    total_step_count = 0
    obs = env.reset()
    if mode == 'image':
        states.append([cv2.resize(obs, dsize=(64, 64)) / 255])
    elif mode == 'raw':
        states.append([env.unwrapped.state])
    else:
        states.append([obs])
    actions.append([])

    while total_step_count < total_steps:
        action = env.action_space.sample()
        actions[-1].append(np.array(action) / max_action)
        obs, reward, done, extra = env.step(action)
        if mode == 'image':
            states[-1].append(cv2.resize(obs, dsize=(64, 64)) / 255)
        elif mode == 'raw':
            states[-1].append(env.unwrapped.state)
        else:
            states[-1].append(obs)
        episode_step_count += 1
        total_step_count += 1

        if done or episode_step_count > steps_per_episode:
            obs = env.reset()
            if mode == 'image':
                states.append([cv2.resize(obs, dsize=(64, 64)) / 255])
            elif mode == 'raw':
                states.append([env.unwrapped.state])
            else:
                states.append([obs])
            actions.append([])
            episode_step_count = 0

    return states, actions


def collect_local_input(env, params):
    starting_state = params['starting_state']
    max_action = np.array(params['max_action'])
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    num_samples = params['num_samples']

    obs = []
    action_chains = []
    obs_t = []
    for i in range(num_samples):
        env.reset()
        env.unwrapped.state = starting_state
        complete_list = [cv2.resize(env.unwrapped.get_obs(), dsize=(64, 64)) / 255]
        action_list = []
        for t in range(T + num_steps_observation - 1):
            action = env.action_space.sample()
            o, reward, done, extra = env.step(action)
            complete_list.append(cv2.resize(o, dsize=(64, 64)) / 255)
            action_list.append(action / max_action)
        obs_list = complete_list[:num_steps_observation]
        obs_t_list = complete_list[T:]
        action_list = action_list[:T]
        obs.append(np.concatenate(obs_list, axis=2))
        action_chains.append(np.concatenate(action_list, axis=0))
        obs_t.append(np.concatenate(obs_t_list, axis=2))
    data = {'obs': np.array(obs),
            'actions': np.array(action_chains),
            'obs_t': np.array(obs_t)}
    return data


def collect_raw_local_input(env, params):
    starting_state = params['starting_state']
    max_action = np.array(params['max_action'])
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    num_samples = params['num_samples']

    obs = []
    action_chains = []
    obs_t = []
    for i in range(num_samples):
        env.reset()
        env.unwrapped.state = starting_state
        complete_list = [env.unwrapped.state]
        action_list = []
        for t in range(T + num_steps_observation - 1):
            action = env.action_space.sample()
            env.step(action)
            complete_list.append(env.unwrapped.state)
            action_list.append(action / max_action)
        obs_list = complete_list[:num_steps_observation]
        obs_t_list = complete_list[T:]
        action_list = action_list[:T]
        obs.append(np.concatenate(obs_list, axis=0))
        action_chains.append(np.concatenate(action_list, axis=0))
        obs_t.append(np.concatenate(obs_t_list, axis=0))
    data = {'obs': np.array(obs),
            'actions': np.array(action_chains),
            'obs_t': np.array(obs_t)}
    return data


@ray.remote
def ray_mj_raw_local_input(params):
    return mj_raw_local_input(params)


def mj_raw_local_input(params):
    env_name = params['env_name']
    starting_state = params['starting_state']
    max_action = np.array(params['max_action'])
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    num_samples = params['num_samples']

    env = gym.make(env_name)
    action_chains = []
    obs_t = []
    for i in range(num_samples):
        complete_list = [env.reset()]
        env.sim.set_state(starting_state)
        action_list = []
        for t in range(T + num_steps_observation - 1):
            action = env.action_space.sample()
            complete_list.append(env.step(action)[0])
            action_list.append(action / max_action)
        obs_t_list = complete_list[T:]
        action_list = action_list[:T]
        action_chains.append(np.concatenate(action_list, axis=0))
        obs_t.append(np.concatenate(obs_t_list, axis=0))
    data = {'actions': np.array(action_chains),
            'obs_t': np.array(obs_t)}
    return data


def mp_mj_raw_local_input(params, num_processes=16):
    ray.init()
    params['num_samples'] = params['num_samples'] // num_processes

    input_list = ray.get([ray_mj_raw_local_input.remote(params) for i in range(num_processes)])
    ray.shutdown()

    actions = np.concatenate([i['actions'] for i in input_list], axis=0)
    obs_t = np.concatenate([i['obs_t'] for i in input_list], axis=0)

    data = {'actions': actions,
            'obs_t': obs_t}
    return data


def trajectories_to_input(states_and_actions, params):
    states, actions = states_and_actions
    num_steps_observation = params['num_steps_observation']
    T = params['T']

    obs = []
    action_chains = []
    obs_t = []
    for i in range(len(states)):
        for j in range(len(states[i]) - T - num_steps_observation + 1):
            obs_list = [states[i][k] for k in range(j, j+num_steps_observation)]
            obs.append(np.concatenate(obs_list, axis=-1))

            action_list = [actions[i][k] for k in range(j, j+T)]
            action_chains.append(np.concatenate(action_list, axis=0))

            obs_t_list = [states[i][k] for k in range(j+T, j+T+num_steps_observation)]
            obs_t.append(np.concatenate(obs_t_list, axis=-1))

    data = {'obs': np.array(obs),
            'actions': np.array(action_chains),
            'obs_t': np.array(obs_t)}
    return data


@ray.remote
def trajectories_to_input_ray_task(states, actions, actions_range, num_steps_observation, T, neg_log_prob=None):
    obs = []
    action_chains = []
    obs_t = []
    chain_neg_log_prob = []

    for i in range(len(states) - T - num_steps_observation + 1):
        obs_list = [states[j] for j in range(i, i+num_steps_observation)]
        obs.append(np.concatenate(obs_list, axis=-1))

        action_list = [actions[j] / actions_range for j in range(i, i+T)]
        action_chains.append(np.concatenate(action_list, axis=0))
        if neg_log_prob is not None:
            chain_neg_log_prob.append(np.sum(neg_log_prob[j] for j in range(i, i+T)))

        obs_t_list = [states[j] for j in range(i+T, i+T+num_steps_observation)]
        obs_t.append(np.concatenate(obs_t_list, axis=-1))

    if len(obs) < 1:
        return None

    data = {'obs': np.array(obs),
            'actions': np.array(action_chains),
            'obs_t': np.array(obs_t)}
    if neg_log_prob:
        data['neg_log_prob'] = np.array(chain_neg_log_prob)
    return data


def mp_trajectories_to_input(states_and_actions, params, existing_ray=True, neg_log_prob=None):
    if not existing_ray:
        ray.init()
    states, actions = states_and_actions
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    actions_range = params['actions_range']

    if neg_log_prob is not None:
        data_array = ray.get([trajectories_to_input_ray_task.remote(states[i], actions[i], actions_range,
                                                                    num_steps_observation, T, neg_log_prob[i])
                              for i in range(len(states))])
    else:
        data_array = ray.get([trajectories_to_input_ray_task.remote(states[i], actions[i], actions_range,
                                                                    num_steps_observation, T)
                              for i in range(len(states))])
    data_array = [d for d in data_array if d]

    data = {'obs': np.concatenate([d['obs'] for d in data_array]),
            'actions': np.concatenate([d['actions'] for d in data_array]),
            'obs_t': np.concatenate([d['obs_t'] for d in data_array])}
    if neg_log_prob is not None:
        data['neg_log_prob'] = np.concatenate([d['neg_log_prob'] for d in data_array])
    return data


@ray.remote
def collect_input_ray_task(env_name, trajectories_params, input_params):
    env = gym.make(env_name)
    result = collect_trajectories(env, trajectories_params)
    # env.close()
    result = trajectories_to_input(result, input_params)
    return result


def mp_collect_input(env_name, trajectories_params, input_params, num_processes=20, existing_ray=True):
    if not existing_ray:
        ray.init()
    trajectories_params['total_steps'] = trajectories_params['total_steps'] // num_processes

    input_list = ray.get([collect_input_ray_task.remote(env_name, trajectories_params, input_params)
                          for i in range(num_processes)])
    if not existing_ray:
        ray.shutdown()

    obs = np.concatenate([i['obs'] for i in input_list], axis=0)
    actions = np.concatenate([i['actions'] for i in input_list], axis=0)
    obs_t = np.concatenate([i['obs_t'] for i in input_list], axis=0)

    data = {'obs': obs,
            'actions': actions,
            'obs_t': obs_t}
    return data


def collect_input_from_state(params, starting_states):
    obs = []
    action_chains = []
    obs_t = []

    env_name = params['env_name']
    max_action = np.array(params['actions_range'])
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    total_steps = params['total_steps']
    env = gym.make(env_name)
    for s in starting_states:
        env.reset()
        env.unwrapped.set_state(s)
        complete_list = [s]
        complete_action_list = []
        for t in range(total_steps):
            action = env.action_space.sample()
            o, r, d, info = env.step(action)
            complete_list.append(o)
            complete_action_list.append(action / max_action)
        for i in range(total_steps - T - num_steps_observation + 2):
            obs_list = complete_list[i:i+num_steps_observation]
            obs_t_list = complete_list[i+T:i+T+num_steps_observation]
            action_list = complete_action_list[i:i+T]
            obs.append(np.concatenate(obs_list, axis=0))
            action_chains.append(np.concatenate(action_list, axis=0))
            obs_t.append(np.concatenate(obs_t_list, axis=0))
    obs = np.array(obs)
    action_chains = np.array(action_chains)
    obs_t = np.array(obs_t)
    env.close()
    return obs, action_chains, obs_t


@ray.remote
def collect_input_from_state_ray_task(params, starting_states):
    return collect_input_from_state(params, starting_states)


def mp_collect_input_from_state(params, starting_states, num_processes=16, existing_ray=True):
    if not existing_ray:
        ray.init()
    spw = (len(starting_states) - 1) // num_processes + 1
    ray_fn = collect_input_from_state_ray_task
    results = ray.get([ray_fn.remote(params, starting_states[w * spw: w * spw + spw]) for w in range(num_processes)])
    obs = []
    action_chains = []
    obs_t = []
    for r in results:
        if len(r[0]) > 0:
            obs.append(r[0])
            action_chains.append(r[1])
            obs_t.append(r[2])
    obs = np.concatenate(obs, axis=0)
    action_chains = np.concatenate(action_chains, axis=0)
    obs_t = np.concatenate(obs_t, axis=0)

    if not existing_ray:
        ray.shutdown()

    return obs, action_chains, obs_t


def mj_collect_input_from_state(params, starting_states):
    obs = []
    action_chains = []
    obs_t = []

    env_name = params['env_name']
    max_action = np.array(params['actions_range'])
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    total_steps = params['total_steps']
    env = gym.make(env_name)
    for s in starting_states:
        env.reset()
        env.unwrapped.set_state(s[0], s[1])
        o, r, d, info = env.step(env.action_space.sample())
        complete_list = [o]
        complete_action_list = []
        for t in range(total_steps):
            action = env.action_space.sample()
            o, r, d, info = env.step(action)
            complete_list.append(o)
            complete_action_list.append(action / max_action)
        for i in range(total_steps - T - num_steps_observation + 2):
            obs_list = complete_list[i:i+num_steps_observation]
            obs_t_list = complete_list[i+T:i+T+num_steps_observation]
            action_list = complete_action_list[i:i+T]
            obs.append(np.concatenate(obs_list, axis=0))
            action_chains.append(np.concatenate(action_list, axis=0))
            obs_t.append(np.concatenate(obs_t_list, axis=0))
    obs = np.array(obs)
    action_chains = np.array(action_chains)
    obs_t = np.array(obs_t)
    return obs, action_chains, obs_t


@ray.remote
def mj_collect_input_from_states_ray_task(params, starting_states):
    return mj_collect_input_from_state(params, starting_states)


def mp_mj_collect_input_from_state(params, starting_states, num_processes=16, existing_ray=True):
    if not existing_ray:
        ray.init()
    spw = (len(starting_states) - 1) // num_processes + 1
    ray_fn = mj_collect_input_from_states_ray_task
    results = ray.get([ray_fn.remote(params, starting_states[w * spw: w * spw + spw]) for w in range(num_processes)])
    obs = []
    action_chains = []
    obs_t = []
    for r in results:
        if len(r[0]) > 0:
            obs.append(r[0])
            action_chains.append(r[1])
            obs_t.append(r[2])
    obs = np.concatenate(obs, axis=0)
    action_chains = np.concatenate(action_chains, axis=0)
    obs_t = np.concatenate(obs_t, axis=0)

    if not existing_ray:
        ray.shutdown()

    return obs, action_chains, obs_t


def mj_collect_input_from_state_return_final(params, starting_states):
    obs = []
    action_chains = []
    obs_t = []

    env_name = params['env_name']
    max_action = np.array(params['actions_range'])
    num_steps_observation = params['num_steps_observation']
    T = params['T']
    total_steps = params['total_steps']
    env = gym.make(env_name)
    for s in starting_states:
        env.reset()
        env.unwrapped.set_state(s[0], s[1])
        o, r, d, info = env.step(env.action_space.sample())
        complete_list = [o]
        complete_action_list = []
        for t in range(total_steps):
            action = env.action_space.sample()
            o, r, d, info = env.step(action)
            complete_list.append(o)
            complete_action_list.append(action / max_action)
        for i in range(total_steps - T - num_steps_observation + 2):
            obs_list = complete_list[i:i+num_steps_observation]
            obs_t_list = complete_list[i+T:i+T+num_steps_observation]
            action_list = complete_action_list[i:i+T]
            obs.append(np.concatenate(obs_list, axis=0))
            action_chains.append(np.concatenate(action_list, axis=0))
            obs_t.append(np.concatenate(obs_t_list, axis=0))
    obs = np.array(obs)
    action_chains = np.array(action_chains)
    obs_t = np.array(obs_t)
    qpos = env.unwrapped.sim.data.qpos.copy()
    qvel = env.unwrapped.sim.data.qvel.copy()
    mujoco_sim = np.array([qpos, qvel])
    return obs, action_chains, obs_t, mujoco_sim


@ray.remote
def mj_collect_input_from_states_return_final_ray_task(params, starting_states):
    return mj_collect_input_from_state_return_final(params, starting_states)


def mp_mj_collect_input_from_state_return_final(params, starting_states, num_processes=16, existing_ray=True):
    if not existing_ray:
        ray.init()
    spw = (len(starting_states) - 1) // num_processes + 1
    ray_fn = mj_collect_input_from_states_return_final_ray_task
    results = ray.get([ray_fn.remote(params, starting_states[w * spw: w * spw + spw]) for w in range(num_processes)])
    obs = []
    action_chains = []
    obs_t = []
    mujoco_sim = []
    for r in results:
        if len(r[0]) > 0:
            obs.append(r[0])
            action_chains.append(r[1])
            obs_t.append(r[2])
            mujoco_sim.append(r[3])
    obs = np.concatenate(obs, axis=0)
    action_chains = np.concatenate(action_chains, axis=0)
    obs_t = np.concatenate(obs_t, axis=0)
    mujoco_sim = np.array(mujoco_sim)

    if not existing_ray:
        ray.shutdown()

    return obs, action_chains, obs_t, mujoco_sim