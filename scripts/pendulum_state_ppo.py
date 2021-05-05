import gym
import latent_gce
import os
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
import matplotlib.pyplot as plt
from pathlib import Path

ENV_NAME = 'PendulumCustom-v0'
SAVE_NP = True

print('##########')
print('Debugging...')
print('##########')

exp_name = 'pendulum-state-ppo'
exp_idx = '0'
T = 8
reward_weight = 0
emp_weight = 1

load = 'None'
save = '0'

N_ENVS = 5
emp_trajectory_options = {'env_name': ENV_NAME,
                          'T': T,
                          'num_steps_observation': 1,
                          'actions_range': 2,
                          'total_steps': T}
emp_options = {'reward_weight': reward_weight,
               'emp_weight': emp_weight,
               'buffer_size': 10000,
               'learning_rate': 1e-4,
               'obs_selection': None,
               'is_mujoco': False,
               'action_penalty': 0,
               'exp_emp': True,
               'uniform_actions': True,
               'logging': ['avg_sq_dist'],
               }


def env_maker(**kwargs):
    return gym.make(ENV_NAME, **kwargs)


env = make_vec_env(env_maker, n_envs=N_ENVS)
for e in env.envs:
    e.env._max_episode_steps = 200

# Prepare EMP landscape debugging
GRID_BINS = 65
debug_env = gym.make(ENV_NAME)
x1 = np.linspace(-np.pi, np.pi, GRID_BINS)
x2 = np.linspace(-8, 8, GRID_BINS)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros(X1.shape)
targets = []
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = X1[i][j]
        y = X2[i][j]
        debug_env.unwrapped.state = np.array([x, y])
        obs_list = [debug_env.unwrapped.get_obs()]
        for step in range(emp_trajectory_options['num_steps_observation'] - 1):
            debug_env.step([0])
            obs_list.append(debug_env.unwrapped.get_obs())
        obs = np.concatenate(obs_list)
        targets.append(obs)

model = latent_gce.GcePPO(exp_idx, MlpPolicy, env, emp_trajectory_options, emp_options,
                          verbose=1, tensorboard_log=exp_name, gamma=0.5,
                          n_steps=1000, learning_rate=0.001)
if load != 'None':
    model.load_parameters(load)

Path('./pendulum-state-ppo').mkdir(parents=True, exist_ok=True)

gce_result_array = []
policy_result_array = []
state_count_result_array = []
for i in range(50):
    model.learn(total_timesteps=10000, reset_num_timesteps=False, dump_log=True)
    gce_result = model.runner.gce_model.water_filling_from_observations(targets).reshape(X1.shape)
    policy_result = model.predict(targets)[0].flatten().reshape(X1.shape)
    state_count = np.zeros(X1.shape)
    theta_bin_width = 2 * np.pi / (GRID_BINS - 1)
    theta_dot_bin_width = 16 / (GRID_BINS - 1)
    angles = np.arctan2(model.obs[:, 1], model.obs[:, 0])
    angle_dots = model.obs[:, 2]
    angles_idx = np.clip((angles + np.pi + theta_bin_width / 2) // theta_bin_width, 0, GRID_BINS - 1)
    angle_dots_idx = np.clip((angle_dots + 8 + theta_dot_bin_width / 2) // theta_dot_bin_width, 0, GRID_BINS - 1)
    for idx in range(len(angles_idx)):
        state_count[int(angle_dots_idx[idx]), int(angles_idx[idx])] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(gce_result, interpolation='nearest', origin='lower', cmap='jet', extent=(-np.pi, np.pi, -8, 8))
    ax.set_aspect('auto')
    plt.colorbar(im)
    plt.savefig('./pendulum-state-ppo/%03d-emp.png' % (i + 1))
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(policy_result, interpolation='nearest', origin='lower', cmap='jet', extent=(-np.pi, np.pi, -8, 8))
    ax.set_aspect('auto')
    plt.colorbar(im)
    plt.savefig('./pendulum-state-ppo/%03d-policy.png' % (i + 1))
    plt.close(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(state_count, interpolation='nearest', origin='lower', cmap='jet', extent=(-np.pi, np.pi, -8, 8))
    ax.set_aspect('auto')
    plt.colorbar(im)
    plt.savefig('./pendulum-state-ppo/%03d-state.png' % (i + 1))
    plt.close(fig)

    gce_result_array.append(gce_result.copy())
    policy_result_array.append(policy_result.copy())
    state_count_result_array.append(state_count.copy())

if save != 'None':
    model.save(exp_name + '/' + save)
env.close()

if SAVE_NP:
    np.save('./pendulum-state-ppo/emp', np.array(gce_result_array))
    np.save('./pendulum-state-ppo/policy', np.array(policy_result_array))
    np.save('./pendulum-state-ppo/states', np.array(state_count_result_array))

print('Connecting images into videos... Make sure ffmpeg is installed!')
os.system('ffmpeg -i ./pendulum-state-ppo/%03d-emp.png -pix_fmt yuv420p ./pendulum-state-ppo/emp.mp4')
os.system('ffmpeg -i ./pendulum-state-ppo/%03d-policy.png -pix_fmt yuv420p ./pendulum-state-ppo/policy.mp4')
os.system('ffmpeg -i ./pendulum-state-ppo/%03d-state.png -pix_fmt yuv420p ./pendulum-state-ppo/state.mp4')

print('Finished creating videos. Deleting the images...')
os.system('rm ./pendulum-state-ppo/*.png')
