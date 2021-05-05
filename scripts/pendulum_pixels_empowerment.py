import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import latent_gce
import custom_envs
from pathlib import Path

GPU = True

PATH_PREFIX = './pendulum_pixels/'

# Environment options:
NUM_STEPS_OBSERVATION = 2
T = 4

# Trajectory options: 1: Collect data. 2: Load data. 3: None.
DATA_OPTION = 1
DATA_DIR = PATH_PREFIX
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Pre-trained model options: 1: Cold-start. 2: Load model.
MODEL_OPTION = 1
MODEL_DIR = PATH_PREFIX + 'model/pendulum_pixels.ckpt'

# Training options:
LOG_DIR = PATH_PREFIX + 'logs'
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TRAINING_EPOCH = 120
ADDITIONAL_TRAINING_EPOCH = 0

env = gym.make('PendulumPixels-v0')
env.reset()

if MODEL_OPTION == 1:
    load_model = None
else:
    load_model = MODEL_DIR

model = latent_gce.LatentGCEImage(env=env,
                                  num_steps_observation=NUM_STEPS_OBSERVATION,
                                  action_raw_dim=T,
                                  state_latent_dimension=32,
                                  action_latent_dimension=32,
                                  learning_rate=LEARNING_RATE,
                                  log_dir=LOG_DIR,
                                  use_gpu=GPU,
                                  load_model=load_model)

if DATA_OPTION == 2:
    train_data = {
        'obs': np.load(DATA_DIR + 'obs.npy'),
        'actions': np.load(DATA_DIR + 'actions.npy'),
        'obs_t': np.load(DATA_DIR + 'obs_t.npy')
    }
elif DATA_OPTION == 1:
    trajectories_params = {'total_steps': 150000,
                           'steps_per_episode': 4,
                           'max_action': [2],
                           'mode': 'image'}
    input_params = {'num_steps_observation': NUM_STEPS_OBSERVATION,
                    'T': T}

    env_name = 'PendulumPixels-v0'
    train_data = latent_gce.mp_collect_input(env_name, trajectories_params, input_params, existing_ray=False)
    np.save(DATA_DIR + 'obs.npy', train_data['obs'])
    np.save(DATA_DIR + 'actions.npy', train_data['actions'])
    np.save(DATA_DIR + 'obs_t.npy', train_data['obs_t'])
else:
    train_data = None

if MODEL_OPTION == 1:
    model.train(train_data, batch_size=BATCH_SIZE, num_epoch=TRAINING_EPOCH)
    model.save(MODEL_DIR)

if ADDITIONAL_TRAINING_EPOCH:
    model.train(train_data, batch_size=BATCH_SIZE, num_epoch=ADDITIONAL_TRAINING_EPOCH)
    model.save(MODEL_DIR)

x1 = np.linspace(0, 2 * math.pi, 65)
x2 = np.linspace(-8, 8, 65)

X1, X2 = np.meshgrid(x1, x2)
targets = []
print('Collecting images...')
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = X1[i][j]
        y = X2[i][j]
        env.unwrapped.state = np.array([x, y])
        obs_list = [env.unwrapped.get_obs() / 255]
        for step in range(NUM_STEPS_OBSERVATION - 1):
            obs_list.append(env.step([0])[0] / 255)
        obs = np.concatenate(obs_list, axis=2)
        targets.append(obs)

result = model.water_filling_from_observations(targets)

Z = result.reshape(X1.shape)

print('Generating encoding visualizations')
env.unwrapped.state = np.array([0, 0])
obs1 = env.unwrapped.get_obs()
plt.imsave('org.png', obs1)
obs2 = env.step([0])[0]
obs = np.concatenate([obs1, obs2], axis=2)
i = 0
for a in [np.array([-1, -1, -1, -1]), np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])]:
    pred = model.draw_showcase(obs, a)
    img1 = pred[:, :, :3]
    img2 = pred[:, :, 3:]
    plt.imsave(str(i) + '-1.png', img1)
    plt.imsave(str(i) + '-2.png', img2)
    i += 1

i = 0
for a_s in [np.array([-1, -1, -1, -1]), np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])]:
    env.unwrapped.state = np.array([0, 0])
    for a in a_s:
        env.step([a])
    img1 = env.unwrapped.get_obs()
    img2 = env.step([0])[0]
    plt.imsave('real-' + str(i) + '-1.png', img1)
    plt.imsave('real-' + str(i) + '-2.png', img2)
    i += 1

fig, ax = plt.subplots()
im = ax.imshow(Z, origin='lower', cmap='jet', extent=(0, 2 * math.pi, -8, 8))
ax.set_aspect('auto')
plt.colorbar(im)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot{\theta}$')

plt.show()
