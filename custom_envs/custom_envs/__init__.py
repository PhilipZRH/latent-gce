from gym.envs.registration import register

# Ball in box environments
from custom_envs.ball_in_box import BallInBoxEnv
# Pendulum environments
from custom_envs.pendulum_pixels import PendulumPixelsEnv
from custom_envs.pendulum_custom import PendulumCustomEnv
# Other environments
from custom_envs.mountain_car_custom import MountainCarCustomEnv
from custom_envs.cartpole_custom import CartPoleCustomEnv

register(
    id='BallInBox-v0',
    entry_point=BallInBoxEnv,
    max_episode_steps=200,
)

# Pendulum environments
register(
    id='PendulumPixels-v0',
    entry_point=PendulumPixelsEnv,
    max_episode_steps=200,
)

register(
    id='PendulumCustom-v0',
    entry_point=PendulumCustomEnv,
    max_episode_steps=200,
)

register(
    id='CartPoleCustom-v0',
    entry_point=CartPoleCustomEnv,
    max_episode_steps=200,
)

register(
    id='MountainCarCustom-v0',
    entry_point=MountainCarCustomEnv,
    max_episode_steps=200,
)
