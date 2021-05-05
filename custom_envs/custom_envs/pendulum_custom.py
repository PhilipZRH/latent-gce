import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

R = 0.01
K = 1


class PendulumCustomEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, reward_mode=0):
        self.dt = .1
        self.max_speed = 8
        self.max_torque = 2.
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.reward_mode = reward_mode

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.np_random = None
        self.state = np.array([0, 0])
        self.seed()
        self.emp_map = None

        self.sum_sq_dist = 0
        self.t = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        friction_coeff = 0

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u - friction_coeff*thdot) * dt

        newth = angle_normalize(th + newthdot*dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        if self.reward_mode == 0:
            costs = th**2 + .1*thdot**2 + .001*(u**2)
            r = -costs
        else:
            r = 0

        self.state = np.array([newth, newthdot])
        self.sum_sq_dist += newth ** 2
        self.t += 1

        info = {
            'avg_sq_dist': self.sum_sq_dist / self.t,
        }

        return self._get_obs(), r, False, info

    def reset(self):
        # Reset the pendulum around the bottom.
        high = np.array([0.5, 0.5])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.state[0] += np.pi
        self.last_u = None

        self.sum_sq_dist = 0
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        # return self.state
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def set_state(self, state):
        theta = np.arctan2(state[1], state[0])
        self.state = np.array([theta, state[2]])

    def get_obs(self):
        return self._get_obs()

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "../classic_control/assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi
