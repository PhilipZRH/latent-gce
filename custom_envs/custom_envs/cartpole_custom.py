import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


REWARD_SCHEME = 1


class CartPoleCustomEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates
        self.tau = 0.1
        self.kinematics_integrator = 'euler'

        self.x_threshold = 0.5
        self.x_dot_threshold = 2

        # Cap max angular velocity
        self.theta_dot_threshold = 8

        high = np.array([
            self.x_threshold,
            self.x_dot_threshold,
            1,
            1,
            self.theta_dot_threshold])

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.sum_sq_dist = 0
        self.t = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * action[0]
        if x <= -self.x_threshold and force < 0:
            force = 0
        if x >= self.x_threshold and force > 0:
            force = 0
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if x < -self.x_threshold:
            x = -self.x_threshold
            x_dot = 0
        if x > self.x_threshold:
            x = self.x_threshold
            x_dot = 0
        theta_dot = np.clip(theta_dot, -self.theta_dot_threshold, self.theta_dot_threshold)
        x_dot = np.clip(x_dot, -self.x_dot_threshold, self.x_dot_threshold)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.state = (x, x_dot, theta, theta_dot)
        if REWARD_SCHEME == 0:
            if np.abs(theta) < np.pi / 10:
                reward = 1
            else:
                reward = 0
        elif REWARD_SCHEME == 1:
            reward = - theta ** 2
        else:
            reward = 0
        done = False
        self.sum_sq_dist += theta ** 2
        self.t += 1

        info = {
            'avg_sq_dist': self.sum_sq_dist / self.t,
        }

        return self.get_obs(), reward, done, info

    def get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])

    def set_state(self, state):
        x, x_dot, cos, sin, theta_dot = state
        theta = np.arctan2(sin, cos)
        self.state = (x, x_dot, theta, theta_dot)

    def reset(self):
        # high = np.array([self.x_threshold, 0.05, np.pi, 0.05])
        # self.state = self.np_random.uniform(low=-high, high=high)
        high = np.array([self.x_threshold, 0.05, 0.05, 0.05])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.state[2] += np.pi
        self.sum_sq_dist = 0
        self.t = 0
        return self.get_obs()

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        world_width = 2.4 * 2
        scale = screen_width/world_width
        carty = 200  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length) * 1.25
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = cartheight/4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
