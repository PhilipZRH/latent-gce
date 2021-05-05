import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

IMG = False
REWARD = 0


class MountainCarCustomEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.1
        self.max_position = 2 * np.pi / 3 + self.min_position
        self.max_speed = 0.07
        self.goal_position = np.pi / 6
        self.power = 0.0015

        self.low_state = np.array([-np.inf, -self.max_speed])
        self.high_state = np.array([np.inf, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.seed()
        self.state = np.array([-np.pi / 6, 0])
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        position = self.state[0]
        velocity = self.state[1]
        force = np.clip(u, -1, 1)[0]

        multiplier = 4

        velocity += multiplier * (force * self.power - 0.0025 * math.cos(3 * position))
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += multiplier * velocity
        done = False

        reward = 0
        if done:
            reward = 100.0

        self.state = np.array([position, velocity])
        return self.get_obs(), reward, done, {}

    def get_obs(self):
        if IMG:
            return self.render(mode='rgb_array')
        else:
            return self.state

    def set_state(self, s):
        self.state = np.array(s)

    def reset(self):
        low = np.array([-np.pi / 6 - 0.1, -0.007])
        high = np.array([-np.pi / 6 + 0.1, 0.007])
        self.state = self.np_random.uniform(low=low, high=high)
        return self.get_obs()

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(8)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.set_color(1, 0, 0)
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = (self.state[0] - self.min_position) % (self.max_position - self.min_position) + self.min_position
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
