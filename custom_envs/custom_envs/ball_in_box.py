import gym
from gym import spaces
import numpy as np
from gym.utils import seeding


class BallInBoxEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.vmax = 1
        self.r = 1
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10

        # x, y
        high = np.array([10, 10])

        self.action_space = spaces.Box(low=-self.vmax, high=self.vmax, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.history = []
        self.t = 0
        self.num_collisions = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.t += 1
        u = np.clip(u, -self.vmax, self.vmax)

        old_state = self._get_obs()
        oldx = self.x
        oldy = self.y

        collide = False
        self.x += u[0]
        self.y += u[1]

        clip_x = np.clip(self.x, self.xmin + self.r, self.xmax - self.r)
        clip_y = np.clip(self.y, self.ymin + self.r, self.ymax - self.r)
        if clip_x != self.x or clip_y != self.y:
            collide = True
            self.x = clip_x
            self.y = clip_y

        obs = self._get_obs()
        step_reward = 0
        self.num_collisions += collide
        done = False

        return obs, step_reward, done, {'collisions': self.num_collisions}

    def reset(self):
        self.t = 0
        self.num_collisions = 0

        self.x = self.np_random.uniform(low=self.xmin + self.r, high=self.xmax - self.r)
        self.y = self.np_random.uniform(low=self.ymin + self.r, high=self.ymax - self.r)

        obs = self._get_obs()
        return obs

    def get_obs(self):
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.x, self.y])

    def set_state(self, state):
        self.x = state[0]
        self.y = state[1]

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(64, 64)
            self.viewer.set_bounds(-10, 10, -10, 10)

            agent = rendering.make_circle(radius=self.r)
            agent.set_color(0.3, 0.45, 0.85)
            self.agent_trans = rendering.Transform()
            agent.add_attr(self.agent_trans)
            self.viewer.add_geom(agent)

        self.agent_trans.set_translation(self.x, self.y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
