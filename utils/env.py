from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.core import Env, Wrapper, ObsType
from typing import Any

# from cnn import Args
# from mlp import Args


class RewardBonus(Wrapper):

    def __init__(self, env: Env, gamma: float):
        super().__init__(env)

        self.gamma = gamma
        self.distances = []
        self.agent_coordinates = []
        self.step_count = 0

    def L2(self, agent, goal):
            return ((agent[0] - goal[0])**2 + (agent[1] - goal[1])**2)**(.5) # L2
            # return (abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])) # L1



    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        self.goal_position = tuple([grid.cur_pos for grid in self.env.grid.grid if grid is not None and grid.type == "goal"][0])

        dist = self.L2(self.agent_pos, self.goal_position)
        self.distances.append(dist)
        self.agent_coordinates.append(self.agent_pos)

        # print(self.agent_pos)
        # print(self.goal_position)
        # print(dist)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        # print(f"Step: {self.step_count}")
        dist = self.L2(self.agent_pos, self.goal_position)
        dist_diff = self.distances[-1] - dist

        add_reward = 0

        if dist_diff > 0:
            add_reward += 0.01

        else:
            add_reward += 0


        if action == 2:
            if self.agent_coordinates[-1] != self.agent_pos:
                add_reward += 0.005
            else:
                add_reward -= 0.005

        self.distances.append(dist)
        self.agent_coordinates.append(self.agent_pos)

        add_reward = add_reward * (self.gamma)**(self.step_count-1)
        reward += add_reward


        if np.logical_or(terminated, truncated):
            self.distances.clear()
            self.agent_coordinates.clear()
            self.step_count = 0



        return obs, reward, terminated, truncated, info

def make_env(env_id, idx, capture_video, run_name, gamma, max_steps):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_steps=max_steps, max_episode_steps=max_steps)
            # env = ViewSizeWrapper(env, agent_view_size=3)
            print(env.metadata["render_fps"])
            env = gym.wrappers.RecordVideo(env=env, video_folder=f"videos/{run_name}", episode_trigger=lambda x: x % 20 == 0.0 ) #, episode_trigger=lambda x: x % 50 == 0.0 )
        else:
            env = gym.make(env_id)
        env = RewardBonus(env, gamma)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
