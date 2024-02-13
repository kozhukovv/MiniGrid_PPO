from __future__ import annotations
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces import Discrete
from minigrid.wrappers import ViewSizeWrapper
from typing import Any
from gymnasium.core import Env, Wrapper, ObsType
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils.env import make_env

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # exp_name: str = "mlp_empty_1103"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MiniGrid-FourRooms-v0"
    # env_id: str = "MiniGrid-Empty-8x8-v0"
    # env_id: str = "MiniGrid-Empty-Random-6x6-v0"
    """the id of the environment"""
    total_timesteps: int = 2_000_000 # 500_000 # 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4 # 4
    """the number of parallel game environments"""
    num_steps: int = 512 # 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 # 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10 # 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.006 # 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic_cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=2, padding=0, stride=1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            layer_init(nn.Conv2d(in_channels=32, out_channels=64,
                                 kernel_size=2, padding=0, stride=1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            layer_init(nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=2, padding=0, stride=1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.GRU(64,64)
            )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(in_features=65, out_features=128)),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            layer_init(nn.Linear(in_features=128, out_features=1)),
        )

        self.actor_cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=2, padding=0, stride=1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            layer_init(nn.Conv2d(in_channels=32, out_channels=64,
                                 kernel_size=2, padding=0, stride=1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            layer_init(nn.Conv2d(in_channels=64, out_channels=64,
                                 kernel_size=2, padding=0, stride=1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.GRU(64,64)
            )

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(in_features=65, out_features=128)),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            layer_init(nn.Linear(in_features=128, out_features=3))
        )

    def get_value(self, image, direction):
        return self.critic_mlp(torch.concat((self.critic_cnn(image.permute(0,3,1,2))[0], direction), dim=1))

    def get_action_and_value(self, image, direction, action=None):
        logits = self.actor_mlp(torch.concat((self.actor_cnn(image.permute(0,3,1,2))[0], direction), dim=1))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_mlp(torch.concat((self.critic_cnn(image.permute(0,3,1,2))[0], direction), dim=1))



if __name__ == "__main__":

    action_space = Discrete(3) # Initialization Action Space

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.num_steps) for i in range(args.num_envs)],
    )

    assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # ALGO Logic: Storage setup

    # obs = torch.zeros((args.num_steps, args.num_envs) + tuple(envs.single_observation_space['image'].shape) + tuple([1])).to(device)
    obs_image = torch.zeros((args.num_steps, args.num_envs) + tuple(envs.single_observation_space['image'].shape)).to(device)
    obs_direction = torch.zeros((args.num_steps, args.num_envs) + tuple([1])).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # >>> TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    # <<< TRY NOT TO MODIFY: start the game

    next_image = torch.Tensor(next_obs['image'])
    next_direction = torch.Tensor(next_obs['direction']).view(args.num_envs, 1)

    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):


            global_step += args.num_envs
            obs_image[step] = next_image
            obs_direction[step] = next_direction
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_image, next_direction)

                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu())


            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_image = torch.Tensor(next_obs['image'])
            next_direction = torch.Tensor(next_obs['direction']).view(args.num_envs, 1)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:

                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_image, next_direction).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch

        b_obs_image = obs_image.reshape((-1,) + envs.single_observation_space['image'].shape)
        b_obs_direction = obs_direction.reshape((-1,) + tuple([1]))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space.shape)

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print(b_obs.size())
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_image[mb_inds], b_obs_direction[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # if args.target_kl is not None and approx_kl > args.target_kl:
                # break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        # TRY NOT TO MODIFY: record rewards for plotting purposes

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
