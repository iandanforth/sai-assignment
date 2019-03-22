"""
WORK IN PROGRESS

This script will run `worker_count` instances of an AsyncDeepQAgent
to learn in the SpaceInvaders Atari Gym environment.

TODO:
  - Don't consume all the CPU in the world.
"""

# Standard Library
from __future__ import print_function
from __future__ import division
import argparse
import multiprocessing as mp
from random import randint
# Third Party
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines.common.atari_wrappers import WarpFrame, FrameStack
# Local
from wip.deep_q_agent import AsyncDeepQAgent
from wip.deep_processrunner import ProcessRunner


class DQN(nn.Module):
    """
    A convolutional neural network in the style of Mnih et al.

    Simplified from:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(9 * 9 * 32, 256)
        self.head = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten filters
        x = self.fc(x)
        x = F.relu(x)
        return self.head(x)


def main(args):
    # Train agent using `worker_count` workers
    worker_counts = range(args.min_w, args.max_w + 1, args.w_step)
    for worker_count in worker_counts:
        print("Using {} workers.".format(worker_count))
        shared_policy = DQN()
        policy_lock = mp.RLock()

        def init_agent(shared_policy, policy_lock):
            # Use the built-in frameskip
            # Don't let ALE repeat actions
            env = gym.make('SpaceInvadersDeterministic-v4')
            env = WarpFrame(env)
            env = FrameStack(env, 4)
            agent = AsyncDeepQAgent(
                env,
                shared_policy,
                policy_lock,
            )
            return agent

        # Start all our our agents
        seeds = [randint(1, 10000) for _ in range(worker_count)]
        init_fns = [init_agent for _ in range(worker_count)]
        print("Seeds used in this run: {}".format(seeds))
        runner = ProcessRunner(
            seeds,
            init_fns,
            shared_policy,
            policy_lock
        )

        # Collect average cumulative reward over time
        all_c_rewards = []
        for i in range(args.max_steps):
            c_rewards, output = zip(*runner.step())
            print(i)  # TODO: Remove
            all_c_rewards.append(c_rewards)
            if i % 50000 == 0:
                print("Step {} ...".format(i))
            avg_reward = np.mean(c_rewards)
            if avg_reward > args.reward_threshold:
                print("Reward threshold reached at step {}".format(i))
                break

        runner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a multiple Q-Learning agents asynchronously \
        in to play Space Invaders."
    )

    parser.add_argument(
        "--reward-threshold",
        default=200,
        type=int,
        help="Stop training if the average cumulative reward exceeds \
        this value"
    )

    parser.add_argument(
        "--max-steps",
        default=5000000,
        type=int,
        help="The maximum number of steps each agent will perform."
    )

    parser.add_argument(
        "--min-w",
        default=2,
        type=int,
        help="The minimum number of workers to search over. Valid if \
        0 < args.min_workers <= args.max_workers"
    )

    parser.add_argument(
        "--max-w",
        default=8,
        type=int,
        help="The maximum number of workers to search over."
    )

    parser.add_argument(
        "--w-step",
        default=2,
        type=int,
        help="How worker count will be incremented between min and max."
    )

    args = parser.parse_args()
    assert 0 < args.min_w <= args.max_w

    main(args)
