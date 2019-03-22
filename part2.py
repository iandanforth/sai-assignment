# Standard Library
from __future__ import print_function
from __future__ import division
import argparse
import multiprocessing as mp
from random import randint
# Third Party
import numpy as np
# Local
from envs import GridworldEnv
from agents import AsyncQAgent
from processrunner import ProcessRunner
from plotting import (plot_async_rewards,
                      plot_avg_async_rewards,
                      plot_Q,)


def get_shared_Q():
    world_shape = (6, 9)
    nS = np.prod(world_shape)
    nA = 4
    shared_Q = mp.Array('d', int(nS * nA))
    return shared_Q


def main(args):
    for i in range(args.full_runs):
        # Collect data to demonstrate speedup
        avg_c_rewards = []
        # Train agent using `worker_count` workers
        worker_counts = range(args.min_w, args.max_w + 1, args.w_step)
        for worker_count in worker_counts:
            print("Using {} workers.".format(worker_count))
            shared_Q = get_shared_Q()

            def init_agent(shared_Q):
                # Ensure learning prior to wall shift
                # See README.md - Part 1 - Notes
                env = GridworldEnv(wall_shift=2000)
                # A low learning rate helps to emphasize the influence
                # of parallelism
                alpha = 0.05
                min_eps = 0.4
                agent = AsyncQAgent(
                    env,
                    shared_Q,
                    alpha=alpha,
                    min_eps=min_eps
                )
                return agent

            # Start all our our agents
            seeds = [randint(1, 10000) for _ in range(worker_count)]
            init_fns = [init_agent for _ in range(worker_count)]
            print("Seeds used in this run: {}".format(seeds))
            runner = ProcessRunner(
                seeds,
                init_fns,
                shared_Q
            )

            # Collect average cumulative reward over time
            all_c_rewards = []
            for i in range(args.max_steps):
                c_rewards, output = zip(*runner.step())
                all_c_rewards.append(c_rewards)
                if i % 50000 == 0:
                    print("Step {} ...".format(i))
                avg_reward = np.mean(c_rewards)
                if avg_reward > args.reward_threshold:
                    print("Reward threshold reached at step {}".format(i))
                    break

            runner.close()

            dummy_agent = AsyncQAgent(GridworldEnv(), shared_Q)
            plot_Q(dummy_agent, "final_Q.html")
            plot_async_rewards(all_c_rewards)

            all_c_rewards = np.array(all_c_rewards)
            avg_c_reward = np.mean(all_c_rewards, axis=1)
            avg_c_rewards.append(avg_c_reward)

        plot_avg_async_rewards(avg_c_rewards, worker_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a multiple Q-Learning agents asynchronously \
        in a gridworld."
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
        default=500000,
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

    parser.add_argument(
        "--full-runs",
        default=1,
        type=int,
        help="Run the full training loop N times."
    )

    args = parser.parse_args()
    assert 0 < args.min_w <= args.max_w

    main(args)
