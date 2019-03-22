# Standard Library
from __future__ import print_function
from __future__ import division
from random import randint
import argparse
# Third Party
import numpy as np
# Local
from envs import GridworldEnv
from agents import QAgent
from plotting import plot_Q, plot_rewards


def train(agent, max_steps, r_thresh, plot_early=False):
    """
    Runs an agent for `max_steps` returning a trace
    of its cumulative reward over time
    """
    rewards = []
    for i in range(max_steps):
        action, _, _, _ = agent.step()
        rewards.append(agent.episodes)
        if plot_early and i == agent.env.wall_shift:
            plot_Q(agent, filename="early_policy.html")
        if i % 10000 == 0:
            print("Step {} ...".format(i))
        if agent.episodes > r_thresh:
            print("Reward threshold reached after {} steps".format(i))
            break

    return rewards


def main(args):
    for i in range(args.full_runs):
        print("Training run {}".format(i))

        seed_dict = dict(
            bad=9696,
            good=7843,
            random=randint(1, 10000),
        )

        seed = seed_dict[args.seed_type]

        print("Random seed: {}".format(seed))

        # Prepare a baseline
        print("Random Agent ...")
        np.random.seed(seed)
        env = GridworldEnv()
        random_agent = QAgent(env, min_eps=1.0)
        random_agent_rewards = train(
            random_agent,
            args.max_steps,
            args.reward_threshold
        )

        # Train the agent
        print("Learning Agent ...")
        np.random.seed(seed)  # Reset prng state
        env = GridworldEnv()
        alpha = 0.1  # World is fully deterministic
        learning_agent = QAgent(env, alpha=alpha)
        learning_agent_rewards = train(
            learning_agent,
            args.max_steps,
            args.reward_threshold,
            plot_early=True
        )

        plot_rewards(
            seed,
            random_agent_rewards[:env.wall_shift],
            learning_agent_rewards[:env.wall_shift],
            filename="part_1_initial_rewards.html"
        )
        plot_Q(learning_agent, filename="final_policy.html")
        plot_rewards(
            seed,
            random_agent_rewards,
            learning_agent_rewards,
            filename="part_1_full_rewards.html"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a single Q-Learning agent in a gridworld."
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
        help="The total number of steps the agent will perform."
    )

    seed_types = ["good", "bad", "random"]
    parser.add_argument(
        "--seed-type",
        default=seed_types[0],
        choices=seed_types,
        help="What type of seed to use. 'good' demonstrates learning. \
        'bad' demonstrates a failure case."
    )

    parser.add_argument(
        "--full-runs",
        default=1,
        type=int,
        help="Run the full training loop N times. Note: Only useful \
        if seed_type is 'random'"
    )

    args = parser.parse_args()

    main(args)
