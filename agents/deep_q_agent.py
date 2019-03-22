"""
WORK IN PROGRESS

TODO:
  - Consolidate with other agents, factor out shared code
  - Write tests
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


def select_action(env, policy, state, initial_eps, min_eps, eps_decay):
    """
    Select an action randomly from Q with probability eps and
    greedily with probability 1 - eps

    :param env: (Gym Environment)
    :param policy: (nn.Module) Net to predict value from state
    :param state: (torch.Tensor) A network compatible version of an
        environment observation.
    :param initial_eps: (float) Initial probability of random action
    :param min_eps: (float) Minimum value for epsilon.
    :param eps_decay: (float) Value used to decay epsilon
    """
    # Decay epsilon to encourage initial exploration and later
    # exploitation.
    epsilon = initial_eps / eps_decay
    epsilon = np.clip(epsilon, min_eps, initial_eps)
    sample = np.random.uniform()
    if sample <= epsilon:
        nA = env.action_space.n
        action = np.random.randint(0, nA)
    else:
        prediction = policy(state)
        action, _ = av_from_prediction(prediction)
    return action


def state_from_obs(obs):
    obs = np.moveaxis(obs, 2, 0)
    inp = torch.FloatTensor(obs)
    state = inp.unsqueeze(0)
    return state


def av_from_prediction(prediction):
    action = np.argmax(prediction.detach()).item()
    value = prediction[0][action]
    return action, value


class AsyncDeepQAgent(object):
    """
    An Async DQN-Learning Agent

    :param env: (Gym Environment) The environment in which this
        agent will take actions.
    :param shared_policy: (torch.nn.Module) A network to use
        as the policy for this agent and parallel agents.
    :param policy_lock: (mp.RLock) A recurssive lock to prevent
        simultaneous access to the shared_policy
    :param async_update: (int) Number of steps between each update
        to the policy.

    Inherited
    :param initial_eps: (float) Starting value for epsilon.
    :param min_eps: (float) Minimum value for epsilon after decay.
    :param alpha: (float) Learning rate.
    :param gamma: (float) Discount factor.
    """
    def __init__(
        self,
        env,
        shared_policy,
        policy_lock,
        async_update=5,
        initial_eps=1.0,
        min_eps=0.1,
        alpha=0.1,
        gamma=0.95
    ):
        self.env = env
        self.async_update = async_update
        self.initial_eps = initial_eps
        self.min_eps = min_eps
        self.alpha = alpha
        self.gamma = gamma

        self._policy = shared_policy
        self._policy_lock = policy_lock
        self._opt = optim.RMSprop(self._policy.parameters())

        self.reset()

    def _handle_episode_end(self, verbose=False):
        """
        State bookeeping and cleanup. To be called at the end of an episode.
        """
        self.eps_decay += 1
        obs = self.env.reset()
        self.state = state_from_obs(obs)
        prediction = self._policy(self.state)
        _, self.state_value = av_from_prediction(prediction)
        self.steps = 0
        self.episodes += 1

    def _reset_epsilon_decay(self):
        """
        Reset the epsilon decay, starting epsilon back at 1.0
        """
        self.eps_decay = 1.0

    def reset(self):
        """
        Reset the agent and env to initial conditions.
        """
        # Initialize / clear our Q table
        self._reset_epsilon_decay()
        obs = self.env.reset()
        self.state = state_from_obs(obs)
        prediction = self._policy(self.state)
        _, self.state_value = av_from_prediction(prediction)
        self.episodes = 0
        self.steps = 0
        return self.state

    def step(self):
        """
        Takes an action, calculates Q-update and, conditionally,
        updates the shared Q table.

        TODO: Factor out components of this large function.
        """

        # Act
        action = select_action(
            self.env,
            self._policy,
            self.state,
            self.initial_eps,
            self.min_eps,
            self.eps_decay
        )
        next_obs, reward, done, _ = self.env.step(action)
        next_state = state_from_obs(next_obs)

        # Q-Learn
        prediction = self._policy(next_state)
        _, next_state_value = av_from_prediction(prediction)
        expected_value = reward + (0.95 * next_state_value)
        loss = F.smooth_l1_loss(expected_value, self.state_value)

        loss.backward(retain_graph=True)
        for param in self._policy.parameters():
            param.grad.data.clamp_(-1, 1)

        if done or (self.steps % self.async_update == 0):
            with self._policy_lock:
                self._opt.step()
                self._opt.zero_grad()

        # Prepare for next step
        self.state = next_state
        self.state_value = next_state_value
        self.steps += 1

        if done:
            self._handle_episode_end()

        return action, next_state, reward, done
