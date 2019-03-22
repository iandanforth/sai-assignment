# Standard Library
from __future__ import print_function
from __future__ import division
# Third Party
import numpy as np


def calc_td_error(
    Q,
    state_index,
    action,
    reward,
    next_state_index,
    gamma
):
    """
    Calculated the Temporal Difference error for this step.

    :param Q: (np.array) A Q table of state-action-values
    :param state_index: (int) The location of the agent in the
        flattened view of the world, prior to `action`
    :param action: (int) The action just taken by the agent.
    :param reward: (int) The reward received for the action taken.
    :param next_state_index: (int) The location of the agent in
        the flattened view of the world, following `action`
    :param gamma: (float) Discount factor for expected reward
    """
    q_val = Q[state_index][action]
    expected_q_val = np.max(Q[next_state_index])
    discounted_exp = gamma * expected_q_val
    td_target = reward + discounted_exp
    td_error = td_target - q_val
    return td_error


def update_policy(Q, Q_deltas, alpha):
    """
    Updates the policy (Q-Table) state-action value
    according to the given td_error and alpha.

    :param Q: (np.array) Q-table which will be updated in place.
    :param Q_deltas: (np.array) An array of updates of the same
        shape as Q to apply to the Q-table.
    :param alpha: (float) The learning rate for applying `td_error`
        to the existing value of the state action pair.
    """
    Q += alpha * Q_deltas


def select_action(Q, state_index, initial_eps, min_eps, eps_decay):
    """
    Select an action randomly from Q with probability eps and
    greedily with probability 1 - eps

    :param state_index: (int) The location of the agent in the
        flattened view of the world.
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
        nA = Q.shape[1]
        action = np.random.randint(0, nA)
    else:
        action = np.argmax(Q[state_index])
    return action


class QAgent(object):
    """
    Q-Learning Agent

    :param env: (Gym Environment) The environment in which this
        agent will take actions.
    :param initial_eps: (float) Starting value for epsilon.
    :param min_eps: (float) Minimum value for epsilon after decay.
    :param alpha: (float) Learning rate.
    :param gamma: (float) Discount factor.
    """
    def __init__(
        self,
        env,
        initial_eps=1.0,
        min_eps=0.1,
        alpha=0.1,
        gamma=0.95,
    ):
        self.env = env
        self.state_count = np.prod(self.env.shape)
        self.initial_eps = initial_eps
        self.min_eps = min_eps
        self.alpha = alpha
        self.gamma = gamma
        self.eps_decay = 1.0  # 1.0 corresponds to no decay

        # Set up additional state and attributes
        self.reset()

    def _act(self):
        action = select_action(
            self._Q,
            self.state_index,
            self.initial_eps,
            self.min_eps,
            self.eps_decay
        )
        next_state_index, reward, done, _ = self.env.step(action)

        return action, next_state_index, reward, done

    def _handle_episode_end(self, verbose=False):
        """
        State bookeeping and cleanup. To be called at the end of an episode.
        """
        self.eps_decay += 1
        self.state_index = self.env.reset()
        self.steps = 0
        self.episodes += 1

    def _learning_step(self, done):
        """
        Directly update local policy

        :param done: (bool) Unused here but useful in child classes
        """
        update_policy(self._Q, self._delta_Q, self.alpha)
        self._reset_q_buffer()

    def _reset_epsilon_decay(self):
        """
        Reset the epsilon decay, starting epsilon back at 1.0
        """
        self.eps_decay = 1.0

    def _reset_q_buffer(self):
        self._delta_Q = np.zeros((self.state_count, self.env.action_space.n))

    def reset(self):
        """
        Reset the agent and env to initial conditions.
        """
        # Initialize / clear our Q table
        self._Q = np.zeros((self.state_count, self.env.action_space.n))
        self._reset_q_buffer()
        self._reset_epsilon_decay()
        self.state_index = self.env.reset()
        self.episodes = 0
        self.steps = 0
        return self.state_index

    def step(self):
        """
        Takes an action, calculates Q-update and, conditionally,
        updates the shared Q table.
        """
        action, next_state_index, reward, done = self._act()

        # Calculate error
        td_error = calc_td_error(
            self._Q,
            self.state_index,
            action,
            reward,
            next_state_index,
            self.gamma
        )

        # Accumulate that error
        self._delta_Q[self.state_index][action] += td_error

        # Apply error as needed
        self._learning_step(done)

        # Prepare for next step
        self.state_index = next_state_index
        self.steps += 1

        if done:
            self._handle_episode_end()

        return action, next_state_index, reward, done


class AsyncQAgent(QAgent):
    """
    An Async Q-Learning Agent

    :param env: (Gym Environment) The environment in which this
        agent will take actions.
    :param shared_Q: (multiprocessing.Array) A shared memory object to use
        as the Q-table for this agent and parallel agents.
    :param async_update: (int) Number of steps between each update
        to the policy.

    Inherited
    :param initial_eps: (float) Starting value for epsilon.
    :param min_eps: (float) Minimum value for epsilon after decay.
    :param alpha: (float) Learning rate.
    :param gamma: (float) Discount factor.
    """
    def __init__(self, env, shared_Q, async_update=5, **kwargs):
        super(AsyncQAgent, self).__init__(env, **kwargs)

        # Create a 2d np array which shares the memory of the provided Q
        self._shared_Q = shared_Q
        q_array = np.frombuffer(self._shared_Q.get_obj())
        view = q_array.view()
        view.shape = (self.state_count, self.env.action_space.n)
        self._Q = view

        self.async_update = async_update

    def _learning_step(self, done):
        """
        Locks the shared policy and updates it in place.
        """
        if done or (self.steps % self.async_update == 0):
            with self._shared_Q.get_lock():
                update_policy(self._Q, self._delta_Q, self.alpha)
            self._reset_q_buffer()
