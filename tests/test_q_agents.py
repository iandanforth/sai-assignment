# Hackery for testing non-installed module
import sys
sys.path.append("..")
import pytest
import multiprocessing as mp
import numpy as np
from envs import GridworldEnv
from processrunner import ProcessRunner
import agents as a

np.random.seed(0)  # Tests should be deterministic


def test_calc_td_error():
    # 1x3 world:
    # [[ o, S, G]]
    # With 2 actions:
    RIGHT = 1
    world_shape = (1, 3)
    nS = np.prod(world_shape)
    nA = 2
    state_index = np.ravel_multi_index((0, 1), world_shape)
    action = RIGHT
    reward = 1.0
    next_state_index = np.ravel_multi_index((0, 2), world_shape)

    # Error should be equal to reward when Q is all zeros
    Q = np.zeros((nS, nA))
    gamma = 1.0
    td_error = a.calc_td_error(
        Q,
        state_index,
        action,
        reward,
        next_state_index,
        gamma
    )
    assert td_error == 1.0

    # This should be irrespective of gamma
    Q = np.zeros((nS, nA))
    gamma = 0.5
    td_error = a.calc_td_error(
        Q,
        state_index,
        action,
        reward,
        next_state_index,
        gamma
    )
    assert td_error == 1.0

    # If our state-action-value error is accurate
    # there should be no error
    Q = np.zeros((nS, nA))
    Q[state_index][action] = 1.0
    gamma = 1.0
    td_error = a.calc_td_error(
        Q,
        state_index,
        action,
        reward,
        next_state_index,
        gamma
    )
    assert td_error == 0.0

    # Errors should be proportional to gamma
    Q = np.zeros((nS, nA))
    Q[state_index][action] = 1.0
    Q[next_state_index][0] = 1.0
    gamma = 0.3
    td_error = a.calc_td_error(
        Q,
        state_index,
        action,
        reward,
        next_state_index,
        gamma
    )
    assert td_error == pytest.approx(gamma)

    # Errors should assume greedy action selection
    Q = np.zeros((nS, nA))
    Q[state_index][action] = 1.0
    Q[next_state_index][:] = [0.4, 0.8]
    gamma = 0.5
    td_error = a.calc_td_error(
        Q,
        state_index,
        action,
        reward,
        next_state_index,
        gamma
    )
    assert td_error == pytest.approx(0.4)


def test_update_policy():
    # 1x3 world:
    # [[ o, S, G]]
    world_shape = (1, 3)
    nS = np.prod(world_shape)
    nA = 2

    # Array add works
    Q = np.zeros((nS, nA))
    delta = 0.1
    Q_deltas = np.full((nS, nA), delta)
    alpha = 1.0
    a.update_policy(Q, Q_deltas, alpha)
    assert (Q == delta).all()

    # Alpha works
    Q = np.zeros((nS, nA))
    delta = 1.0
    Q_deltas = np.full((nS, nA), delta)
    alpha = 0.5
    a.update_policy(Q, Q_deltas, alpha)
    assert (Q == alpha).all()


def test_select_action():
    # 1x3 world:
    # [[ o, S, G]]
    # With 2 actions:
    LEFT = 0
    RIGHT = 1
    world_shape = (1, 3)
    nS = np.prod(world_shape)
    nA = 2
    state_index = np.ravel_multi_index((0, 1), world_shape)

    # With epsilon 1.0 and no decay we should pick randomly
    Q = np.zeros((nS, nA))
    action = a.select_action(Q, state_index, 1.0, 0.0, 1.0)
    assert action == RIGHT

    # With epsilon 0.0 we should always act greedily
    Q = np.zeros((nS, nA))
    Q[state_index][:] = [1.0, 0.0]
    for i in range(100):
        action = a.select_action(Q, state_index, 0.0, 0.0, 1.0)
        assert action == LEFT

    # Golden master
    Q = np.zeros((nS, nA))
    Q[state_index][:] = [0.2, 0.3]
    expected_actions = [1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    actions = []
    for i in range(10):
        action = a.select_action(Q, state_index, 0.5, 0.0, 1.0)
        actions.append(action)

    assert actions == expected_actions


def test_init_qagent():
    env = GridworldEnv()
    agent = a.QAgent(env)
    assert agent.state_count == np.prod(env.shape)
    # Q and update buffers should start as 0s
    assert not agent._Q.any()
    assert not agent._delta_Q.any()


def test_step():
    RIGHT = 1
    env = GridworldEnv()
    agent = a.QAgent(env)
    action, next_state_index, reward, done = agent.step()
    assert reward == 0.0
    assert not done

    # A deterministic agent one step from the goal and
    # a clear policy choice should reach terminal state.
    env = GridworldEnv()
    agent = a.QAgent(env, initial_eps=0.0, min_eps=0.0)
    # Put the agent + env one step from the goal
    state = 7
    agent.state_index = env.s = state
    # Make RIGHT very appealing
    agent._Q[state][:] = [0.0, 1.0, 0.0, 0.0]
    action, next_state_index, reward, done = agent.step()
    assert action == RIGHT
    assert next_state_index == 8
    assert reward == 1.0
    assert done


##############################################################
# AsyncQAgent
def test_init_async_qagent():
    wall_shift = 2000
    env = GridworldEnv(wall_shift=wall_shift)
    nS = np.prod(env.shape)
    nA = env.action_space.n
    shared_Q = mp.Array('d', nS * nA)
    agent = a.AsyncQAgent(env, shared_Q)

    # Using seed 0 we should complete exactly
    # 3 episodes in 2000 steps
    for i in range(wall_shift):
        agent.step()

    assert agent.episodes == 3

    # Multiple agents should run in parallel
    shared_Q = mp.Array('d', nS * nA)

    def init_agent(shared_Q):
        env = GridworldEnv(wall_shift=wall_shift)
        agent = a.AsyncQAgent(env, shared_Q)
        return agent

    seeds = [0, 1, 2]
    init_fns = [init_agent for _ in range(len(seeds))]
    runner = ProcessRunner(
        seeds,
        init_fns,
        shared_Q
    )

    for i in range(wall_shift):
        c_rewards, output = zip(*runner.step())
        avg_reward = np.mean(c_rewards)
    assert avg_reward == pytest.approx(124.3333)

    # Resetting agents should put them all back to the start
    states = runner.reset()
    expected_state = np.ravel_multi_index((5, 3), env.shape)
    assert all(states == expected_state)

    runner.close()
