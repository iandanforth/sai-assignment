# Hackery for testing non-installed module
import sys
sys.path.append("..")
import os
import cStringIO
import numpy as np
import pytest
from envs import GridworldEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def test_init():
    env = GridworldEnv()
    assert env.shape == (6, 9)
    # Check start state
    expected_state = np.ravel_multi_index((5, 3), env.shape)
    assert env.s == expected_state
    # Make sure our wall is properly aligned
    assert all(env._wall[3, 0:-1])
    assert env._wall[3, -1] == False


def test_set_wall_position():
    env = GridworldEnv()

    # Moving into the gap should work
    prev_state = np.ravel_multi_index((4, 8), env.shape)
    expected_state = np.ravel_multi_index((3, 8), env.shape)
    env.s = prev_state
    next_state, _, _, _ = env.step(UP)
    assert next_state == expected_state

    env._set_wall_position(RIGHT)
    assert all(env._wall[3, 1:])
    assert env._wall[3, 0] == False

    # Moving into the old gap should fail
    prev_state = np.ravel_multi_index((4, 8), env.shape)
    env.s = prev_state
    next_state, _, _, _ = env.step(UP)
    assert next_state == prev_state

    # Moving into the new gap should work
    prev_state = np.ravel_multi_index((4, 0), env.shape)
    expected_state = np.ravel_multi_index((3, 0), env.shape)
    env.s = prev_state
    next_state, _, _, _ = env.step(UP)
    assert next_state == expected_state


def test_step():
    env = GridworldEnv()

    # Requires action
    with pytest.raises(TypeError):
        env.step()

    # Moving into a world edge should return to the same state
    prev_state = env.s
    next_state, _, _, _ = env.step(DOWN)
    assert next_state == prev_state

    # Moving UP from the starting position should work and return
    # 0 reward
    next_state, reward, _, _ = env.step(UP)
    expected_state = np.ravel_multi_index((4, 3), env.shape)
    assert next_state == expected_state
    assert reward == 0.0

    # Moving into a wall should return you to previous state
    prev_state = np.ravel_multi_index((4, 0), env.shape)
    env.s = prev_state
    next_state, _, _, _ = env.step(UP)
    assert next_state == prev_state

    # Moving into the goal should return a reward of 1
    prev_state = np.ravel_multi_index((0, 7), env.shape)
    expected_state = np.ravel_multi_index((0, 8), env.shape)
    env.s = prev_state
    next_state, reward, done, _ = env.step(RIGHT)
    assert next_state == expected_state
    assert reward == 1.0
    assert done


def test_wall_move():
    env = GridworldEnv()

    [env.step(DOWN) for _ in range(1000)]

    # Make sure our wall hasn't moved
    assert all(env._wall[3, 0:-1])
    assert env._wall[3, -1] == False

    # Step and check again
    env.step(DOWN)
    assert all(env._wall[3, 1:])
    assert env._wall[3, 0] == False

    # Step some more and double check
    [env.step(DOWN) for _ in range(2000)]
    assert all(env._wall[3, 1:])
    assert env._wall[3, 0] == False


def test_render():
    env = GridworldEnv()
    out = cStringIO.StringIO()
    env.render(out)
    contents = out.getvalue()
    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "artifacts",
        "render_sample"
    )
    with open(sample_path, "r") as fh:
        expected_output = fh.read()
    assert contents == expected_output

    # Actions should change the rendered world
    env.step(UP)
    out = cStringIO.StringIO()
    env.render(out)
    contents = out.getvalue()
    assert contents != expected_output

    # 100% code coverage :)
    env.render()


def test_reset():
    env = GridworldEnv()
    start_state = env.s
    new_state = np.ravel_multi_index((4, 0), env.shape)
    env.s = new_state
    reset_state = env.reset()
    assert reset_state == start_state
    assert env.s == start_state
