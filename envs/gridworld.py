# Standard Library
from __future__ import print_function
from __future__ import division
import sys
# Third Party
import numpy as np
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    This gridworld is a modified version of CliffWalkingEnv from the
    standard suite of OpenAI Gym envs.

    As described in the Sanctuary AI prompt:

    The world is a 6x9 matrix, with (using Numpy matrix indexing):
        [5, 3] as the start position on the bottom row
        [0, 8] as the goal at the top right

    There are two wall positions. In the first 1000 steps the wall occupies:
        [3, 0..7]
    After the first 1000 steps it shifts to the left one square to occupy:
        [3, 1..8]

    The action space consists of 4 options. Moving UP, RIGHT, DOWN, or LEFT

    Reward is zero on all transitions, except those into the goal state
    which return +1. An episode terminates when the agent reaches the goal.
    Any action which would result in the agent being outside the maze or
    in collision with the wall instead results in the agent being placed
    back into the square where it initiated the action.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, wall_shift=1000):
        self.shape = (6, 9)
        self.start = (5, 3)
        self.goal = (0, 8)
        self.start_state_index = np.ravel_multi_index(self.start, self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        # Wall Location
        # This will shift when _elapsed_steps == self.wall_shift
        # see step() below
        self.wall_shift = wall_shift
        self._elapsed_steps = 0
        self._set_wall_position(LEFT)

        # Calculate initial state distribution
        # We always start in state (5, 3)
        isd = np.zeros(self.nS)
        isd[self.start_state_index] = 1.0

        super(GridworldEnv, self).__init__(self.nS, self.nA, self.P, isd)

    def _calculate_transition_probabilities(self):
        P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])
        return P

    def _set_wall_position(self, side):
        """
        Defines a wall that extends the full width of the world
        save one square (the gap).
        """
        self._wall = np.zeros(self.shape, dtype=np.bool)

        if side == LEFT:
            self._wall[3, 0:-1] = True
        elif side == RIGHT:
            self._wall[3, 1:] = True

        self.P = self._calculate_transition_probabilities()

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        # If the agent would be in the wall, return to current position
        if self._wall[tuple(new_position)]:
            new_state = np.ravel_multi_index(tuple(current), self.shape)

        # Finish episode and return positive reward if goal is reached
        is_done = tuple(new_position) == self.goal

        return [(1.0, new_state, int(is_done), is_done)]

    def render(self, outfile=None):
        if outfile is None:
            outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == self.start:
                output = " S "
            elif position == self.goal:
                output = " G "
            elif self._wall[position]:
                output = " = "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)

        outfile.write('\n')
        outfile.flush()

    def step(self, action):
        # Update the maze at step 1000
        if self._elapsed_steps == self.wall_shift:
            # print("Updating wall ...")
            self._set_wall_position(RIGHT)
        self._elapsed_steps += 1
        return super(GridworldEnv, self).step(action)
