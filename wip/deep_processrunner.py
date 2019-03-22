"""
Classes and functions adapted from SubprocVecEnv from stable-baselines.

https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/vec_env/subproc_vec_env.py

TODO: Refactor primary process runner to support requirements of
      deep agent variants. Remove this file.
"""

import sys
sys.path.append("..")
import multiprocessing as mp
import numpy as np

from processrunner import CloudpickleWrapper


def _worker(
    remote,
    parent_remote,
    seed,
    init_fn,
    shared_policy,
    policy_lock
):
    """
    Thread / process which will be invoked by multiprocessing.

    :param remote: (mp.Pipe) Duplex channel worker <-> parent
    :param parent_remote: (mp.Pipe) Unused duplex channel
    :param seed: (int) Numpy random seed for this process
    :param init_fn: (CloudPickle string) Pickled instantiation
        function to get a `AsyncQAgent` for this process
    :param shared_Q: (mp.Array) Shared memory Q-table for the
        agent in this process.
    """
    # Close this side of the pipe we don't need.
    parent_remote.close()
    # Set the random seed for this process
    np.random.seed(seed)
    # Unwrap and execute our evn/agent init function
    agent = init_fn.var(shared_policy, policy_lock)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                output = agent.step()
                # Add in a measure of cumulative reward
                data = [agent.episodes, output]
                remote.send(data)
            elif cmd == 'reset':
                observation = agent.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(agent.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
        except EOFError:
            break


class ProcessRunner(object):
    """
    State container and communication hub for async q-learning
    agents.

    :param seeds: ([int]) Random seeds for each environment
    :param init_fns: ([function]) List of functions used to instantiate
        the agent in each child process.
    :param shared_Q: (multiprocessing.Array) Shared memory object to
        represent the Q-table shared across all agents.
    """
    def __init__(self, seeds, init_fns, shared_policy, policy_lock):
        self.waiting = False
        self.closed = False
        if len(seeds) != len(init_fns):
            raise Exception(
                "Number of seeds must match number of init functions"
            )
        self.n_agents = len(init_fns)

        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.n_agents)]
        )
        self.processes = []
        for work_remote, remote, seed, init_fn in zip(
            self.work_remotes,
            self.remotes,
            seeds,
            init_fns
        ):
            args = (
                work_remote,
                remote,
                seed,
                CloudpickleWrapper(init_fn),
                shared_policy,
                policy_lock
            )
            process = mp.Process(target=_worker, args=args)
            process.daemon = True
            process.start()
            self.processes.append(process)
            # Close the side of the pipe we won't use
            work_remote.close()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs

    def step(self):
        self.step_async()
        return self.step_wait()

    def step_async(self):
        for remote in self.remotes:
            remote.send(('step', None))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results
