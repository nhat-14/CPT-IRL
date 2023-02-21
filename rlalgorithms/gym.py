"""
Setup an environment, agent and controller for an olfactory search simulation
"""

import os
import json
import h5py
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple, defaultdict

from src.utils import colorize
from src.utils.geometry import Point
from src.agents import silkmoth
from src.controllers import silkmoth_irl, programmed_behavior
from src.envs import smoke_video, wind_tunnel

EpisodeStats = namedtuple("Stats", ["lengths", "rewards"])

class Gym(object):
    def __init__(self,
                input_dir,
                config_file,
                verbosity,
                tlim,
                env='smokevid',
                agt='silkmoth',
                controller='IRL'):

        self.input_dir = input_dir
        self.verbosity = verbosity
        with open(config_file) as f:
            self.cfg = json.load(f)

        self.fps = self.cfg["FPS"]
        self._env = env
        self._agt = agt
        self._ctrl = controller
        self.tlim = tlim
        self.performance = {'success_rate': [], 'search_time': []}
        self.Nsteps = int(np.round(self.tlim * self.fps))
        self.wind_angle = self.cfg["wind_angle"]
        self.bin_edges = None
        self.policy = None
        self.irl_num_states = None

        if self._ctrl[0] == 'IRL':
            self.bin_edges = self.get_bin_edges()
            self.rewards = self.get_rewards()
            self.policy = self.get_policy()
            self.irl_num_states = self.cfg["irl"]["num_states"]

    def get_bin_edges(self):
        f = os.path.join(os.getcwd(), self.input_dir, 'bin_edges.csv')
        bin_edges = pd.read_csv(f)
        return bin_edges

    def get_policy(self):
        f = os.path.join(os.getcwd(), self.input_dir, 'policy.h5')
        with h5py.File(f, 'r') as hf:
            policy = hf['policy'][:]
        return policy

    def get_rewards(self):
        f = os.path.join(os.getcwd(), self.input_dir, 'Reward.csv')
        rewards = pd.read_csv(f, usecols=[1, 2, 3, 4]).values
        return rewards

    def set_agent(self):
        p0 = self.cfg["init_pose"]
        if self.cfg['random_start']:
            eps = self.cfg['init_pose_eps']
            p0 = [(p0[i] + np.random.uniform(-eps[i], eps[i]))
                  for i in range(len(eps))]
            # e = np.random.normal(0., 10.)
        init_pose = tuple(p0)
        return {
            'silkmoth': silkmoth.SilkMoth(*init_pose),
            'robot': None
        }.get(self._agt, None)

    def set_env(self, plume=None):
        cfg = self.cfg['field']

        if self._env == 'smokevid':
            env = smoke_video.SmokeVideo(self.Nsteps, cfg['width'],
                                         cfg['height'], tuple(cfg['srcpos']),
                                         tuple(cfg['xlim']),
                                         tuple(cfg['ylim']), plume)

        elif self._env == 'windtunnel':
            env = wind_tunnel.WindTunnel(plume, self.Nsteps, cfg['width'],
                                         cfg['height'], tuple(cfg['srcpos']),
                                         tuple(cfg['xspace']),
                                         tuple(cfg['yspace']))

        return env

    def set_controller(self):
        return {
            'KPB':
            programmed_behavior.KPB(self.fps),
            'IRL':
            silkmoth_irl.SilkmothIRL(self.fps,
                                     self.bin_edges,
                                     self.irl_num_states,
                                     self.policy,
                                     rewards=self.rewards)
        }.get(self._ctrl[0], None)

    def get_action_space(self):
        return {
            'IRL':
            silkmoth_irl.SilkmothIRL(self.fps,
                                     self.bin_edges,
                                     self.irl_num_states,
                                     self.policy,
                                     rewards=self.rewards).action_space
        }.get(self._ctrl[0], None)

    def set_animation(self, env):

        fig, ax = plt.subplots()

        X, Y = np.mgrid[0:env.height, 0:env.width]
        img = ax.imshow(X,
                        origin='lower',
                        extent=env.extent,
                        vmin=env.vmin,
                        vmax=env.vmax,
                        cmap=env.colormap)

        ax.add_artist(
            plt.Circle((env.mmsrc[0], env.mmsrc[1]),
                       self.cfg["env"]["goalr"],
                       color='yellow',
                       fill=False,
                       linestyle='--'))
        line, = ax.plot([], lw=1, c='w')
        text = ax.text(20, 320, "", color="#00ffff", family='monospace')
        fig.canvas.draw()
        fig.tight_layout()

        plt.show(block=False)

        return fig, img, line, text

    def off_grid(self, p: Point, m):

        xs = self.cfg['field']['xlim']
        ys = self.cfg['field']['ylim']

        og = not ((min(xs) + m < p.x < max(xs) - m) and
                (min(ys) + m < p.y < max(ys) - m))

        return og

    def reached_goal(self, p: Point):

        radius = self.cfg['field']['goalr']

        return (np.sqrt(p.x**2 + p.y**2) < radius)

    def plot_trajectory(self, traj, f):

        xsrc, ysrc = tuple(self.cfg['field']['srcpos'])
        radius = self.cfg['field']['goalr']

        fig, ax = plt.subplots()
        ax.add_artist(Circle((xsrc, ysrc), radius, color='r', fill=False, linestyle='--', linewidth=0.5, zorder=1))
        ax.scatter(traj.iloc[0,0], traj.iloc[0,1], marker='.')
        ax.plot(traj.iloc[:,0], traj.iloc[:,1], zorder=3)
        ax.scatter(xsrc, ysrc, marker='*', c='k')

        ax.set_xlim(self.cfg['field']['xlim'])
        ax.set_ylim(self.cfg['field']['ylim'])
        ax.set_aspect('equal')
        plt.savefig(f, dpi=300)

    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.
        
        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
    
        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    def SARSA(self, episodes, plumes, discount=.1, learning_rate=.01, epsilon=.2, save_log=False):

        episode_stats = EpisodeStats(np.zeros(episodes), np.zeros(episodes))
        dt = 1 / self.fps
        action_space = self.get_action_space()
        nactions = len(action_space)

        Q = defaultdict(lambda: np.zeros(nactions))
        policy = self.make_epsilon_greedy_policy(Q, epsilon, nactions)

        for i in tqdm(range(episodes)):

            plume = np.random.choice(plumes)
            env = self.set_env()
            agent = self.set_agent()
            controller = self.set_controller()
            t = 0

            x_ls = []
            y_ls = []
            log = []
            last_hit = 0
            found_source = 0
            # hit_noise = self.cfg['hit_noise']
            hit_eps = np.random.binomial(1, self.cfg['hit_probability'])
            done = False
            reward = 0.
            prev_state = 0
            next_state = 0

            # if i == (episodes - 1):
            # fig, img, line, text = self.set_animation(env)

            prev_state = controller.state
            action_probs = policy(prev_state)
            action = np.random.choice(np.arange(len(action_probs)),
                                      p=action_probs)

            with h5py.File(plume, 'r') as h5:
                plume = h5['frames']

                # while not done:
                for t in itertools.count():

                    # if i == (episodes - 1):
                    #     img.set_data(env.plume[t])
                    #     line.set_data(x_ls, y_ls)
                    #     text.set_text("T: {:.2f}, episode: {}".format(t * dt, i))
                    #     fig.canvas.draw()
                    #     fig.canvas.flush_events()



                    # prev_state = controller.state

                    # if np.cos(agent.theta - np.pi) > self.wind_angle:
                    if np.cos(np.pi - agent.theta + self.wind_angle) > 0:
                        right_hit = env.hit_at(t,
                                               env.mm2px(agent.right_antenna),
                                               plume)
                        left_hit = env.hit_at(t, env.mm2px(agent.left_antenna),
                                              plume)

                        # if hit_noise:
                        #     right_hit *= hit_eps
                        #     left_hit *= hit_eps

                    else:
                        right_hit = 0
                        left_hit = 0

                    antennae_hit = (lambda l, r: (l << 1) | r)(left_hit, right_hit)
                    if antennae_hit > 0: last_hit = antennae_hit

                    # prev_state = controller.state


                    # controller.control(antennae_hit, last_hit)
                    controller.random_action(antennae_hit, action)
                    agent.move(controller.linear_vel, controller.angular_vel, dt)

                    next_state = controller.state
                    reward = controller.reward
                    next_action_probs = policy(prev_state)
                    next_action = np.random.choice(np.arange(
                        len(next_action_probs)),
                                                p=next_action_probs)

                    if self.off_grid(agent.pos, 1.5 * agent.antenna_length):
                        found_source = np.nan
                        reward -= 1e3
                        done = True
                        # break

                    if t >= (self.Nsteps - 1):
                        found_source = np.nan
                        # reward = -10.
                        done = True

                    if self.reached_goal(agent.pos):
                        found_source = 1
                        # reward += 1e4
                        done = True
                        # break

                    episode_stats.rewards[i] += reward
                    episode_stats.lengths[i] = t

                    # best_next_action = np.argmax(Q[next_state])
                    td_target = reward + discount * Q[next_state][next_action]
                    td_delta = td_target - Q[prev_state][action]
                    Q[prev_state][action] += learning_rate * td_delta



                    controller._tblank += dt
                    # t += 1
                    x_ls.append(agent.pos.x)
                    y_ls.append(agent.pos.y)

                    if save_log:
                        log.append([t / self.fps, antennae_hit] + agent.log_step +
                                controller.log_step)

                    if (self.verbosity is not None) and (self.verbosity <= 10):
                        print('t: {:.4f}, h:{}, {}, {}'.format(
                            t / self.fps, antennae_hit, str(agent),
                            str(controller)))



                    if done: break

                    prev_state = next_state
                    action = next_action

            self.performance['success_rate'].append(found_source)
            self.performance['search_time'].append(t/self.fps)

        traj = pd.DataFrame({'x': x_ls, 'y': y_ls})
        self.plot_trajectory(traj, 'SARSA-traj')
        performance = pd.DataFrame(self.performance)

        return Q, episode_stats, performance



    def Qlearning(self, episodes, plumes, discount=.9, learning_rate=.01, epsilon=.2, save_log=False):

        episode_stats = EpisodeStats(np.zeros(episodes), np.zeros(episodes))
        dt = 1 / self.fps
        action_space = self.get_action_space()
        # nactions = len(controller.action_space())
        nactions = len(action_space)
        nstates = self.rewards.size
        dp_policy = self.policy.T.reshape(nstates, nactions)
        # print(f'Number of actions: {nactions}')

        # Q = defaultdict(lambda: np.zeros(nactions))
        Q = defaultdict(lambda: dp_policy[0])
        policy = self.make_epsilon_greedy_policy(Q, epsilon, nactions)

        for i in tqdm(range(episodes)):

            plume = np.random.choice(plumes)
            env = self.set_env()
            agent = self.set_agent()
            controller = self.set_controller()

            # DP means dynamic programming

            # print(dp_policy)

            t = 0

            x_ls = []
            y_ls = []
            log = []
            last_hit = 0
            found_source = 0
            # hit_noise = self.cfg['hit_noise']
            # hit_eps = np.random.binomial(1, self.cfg['hit_probability'])
            done = False
            reward = 0.
            prev_state = 0
            next_state = 0
            # print(dp_policy[prev_state])
            # if i == (episodes - 1):
            # fig, img, line, text = self.set_animation(env)

            # for t in range(self.Nsteps - 1):
            prev_state = controller.state
            with h5py.File(plume, 'r') as h5:
                plume = h5['frames']


                # while not done:
                for t in itertools.count():

                    # if i == (episodes - 1):
                    #     img.set_data(env.plume[t])
                    #     line.set_data(x_ls, y_ls)
                    #     text.set_text("T: {:.2f}, episode: {}".format(t * dt, i))
                    #     fig.canvas.draw()
                    #     fig.canvas.flush_events()

                    # prev_state = controller.state

                    # if np.cos(agent.theta - np.pi) > self.wind_angle:
                    if np.cos(np.pi - agent.theta + self.wind_angle) > 0:
                        right_hit = env.hit_at(t,
                                               env.mm2px(agent.right_antenna),
                                               plume)
                        left_hit = env.hit_at(t, env.mm2px(agent.left_antenna),
                                              plume)

                        # if hit_noise:
                        #     right_hit *= hit_eps
                        #     left_hit *= hit_eps

                    else:
                        right_hit = 0
                        left_hit = 0

                    antennae_hit = (lambda l, r: (l << 1) | r)(left_hit, right_hit)
                    if antennae_hit > 0: last_hit = antennae_hit

                    # prev_state = controller.state
                    action_probs = policy(prev_state)
                    # action_probs *= dp_policy[prev_state]
                    # action_probs /= action_probs.sum()
                    action = np.random.choice(np.arange(len(action_probs)),
                                            p=action_probs)

                    # controller.control(antennae_hit, last_hit)
                    controller.random_action(antennae_hit, action)
                    agent.move(controller.linear_vel, controller.angular_vel, dt)
                    next_state = controller.state
                    reward = controller.reward

                    if self.off_grid(agent.pos, 1.5 * agent.antenna_length):
                        found_source = np.nan
                        # reward -= 1e3
                        done = True
                        # break

                    if t >= (self.Nsteps - 1):
                        found_source = np.nan
                        # reward = -10.
                        done = True

                    if self.reached_goal(agent.pos):
                        found_source = 1
                        # reward += 1e3
                        done = True
                        # break

                    episode_stats.rewards[i] += reward
                    episode_stats.lengths[i] = t

                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + discount * Q[next_state][best_next_action]
                    td_delta = td_target - Q[prev_state][action]
                    Q[prev_state][action] += learning_rate * td_delta

                    controller._tblank += dt
                    # t += 1
                    x_ls.append(agent.pos.x)
                    y_ls.append(agent.pos.y)

                    if save_log:
                        log.append([t/self.fps, antennae_hit] + agent.log_step + controller.log_step)

                    if (self.verbosity is not None) and (self.verbosity <= 10):
                        print('t: {:.4f}, h:{}, {}, {}'.format(t / self.fps,
                                                            antennae_hit,
                                                            str(agent),
                                                            str(controller)))



                    if done: break
                    prev_state = next_state



            self.performance['success_rate'].append(found_source)
            self.performance['search_time'].append(t/self.fps)
            # print(json.dumps(Q, indent=4))
            # print(Q)

            # if save_log:
            #     log_cols = 'Time, hit, {}, {}'.format(agent.log_header, controller.log_header).split(',')
            #     log = pd.DataFrame(columns=log_cols, data=log)

            #     return traj, Q, log

            # else:
        traj = pd.DataFrame({'x': x_ls, 'y': y_ls})
        self.plot_trajectory(traj, 'QL-traj')
        performance = pd.DataFrame(self.performance)

        n_states = len(Q)
        print('No. of visited states: {}'.format(n_states))
        dfQ = pd.DataFrame(Q).T
        dfQ.sort_index(inplace=True)
        # print(dfQ)
        Q = dfQ.values

        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        opt_pol = np.exp(Q) / np.exp(Q).sum(axis=1).reshape((n_states, 1))
        opt_pol = pd.DataFrame(opt_pol)

        return opt_pol, episode_stats, performance