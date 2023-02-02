"""
Setup an environment, agent and controller for an olfactory search simulation
"""

import os
import logging
import h5py
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
from collections import deque, namedtuple
import config
from utils.geometry import Point
from agents import silkmoth, infotaxis_agt, hybrid
from controllers import silkmoth_irl, programmed_behavior, infotaxis, itx2pb
from envs import smoke_video, wind_tunnel, filament


class Simulator(object):

    def __init__(self,
                input_dir,
                main_config,
                tlim,
                env='smokevid',
                agt='silkmoth',
                controller='KPB'):

        self.input_dir = input_dir
        self.main_conf = main_config
        self.env_conf = config.FieldConf(main_config)

        self.fps = main_config.FPS
        self._env = env
        self._agt = agt
        self._ctrl = main_config.controller_type
        self.tlim = tlim
        self.performance = {'success_rate': [], 'search_time': []}
        self.success_rate = 0
        self.Nsteps = int(np.round(self.tlim * self.fps))
        self.wind_angle = main_config.wind_angle
        self.bin_edges = None
        self.policy = None
        self.irl_num_states = None

        if self._ctrl[0] == 'IRL':
            self.bin_edges = self.get_bin_edges()
            self.policy = self.get_policy()
            self.irl_num_states = main_config["irl"]["num_states"]

    def get_bin_edges(self):
        f = os.path.join(self.input_dir, 'bin_edges',
                         '{}.csv'.format(self._ctrl[1]))
        bin_edges = pd.read_csv(f)

        return bin_edges

    def get_policy(self):
        f = os.path.join(self.input_dir, 'irl_policies',
                         '{}.h5'.format(self._ctrl[2]))

        with h5py.File(f, 'r') as hf:
            policy = hf['policy'][:]

        return policy

    def set_agent(self):
        p0 = self.main_conf.initial_pose
        # _logger.info('Place agent at a random position')
        if self.main_conf.random_start:
            eps = self.main_conf.init_pose_eps
            if self.main_conf.init_pose_epstype == 'uniform':
                p0 = [(p0[i] + np.random.uniform(-eps[i], eps[i]))
                    for i in range(len(eps))]
            else:
                p0 = [(p0[i] + np.random.randint(-eps[i], eps[i]))
                      for i in range(len(eps))]
            # e = np.random.normal(0., 10.)
        init_pose = tuple(p0)
        return {
            'silkmoth': silkmoth.SilkMoth(*init_pose),
            'infotaxis': infotaxis_agt.ItxAgent(init_pose[0], init_pose[1]),
            'hybrid': hybrid.HybridAgent(init_pose[0], init_pose[1]),
            'robot': None
        }.get(self._agt, None)

    def set_env(self):
        env_conf = config.FieldConf(self.main_conf)
        if self._env == 'smokevid':
            env = smoke_video.SmokeVideo(self.Nsteps, env_conf.width,
                                         env_conf.height,
                                         tuple(env_conf.source_pos),
                                         tuple(env_conf.xlim),
                                         tuple(env_conf.ylim))

        elif self._env == 'filament':
            env = filament.Filament(self.Nsteps,
                                    env_conf.width, env_conf.height,
                                    tuple(env_conf.source_pos),
                                    tuple(env_conf.xlim), tuple(env_conf.ylim))

        elif self._env == 'windtunnel':
            env = wind_tunnel.WindTunnel(plume, self.Nsteps, env_conf.width,
                                         env_conf.height,
                                         tuple(env_conf.source_pos),
                                         tuple(env_conf.xlim),
                                         tuple(env_conf.ylim))

        return env

    def set_controller(self):

        if self.main_conf.controller_type[0] == 'ITX':
            controller_conf = config.InfotaxisConf(self.main_conf)
            return infotaxis.Infotaxis(self.fps, tuple(self.env_conf.xlim),
                                       tuple(self.env_conf.ylim), controller_conf)

        elif self.main_conf.controller_type[0] == 'HYB':
            controller_conf = config.InfotaxisConf(self.main_conf)
            return itx2pb.C2R(self.fps, tuple(self.env_conf.xlim),
                              tuple(self.env_conf.ylim), controller_conf)

        elif self.main_conf.controller_type[0] == 'KPB':
            return programmed_behavior.KPB(self.fps)

        elif self.main_conf.controller_type[0] == 'IRL':
            return silkmoth_irl.SilkmothIRL(self.fps, self.bin_edges,
                                            self.irl_num_states, self.policy)
        else:
            return None

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
                       self.env_conf.goalr,
                       color='yellow',
                       fill=False,
                       linestyle='--'))
        line, = ax.plot([], lw=1, c='w')
        text = ax.text(20, 320, "", color="#00ffff", family='monospace')
        fig.canvas.draw()

        plt.show(block=False)

        return fig, img, line, text

    def off_grid(self, p: Point, m):

        xs = self.env_conf.xlim
        ys = self.env_conf.ylim

        og = not ((min(xs) + m < p.x < max(xs) - m) and
                (min(ys) + m < p.y < max(ys) - m))

        return og

    def reached_goal(self, p: Point):

        radius = self.env_conf.goalr

        return (np.sqrt(p.x**2 + p.y**2) < radius)

    def plot_trajectory(self, traj, f):

        xsrc, ysrc = tuple(self.env_conf.source_pos)
        radius = self.env_conf.goalr

        fig, ax = plt.subplots()
        ax.add_artist(Circle((xsrc, ysrc), radius, color='r', fill=False, linestyle='--', linewidth=0.5, zorder=1))
        ax.scatter(traj.iloc[0,0], traj.iloc[0,1], marker='.')
        ax.plot(traj.iloc[:,0], traj.iloc[:,1], zorder=3)
        ax.scatter(xsrc, ysrc, marker='*', c='k')

        ax.set_xlim(self.env_conf.xlim)
        ax.set_ylim(self.env_conf.ylim)
        ax.set_aspect('equal')
        plt.savefig(f, dpi=300)


    def run(self, plume, hit_prob=1.0, draw_animation=False, save_log=False):
        agent = self.set_agent()
        env = self.set_env()
        controller = self.set_controller()
        x_ls = []
        y_ls = []
        cr_his=[]
        cl_his=[]
        log = []
        n_his=10
        last_hit = 0
        dt = 1 / self.fps
        found_source = 0

        if draw_animation:
            fig, img, line, text = self.set_animation(env)

        with h5py.File(plume, 'r') as h5:
            if draw_animation:
                plume = h5['frames'][0:self.Nsteps, 0:env.height, 0:env.width]
            else:
                plume = h5['frames']

            wind_data = pd.read_csv('~/ICT/CPT-IRL/bombyxsim/examples/wind.csv', delimiter=',', header=None).values
            for t in range(self.Nsteps - 1):
                if draw_animation:
                    img.set_data(plume[t])
                    line.set_data(x_ls, y_ls)
                    text.set_text("T: {:.1f}".format(t * dt))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                #if self.off_grid(agent.pos, agent.antenna_length):
                #    # print(colorize('Off grid', 'red'))
                #    found_source = np.nan
                #    break

                if self.reached_goal(agent.pos):
                    found_source = 1
                    self.success_rate += 1
                    if draw_animation:
                        fig.savefig(self.input_dir + 'anime.svg')
                    break

                # if np.cos(agent.theta - np.pi) > self.wind_angle:
                windx=int(agent.pos.x*1280/500)
                windy=int((-agent.pos.y*1024/400)+512)
                current_wind=wind_data[windx,windy]
                #if np.cos(np.pi - agent.theta + self.wind_angle) > 0:
                cr_his.append(env.sample_at(t, agent.right_antenna, plume))
                cl_his.append(env.sample_at(t, agent.left_antenna, plume))
                
                if np.cos(np.pi - agent.theta + current_wind) > 0:
                    
                    if len(cr_his)>=n_his:
                        cr_his_array=np.array(cr_his)
                        cl_his_array=np.array(cl_his)
                        r_threshold = np.percentile(cr_his_array[-n_his:-1],90)
                        l_threshold = np.percentile(cl_his_array[-n_his:-1],90)
                        right_hit = env.hit_at(t, env.mm2px(agent.right_antenna),
                                            plume,threshold=r_threshold)
                        left_hit = env.hit_at(t, env.mm2px(agent.left_antenna),
                                            plume,threshold=l_threshold)
                    else:
                        right_hit = env.hit_at(t, env.mm2px(agent.right_antenna),
                                            plume,threshold=1000)
                        left_hit = env.hit_at(t, env.mm2px(agent.left_antenna),
                                            plume,threshold=1000)

                    if hit_prob < 1:
                        right_hit *= np.random.binomial(1, hit_prob)
                        left_hit *= np.random.binomial(1, hit_prob)

                else:
                    right_hit = 0
                    left_hit = 0

                antennae_hit = (lambda l, r: (l << 1) | r)(left_hit, right_hit)
                if antennae_hit > 0: last_hit = antennae_hit

                if self._ctrl[0] == 'ITX':
                    # print('pos: {}'.format(agent.pos))
                    # print(t)
                    next_pos = controller.control(antennae_hit, agent.pos)
                    agent.move(next_pos)

                elif self._ctrl[0] == 'HYB':
                    action = controller.control(antennae_hit, last_hit, agent.pos)
                    agent.move(controller.policy_mode, action, dt)

                else:
                    #controller.control(antennae_hit, last_hit)
                    controller.control_Cesar(antennae_hit)
                    agent.move(controller.linear_vel, controller.angular_vel, dt)

                controller._tblank += dt
                x_ls.append(agent.pos.x)
                y_ls.append(agent.pos.y)

                if save_log:
                    log.append([t/self.fps, antennae_hit] + agent.log_step + controller.log_step)

                # print('t: {:.2f}, h:{}, {}, {}'.format(t / self.fps, antennae_hit, str(agent), str(controller)))

        traj = pd.DataFrame({'x': x_ls, 'y':y_ls})

        self.performance['success_rate'].append(found_source)
        self.performance['search_time'].append(t/self.fps)

        if save_log:
            log_cols = 'Time,antennae,{},{}'.format(agent.log_header, controller.log_header).split(',')
            log = pd.DataFrame(columns=log_cols, data=log)

        return traj, log