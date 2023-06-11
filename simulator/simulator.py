"""
Setup an environment, agent and controller for an olfactory search simulation
"""

import os
import json
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import numpy as np

from utils.geometry import Point
import silkmoth
from controllers import silkmoth_irl, programmed_behavior
from envs import smoke_video, wind_tunnel

class Simulator(object):
    def __init__(self, input_dir, config_file, tlim, env='smokevid',
                agt='silkmoth', controller='KPB'):
        self.input_dir = input_dir
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
            self.policy = self.get_policy()
            self.irl_num_states = self.cfg["irl"]["num_states"]

    def get_bin_edges(self):
        f = os.path.join(self.input_dir, 'bin_edges', f'{self._ctrl[1]}.csv')
        bin_edges = pd.read_csv(f)
        return bin_edges

    def get_policy(self):
        f = os.path.join(self.input_dir, 'irl_policies', f'{self._ctrl[2]}.h5')
        with h5py.File(f, 'r') as hf:
            policy = hf['policy'][:]
        return policy

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

    def set_env(self, plume):
        cfg = self.cfg['env']

        return {
            'smokevid':
            smoke_video.SmokeVideo(plume, self.Nsteps, cfg['width'],
                                   cfg['height'], tuple(cfg['srcpos']),
                                   tuple(cfg['xspace']), tuple(cfg['yspace'])),
            'windtunnel':
            wind_tunnel.WindTunnel(plume, self.Nsteps, cfg['width'],
                                   cfg['height'], tuple(cfg['srcpos']),
                                   tuple(cfg['xspace']), tuple(cfg['yspace']))
        }.get(self._env, None)

    def set_controller(self):
        return{
            'KPB': programmed_behavior.KPB(self.fps),
            'IRL': silkmoth_irl.SilkmothIRL(
                self.fps, self.bin_edges, self.irl_num_states, self.policy)
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
        fig.canvas.draw()
        plt.show(block=False)
        return fig, img, line

    def off_grid(self, p: Point, m):
        xs = self.cfg['env']['xspace']
        ys = self.cfg['env']['yspace']
        og = not ((min(xs) + m < p.x < max(xs) - m) and
                (min(ys) + m < p.y < max(ys) - m))
        return og

    def reached_goal(self, p: Point):
        radius = self.cfg['env']['goalr']
        return (np.sqrt(p.x**2 + p.y**2) < radius)

    def plot_trajectory(self, traj, f):

        xsrc, ysrc = tuple(self.cfg['env']['srcpos'])
        radius = self.cfg['env']['goalr']

        fig, ax = plt.subplots()
        ax.add_artist(Circle((xsrc, ysrc), radius, color='r', fill=False, linestyle='--', linewidth=0.5, zorder=1))
        ax.scatter(traj.iloc[0,0], traj.iloc[0,1], marker='.')
        ax.plot(traj.iloc[:,0], traj.iloc[:,1], zorder=3)
        ax.scatter(xsrc, ysrc, marker='*', c='k')

        ax.set_xlim(self.cfg['env']['xspace'])
        ax.set_ylim(self.cfg['env']['yspace'])
        ax.set_aspect('equal')
        plt.savefig(f, dpi=300)


    def run(self, plume, draw_animation=False, save_log=False):
        agent = self.set_agent()
        env = self.set_env(plume)
        controller = self.set_controller()
        x_ls = []
        y_ls = []
        log = []
        last_hit = 0
        dt = 1 / self.fps
        found_source = 0
        hit_noise = self.cfg['hit_noise']
        hit_eps = np.random.binomial(1, self.cfg['hit_probability'])

        if draw_animation:
            fig, img, line = self.set_animation(env)

        for t in range(self.Nsteps - 1):
            if draw_animation:
                img.set_data(env.plume[t])
                line.set_data(x_ls, y_ls)
                fig.canvas.draw()
                fig.canvas.flush_events()

            if self.off_grid(agent.pos, agent.antenna_length):
                found_source = np.nan
                break

            if self.reached_goal(agent.pos):
                found_source = 1
                break

            # if np.cos(agent.theta - np.pi) > self.wind_angle:
            if np.cos(np.pi - agent.theta + self.wind_angle) > 0:
                right_hit = env.hit_at(t, env.mm2px(agent.right_antenna))
                left_hit = env.hit_at(t, env.mm2px(agent.left_antenna))

                if hit_noise:
                    right_hit *= hit_eps
                    left_hit *= hit_eps

            else:
                right_hit = 0
                left_hit = 0

            antennae_hit = (lambda l, r: (l << 1) | r)(left_hit, right_hit)
            if antennae_hit > 0: last_hit = antennae_hit

            controller.control(antennae_hit, last_hit, dt)
            agent.move(controller.linear_vel, controller.angular_vel, dt)

            controller._tblank += dt
            x_ls.append(agent.pos.x)
            y_ls.append(agent.pos.y)

            if save_log:
                log.append([t/self.fps, antennae_hit] + agent.log_step + controller.log_step)

        traj = pd.DataFrame({'x': x_ls, 'y':y_ls})

        self.performance['success_rate'].append(found_source)
        self.performance['search_time'].append(t/self.fps)

        if save_log:
            log_cols = f'Time, hit, {agent.log_header}, {controller.log_header}'.split(',')
            log = pd.DataFrame(columns=log_cols, data=log)
            return traj, log
        else:
            return traj