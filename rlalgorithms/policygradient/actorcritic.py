import gym

import numpy as np
import h5py
import pandas as pd
import collections
import itertools
import logging
import sys
# import random
import tensorflow as tf
from tqdm import tqdm

EpisodeStats = collections.namedtuple("Stats", ["lengths", "rewards"])
SearchPerformance = collections.namedtuple("Performance", ["Success", "Time_mean", "Time_std"])

logging.getLogger('matplotlib.font_manager').disabled = True
_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s"
    logging.basicConfig(level=loglevel,
                        stream=sys.stdout,
                        format=logformat,
                        datefmt="%Y-%m-%d %H:%M:%S")


class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    def __init__(self, nstates, nactions, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.nstates = nstates
            self.nactions = nactions

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, self.nstates)
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=self.nactions,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {
            self.state: state,
            self.target: target,
            self.action: action
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Value Function approximator. 
    """
    def __init__(self, nstates, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.nstates = nstates

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, self.nstates)
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(gym,
                 plumes,
                 estimator_policy,
                 estimator_value,
                 n_episodes,
                 discount_factor=0.9):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    setup_logging(logging.DEBUG)
    _logger.info('Starting AC')
    # Keeps track of useful statistics
    stats = EpisodeStats(lengths=np.zeros(n_episodes),
                         rewards=np.zeros(n_episodes))

    dt = 1 / gym.fps
    action_space = gym.get_action_space()
    nactions = len(action_space)
    nstates = gym.rewards.size
    dp_policy = gym.policy.T.reshape(nstates, nactions)
    learned_policy = collections.defaultdict(lambda: np.zeros(nactions))

    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"])

    np.random.seed(99)
    for i in tqdm(range(n_episodes)):
        # Reset the environment and pick the fisrst action
        # state = env.reset()
        plume = np.random.choice(plumes)
        # _logger.info('set_env')
        env = gym.set_env(plume)
        # _logger.info('set_agent')
        agent = gym.set_agent()
        # _logger.info('set_controller')
        controller = gym.set_controller()

        x_ls = []
        y_ls = []
        found_source = 0
        hit_noise = gym.cfg['hit_noise']
        hit_eps = np.random.binomial(1, gym.cfg['hit_probability'])
        done = False
        reward = 0.
        state = 0
        next_state = 0

        # if i == (n_episodes - 1):
        # fig, img, line, text = gym.set_animation(env)

        episode = []

        # _logger.info('Loading h5')
        # with h5py.File(plume, 'r') as h5:
        # h5plume = h5['frames']

        for t in itertools.count():

            # if i == (n_episodes - 1):
            #     img.set_data(env.plume[t])
            #     line.set_data(x_ls, y_ls)
            #     text.set_text("T: {:.2f}, episode: {}".format(t * dt, i))
            #     fig.canvas.draw()
            #     fig.canvas.flush_events()

            # Take a step
            # _logger.info('Sample plume at antennae')
            if np.cos(np.pi - agent.theta + gym.wind_angle) > 0:
                right_hit = env.hit_at(t, env.mm2px(agent.right_antenna))
                left_hit = env.hit_at(t, env.mm2px(agent.left_antenna))

                if hit_noise:
                    right_hit *= hit_eps
                    left_hit *= hit_eps

            else:
                right_hit = 0
                left_hit = 0

            antennae_hit = (lambda l, r: (l << 1) | r)(left_hit, right_hit)

            # _logger.info('estimator_policy.predict(state)')
            action_probs = estimator_policy.predict(state)
            # action_probs = dp_policy[state]
            # action_probs /= action_probs.sum()
            action = np.random.choice(np.arange(len(action_probs)),
                                    p=action_probs)

            controller.random_action(antennae_hit, action)
            agent.move(controller.linear_vel, controller.angular_vel, dt)
            next_state = controller.state
            reward = controller.reward

            if gym.off_grid(agent.pos, 1.5 * agent.antenna_length):
                # print(colorize('Off grid', 'red'))
                found_source = np.nan
                # reward -= 2e4
                done = True
                # break

            if t >= (gym.Nsteps - 1):
                found_source = np.nan
                # reward = -10.
                done = True

            if gym.reached_goal(agent.pos):
                found_source = 1
                # reward += 100
                done = True
                # break
            # next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(
                Transition(state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done))

            # Update statistics
            stats.rewards[i] += reward
            stats.lengths[i] = t

            # Calculate TD Target
            # _logger.info('Calculate TD Target')
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)
            learned_policy[state] = action_probs

            controller._tblank += dt
            x_ls.append(agent.pos.x)
            y_ls.append(agent.pos.y)



            if done:
                break

            state = next_state

        gym.performance['success_rate'].append(found_source)
        gym.performance['search_time'].append(t / gym.fps)

    performance = pd.DataFrame(gym.performance)
    traj = pd.DataFrame({'x': x_ls, 'y': y_ls})
    # print(action_probs)
    # fig, ax = plt.subplots()
    gym.plot_trajectory(traj, 'testtraj')

    return stats, performance, learned_policy
def run(gym, plumes, n_episodes):

    # tf.compat.v1.reset_default_graph()
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(64, 4)
    value_estimator = ValueEstimator(64)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Note, due to randomness in the policy the number of episodes you need to learn a good
        # policy may vary. ~300 seemed to work well for me.
        stats, performance, learned_policy = actor_critic(
            gym, plumes, policy_estimator, value_estimator, n_episodes)

        pistar = pd.DataFrame(learned_policy).T
        pistar.sort_index(inplace=True)
        print(pistar.describe())
        # possible_states = range(64)
        # full_policy = []

        # for s in possible_states:
        #     full_policy.append(policy_estimator.predict(s))

    return stats, performance, pistar#, full_policy
