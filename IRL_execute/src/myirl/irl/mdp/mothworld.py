"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn


class Mothworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, grid_axes, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = [0, 1, 2, 3]
        self.n_actions = len(self.actions)
        self.n_states = grid_size
        self.substate = grid_axes
        # print('Num of states: {}'.format(self.n_states))
        # print('Num of actions: {}'.format(self.n_actions))
        # self.grid_size = np.sqrt(grid_size).astype(int)
        self.grid_size = grid_axes[0]
        # Called 'wind' by original developer but this means the
        # probability of taking each action
        # For now values are hard-coded based on moth observations
        self.wind = {0: 0, 1: 0.061952, 2: 0.463705, 3: 0.474343}
        # Bins for discretization of blank duration τ_b
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability_rev(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
            # [[[self._transition_probability(i, j, k)

    def __str__(self):
        return "Mothworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.substate[0], i // self.substate[0])

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def prob_hit_and_reset(self, action):
        """Dictionary to assign transition probabilities when blank duration is reset to 0

        Args:
            odor_cue_dir (int): Direction of odor stimulus
        """
        return{
            0: 0.000589,
            1: 0.000764,
            2: 0.004918,
            3: 0.004883
        }.get(action, 0)

    def prob_hit_and_nochange(self, action):
        """Dictionary to assign transition probabilities when blank duration stays the same

        Args:
            odor_cue_dir (int): Direction of odor stimulus
        """
        return{
            0: 0.012270,
            1: 0.020916,
            2: 0.101647,
            3: 0.091682
        }.get(action, 0)

    def prob_hit_and_increase(self, action):
        """Dictionary to assign transition probabilities when blank duration increases

        Args:
            odor_cue_dir (int): Direction of odor stimulus
        """
        return{
            0: 0.000554,
            1: 0.001079,
            2: 0.005315,
            3: 0.004207
        }.get(action, 0)

    def prob_no_hit(self, action):
        """Dictionary to assign transition probabilities when blank duration increases

        Args:
            odor_cue_dir (int): Direction of odor stimulus
        """
        return{
            0: 0.081915,
            1: 0.056687,
            2: 0.297817,
            3: 0.303413
        }.get(action, 0)

    def action_rng(self):

        return rn.randint(self.n_actions)
    
    def _transition_probability_rev(self, i, j, k):
        """WIP of revised transition probability

        Args:
            i (int): [Current state]
            j (int): [Action]
            k (int): [Next state]

        -> p(s'| a, s) [(i x j x k) matrix of ints]
        """

        xi, yi = self.int_to_point(i)
        xk, yk = self.int_to_point(k)

        # Any antenna reacted in the future
        if (yk > 0):

            # Blank duration didn't change
            if (xk - xi == 0):
                # print('s_i:{}, s_j:{}, p={}'.format(
                    # (xi, yi), (xk, yk), self.prob_hit_and_nochange(j)))
                # return rn.normal(p_hn, 0.01)
                return self.prob_hit_and_nochange(j)

            # Blank duration got reset to 0
            elif xk == 0:
                # print('s_i:{}, s_j:{}, p={}'.format(
                    # (xi, yi), (xk, yk), self.prob_hit_and_reset(j)))
                # return rn.normal(p_hr, 0.01)
                return self.prob_hit_and_reset(j)

            else:
                # print('s_i:{}, s_j:{}, p=0.0'.format((xi, yi), (xk, yk)))
                return 0.0

        # Blank duration increased by 1 because there was no hit
        # elif not(yk | 0) and (xk - xi == 1):
        elif (xk - xi == 1) and not (yi | yk):
            # print('s_i:{}, s_j:{}, p={}'.format(
                # (xi, yi), (xk, yk), self.prob_no_hit(j)))
            # return rn.normal(p_nh, 0.01)
            return self.prob_no_hit(j)

        else:
            return 0.0

    def _transition_probability(self, i, j, k):
        """
        Get the probability of moving to state sp given a state s and action a
        i: State (int)
        j: Action (int)
        k: Future state (int)

        -> p(s'| a, s) [(i x j x k) matrix of ints]
        """

        xi, yi = self.int_to_point(i)
        xk, yk = self.int_to_point(k)
        # actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        actions = ((0, rn.randint(2)), (0, 2), (1, 3), (-xi, rn.randint(3)))
        xj, yj = actions[j]
        # grid_size = self.grid_size
        xn, yn  = (self.substate[0], self.substate[1])
        pi = self.prob_increase(yk)[j]
        pr = self.prob_reset(yk)[j]
        w = self.wind[j]
        v = (pr*(1 - pi))
        n = self.n_actions

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            
            if (xk - xi > 0):
                # return self.prob_increase(yk)[j]
                return pi
            else:
                # return self.prob_reset(yk)[j]
                return pr
                # return pr

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            # return self.wind[j]/self.n_actions
            return (pr*(1 - pi))/n

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (xn - 1, yn - 1),
                        (0, xn - 1), (yn - 1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < xn and
                    0 <= yi + yj < yn):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                # + 2*(pi + pr)/self.n_actions
                return 1 - v# + 2*v/n
                # return 1 - self.prob_reset(yk)[j] + (self.prob_reset(yk)[j] + self.prob_increase(yk)[j])/self.n_actions
                # return 1 - self.prob_increase(yk)[j]
            else:
                # We can blow off the grid in either direction only by wind.
                return 0.0
                # return (self.prob_increase(yk)[j] + self.n_actions)/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, xn - 1} and
                    yi not in {0, yn - 1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < xn and
                    0 <= yi + yj < yn):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                # return 1 - self.wind[j] + self.wind[j]/self.n_actions
                # + (pi + pr)/self.n_actions
                return 1 - v# + v/n
            else:
                # We can blow off the grid only by wind.
                # return self.wind[j]/self.n_actions
                return 0.0

    def _gw_transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        grid_size = self.grid_size
        xi, yi = self.int_to_point(i)
        xj, yj = actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind[j] + self.wind[j]/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind[j]/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (grid_size-1, grid_size-1),
                        (0, grid_size-1), (grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < grid_size and
                    0 <= yi + yj < grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind[j] + 2*self.wind[j]/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind[j]/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, grid_size-1} and
                    yi not in {0, grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < grid_size and
                    0 <= yi + yj < grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind[j] + self.wind[j]/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind[j]/self.n_actions

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                                  trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory]
                   for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                              random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)
