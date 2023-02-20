import numpy as np
import numpy.random as rn

class MothworldNoise(object):

    def __init__(self, dx, yk, action):

        self.dx = dx
        self.yk = yk
        self.a = action
        self.mean = self.get_mean()
        self.std = self.get_std()
        self.noise = rn.normal(self.mean, self.std)

    def get_mean(self):

        if self.dx == -1:
            return self.get_mean_reset(self.a)[self.yk]

        elif self.dx == 0:
            return self.get_mean_same(self.a)[self.yk]

        elif self.dx == 1:
            return self.get_mean_increase(self.a)[self.yk]

        else:
            return 0.0

    def get_std(self):

        if self.dx == -1:
            return self.get_std_reset(self.a)[self.yk]

        elif self.dx == 0:
            return self.get_std_same(self.a)[self.yk]

        elif self.dx == 1:
            return self.get_std_increase(self.a)[self.yk]

        else:
            return 0.0


    def get_mean_reset(self, action):
        return{
            0: {0:  0.000000, 1: -0.416667, 2: -0.153846, 3:  0.650794},
            1: {0:  0.000000, 1: -0.521739, 2: -0.386364, 3:  0.481481},
            2: {0:  0.000000, 1: -0.458716, 2: -0.183206, 3:  0.475694},
            3: {0:  0.000000, 1: -0.539326, 2: -0.188406, 3:  0.528497}

        }.get(action, 0)

    def get_mean_same(self, action):
        return{
            0: {0: -0.018646, 1: -0.104962, 2:  0.021818, 3:  0.159132},
            1: {0: -0.054674, 1: -0.077482, 2:  0.038377, 3:  0.135881},
            2: {0: -0.047569, 1: -0.102642, 2:  0.037851, 3:  0.145558},
            3: {0: -0.032014, 1: -0.088647, 2:  0.044428, 3:  0.165816}

        }.get(action, 0)

    def get_mean_increase(self, action):
        return{
            0: {0:  0.000000, 1:  1.000000, 2:  2.000000, 3:  3.000000},
            1: {0:  0.000000, 1:  1.000000, 2:  2.000000, 3:  3.000000},
            2: {0:  0.000000, 1:  1.000000, 2:  2.000000, 3:  3.000000},
            3: {0:  0.000000, 1:  1.000000, 2:  2.000000, 3:  3.000000}

        }.get(action, 0)

    def get_std_reset(self, action):
        return{
            0: {0: 0.000000, 1: 0.792961, 2: 0.554700, 3: 0.882788},
            1: {0: 0.000000, 1: 0.897956, 2: 0.537691, 3: 0.742331},
            2: {0: 0.000000, 1: 0.822509, 2: 0.493052, 3: 0.719492},
            3: {0: 0.000000, 1: 0.853551, 2: 0.491545, 3: 0.744391}

        }.get(action, 0)

    def get_std_same(self, action):
        return{
            0: {0: 0.220221, 1: 0.519647, 2: 0.478572, 3: 0.620898},
            1: {0: 0.375751, 1: 0.460620, 2: 0.441603, 3: 0.578062},
            2: {0: 0.346289, 1: 0.546922, 2: 0.444557, 3: 0.598030},
            3: {0: 0.277950, 1: 0.504262, 2: 0.488545, 3: 0.635738}

        }.get(action, 0)

    def get_std_increase(self, action):
        return{
            0: {0: 0.000000, 1: 0.000000, 2: 0.000000, 3: 0.000000},
            1: {0: 0.000000, 1: 0.000000, 2: 0.000000, 3: 0.000000},
            2: {0: 0.000000, 1: 0.000000, 2: 0.000000, 3: 0.000000},
            3: {0: 0.000000, 1: 0.000000, 2: 0.000000, 3: 0.000000}

        }.get(action, 0)
