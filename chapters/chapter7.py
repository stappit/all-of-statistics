import math
from collections import namedtuple


CI = namedtuple('CI', ('lower', 'upper'))


class ECDF(object):

    def __init__(self, obs):
        self.observations = obs

    def __call__(self, x):
        return sum(1 for ob in self.observations if ob <= x) / len(self.observations)

    def error(self, a):
        n = len(self.observations)
        return math.sqrt(math.log(2 / a) / (2*n))

    def ci(self, x, a=0.05):
        e = self.error(a)
        y = self(x)
        l = max(y - e, 0)
        u = min(y + e, 1)
        return CI(l, u)
