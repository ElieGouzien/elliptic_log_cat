#!/usr/bin/env python3
# coding: utf-8
"""
Tools for algorithmic resource estimate.

@author: Élie Gouzien
"""
import numbers
from collections import namedtuple
from datetime import timedelta

AlgoOpts = namedtuple('AlgoOpts',
                      'prob, algo, s, n, we, wm, c, '
                      ' windowed, mesure_based_deand, mesure_based_unlookup,'
                      'parallel_cnots',
                      defaults=('rsa', 'Ekera', 1, None, None, None, None,
                                True, None, True,
                                False))
AlgoOpts.__doc__ = (
"""
AlgoOpts(prob, algo, s, n, we, wm, c, windowed,
         mesure_based_deand, mesure_based_unlookup, 'parallel_cnots')

Parameters:
    prob : addressed problem : 'rsa' or 'elliptic_log'
    algo : used algorithm : 'Ekera' or 'Shor'
    s    : tradeoff in Ekera's algorithm. Must be None if 'Shor' is chosen.
    n    : number of bits of N
    we   : size of the exponentiation window
    wm   : size of the multiplication window
    c    : number of bits added for coset representation
    windowed : use windowed algorithm ? (True or False)
    mesure_based_deand : measurement based AND uncomputation
                        (only for type == '3dcolor', otherwise takes by
                         default the obvious option)
    mesure_based_unlookup : unlookup based on measurement?
                            (only usable for type == 'alice&bob(2)' because |0>
                             preparation is long)
    parallel_cnots : CNOT with multiple targets cost as much as a single CNOT?
                     Be careful, it is not a full parallelisation of the CNOTs
                     (routing might cause problems).
""")

LowLevelOpts = namedtuple('LowLevelOpts', 'debitage, d1, d2, d, n, tc, tr, pp',
                          defaults=(2, None, None, None, None, 1e-6, 1e-6,
                                    1e-3))
LowLevelOpts.__doc__ = """LowLevelOpts(debitage, d1, d2, d, n, tc, tr, pp)

Parameters:
    debitage : cut of tetrahedron for '3dcolor' error correction code
    d1 : distance of first step of distillation/applying
    d2 : distance of second step of distillation/applying
    d  : main code distance
    n  : average number of photons, only for Alice&Bob's cat
    tc : cycle time
    tr : reaction time
    pp : error probability on physical gates (inc. identity)
"""

Params = namedtuple('Params', 'type, algo, low_level')
Params.__doc__ = """Params(type, algo, low_level)

Parameters:
    type      : type of error correction : 'alice&bob2' or None
    algo      : algorithm options, type AlgoOpts
    low_level : low level options, type LowLevelOpts
"""


class PhysicalCost(namedtuple('PhysicalCost', ('p', 't'))):
    """Physical cost of some gates: error probability and runtime.

    Attributes
    ----------
        p : error probability.
        t : execution time.

    Methods
    -------
        Has same interface as namedtuple, except for listed operators.

    Operators
    ---------
        a + b : cost of serial execution of a and b.
        k * a : cost of serial execution of a k times (k can be float).
        a | b : cost of parallel execution of a and b.
        k | b : cost of k parallel executions of b.
    """

    def __add__(self, other):
        """Cost of sequential execution of self and other."""
        if not isinstance(other, __class__):
            return NotImplemented
        return __class__(1 - (1 - self.p)*(1 - other.p), self.t + other.t)

    def __mul__(self, other):
        """Cost of sequential execution of self other times.

        Other does not need to be integer (as some gates are probabilistically
                                           applied).
        """
        if not isinstance(other, numbers.Real):
            return NotImplemented
        if self.p >= 1:
            return __class__(self.p, self.t * other)
        return __class__(1 - (1 - self.p)**other, self.t * other)

    def __rmul__(self, other):
        """Right multiplication."""
        return self * other

    def __sub__(self, other):
        """Subtraction: revert previous of future addition."""
        return self + (-1 * other)

    def __or__(self, other):
        """Cost of parallel execution of self and other.

        If other is a real number, it computes parallel execution of 'other'
        times the 'self' operation.
        """
        if isinstance(other, __class__):
            return __class__(1 - (1 - self.p)*(1-other.p),
                             max(self.t, other.t))
        if isinstance(other, numbers.Real):
            return __class__(1 - (1 - self.p)**other, self.t)
        return NotImplemented

    def __ror__(self, other):
        """Cost of parallel execution of self and other (right version)."""
        return self | other

    @property
    def exp_t(self):
        """Average runtime (several intents might be required)."""
        if self.p is None:
            return self.t
        if self.p >= 1:
            return float('inf')
        return self.t / (1 - self.p)

    @property
    def exp_t_str(self):
        """Format average runtime."""
        try:
            return timedelta(seconds=self.exp_t)
        except OverflowError:
            if self.exp_t == float('inf'):
                return "∞"
            return str(round(self.exp_t/(3600*24*365.25))) + " years"

    def __str__(self):
        """Readable representation of a PhysicalCost."""
        # pylint: disable=C0103
        try:
            t = timedelta(seconds=self.t)
        except OverflowError:
            if self.t == float('inf'):
                t = "∞"
            else:
                t = str(round(self.t/(3600*24*365.25))) + " years"
        return f"PhysicalCost(p={self.p}, t={t}, exp_t={self.exp_t_str})"
