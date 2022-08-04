#!/usr/bin/env python3
# coding: utf-8
# pylint: disable=C0103, C0144
"""
Tools for describing error correction and cost of logical circuits.

@author: Élie Gouzien
"""
from math import exp, floor, ceil, sqrt
from abc import ABC, abstractmethod

from tools import Params, PhysicalCost

USE_MONTGOMERY = True  # TODO: take it as an option.


# %% Abstract class that contains implementation details
class ErrCorrCode(ABC):
    """Abstract class for describing an error correcting code.

    Default implementation assumes same cost for initializing or measuring in
    X or Z basis.
    CZ is supposed as costly as CNOT.
    """

    def __new__(cls, params: Params, *args, **kwargs):
        """Create new instance, choosing concrete class from params.type."""
        if cls is ErrCorrCode:
            if params.type == 'alice&bob2':
                return AliceAndBob2(params, *args, **kwargs)
            elif params.type is None:
                return NoCorrCode(params, *args, **kwargs)
            else:
                raise ValueError("'params.type' not valid!")
        return super().__new__(cls)

    @abstractmethod
    def __init__(self, params: Params):
        """Initialize the code parameters."""
        self.params = params
        # Elementary gates cost
        self.gate1 = None
        self.cnot = None
        self.init = None  # initialisation  of one logical qubit
        self.mesure = None  # measurement of one logical qubit
        # Processor properties
        self.correct_time = None  # time for correcting one logical qubit
        self.proc_qubits = None
        # Set and check for special parameters
        self._set_measure_based_deand()
        self._check_measure_based_deand()
        self._check_mesure_based_unlookup()
        self._check_shor_no_s()

    def _check_shor_no_s(self):
        """Check s option is not used with Shor's algorithm."""
        if self.params.algo.algo == 'Shor' and self.params.algo.s is not None:
            raise ValueError("Shor's algorithm don't have tradeoff s param.")

    @property
    def ne(self):
        """Compute the number of times the elementary algorithm is repeated.

        Correspond with the size of the register to which the QFT is applied.
        Size of the exponent for factorisation or the multiplier for
        elliptic curve discrete logarithm.
        """
        n, s = self.params.algo.n, self.params.algo.s
        if self.params.algo.prob == 'rsa':
            if self.params.algo.algo == 'Shor':
                return 2*n
            elif self.params.algo.algo == 'Ekera':
                m = ceil(n/2) - 1
                # Case s=1: See A.2.1 of eprint.iacr.org/2017/1122 for details.
                delta = 20 if n >= 1024 else 0  # Only for s == 1.
                l = m - delta if s == 1 else ceil(m/s)
                return m + 2*l
            else:
                raise ValueError("'self.params.algo.algo' : 'Shor' or 'Ekera'")
        elif self.params.algo.prob == 'elliptic_log':
            if self.params.algo.algo == 'Shor':
                return 2*n
            elif self.params.algo.algo == 'Ekera':
                return n + 2*ceil(n/s)
            else:
                raise ValueError("'self.params.algo.algo' : 'Shor' or 'Ekera'")
        else:
            raise ValueError("'self.params.algo.prob' must be 'rsa' or "
                             "'elliptic_log'!")

    @property
    def and_gate(self):
        """Cost of AND computation in an ancillary qubit."""
        # see arXiv:1805.03662, fig. 4
        # |T> = T |+>, preparing |+> assumed at same cost as |0>
        return self.init + self.gate1*6 + self.cnot*3

    @property
    def deand(self):
        """Cost of AND uncomputation (measurement-based)."""
        # Hadamard gates are merged with preparation/measurements as X and Z
        # basis measurement are assumed to have equal cost (as in CSS codes).
        return self.mesure + 0.5*self.cnot  # CZ assumed as CNOT

    def _set_measure_based_deand(self, val=True):
        """If 'measure_based_deand' is None, set the correct value."""
        if self.params.algo.mesure_based_deand is None:
            self.params = self.params._replace(algo=self.params.algo._replace(
                mesure_based_deand=val))

    def _check_measure_based_deand(self):
        """Check consistency of the deand method with the parameter."""
        if not self.params.algo.mesure_based_deand:
            raise ValueError("params.algo.mesure_based_deand must be true!")

    @property
    def and_deand(self):
        """Cost of computing and uncomputing AND."""
        return self.and_gate + self.deand

    @property
    def toffoli(self):
        """Cost of a full Toffoli gate."""
        try:
            return self._toffoli
        except AttributeError:
            return self.and_deand + self.cnot

    @toffoli.setter
    def toffoli(self, value):
        self._toffoli = value

    @toffoli.deleter
    def toffoli(self):
        del self._toffoli

    @property
    def fredkin(self):
        """Controlled swap."""
        return self.toffoli + 2*self.cnot

    def multi_and_deand(self, nb_and):
        """AND gate between nb_and qubits.

        Not that the generalized toffoli is typically obtained with an
        additional CNOT (last AND + CNOT can be merged in a Toffoli if
                         available).
        """
        # Implementation assuming available ancillary qubits
        return (nb_and-1)*self.and_deand

    @property
    def maj(self):
        """Cost of MAJ operation, with ancillary qubit."""
        # See arXiv:quant-ph/0410184 for MAJ and UMA notation
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return self.and_gate + nb_cnots*self.cnot

    @property
    def maj_dag(self):
        """Cost of MAJ^dagger operation, with ancillary qubit."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return self.deand + nb_cnots*self.cnot

    @property
    def uma(self):
        """Cost of UMA operation, with ancillary qubit."""
        # No parallelization here
        return 3*self.cnot + self.deand

    @property
    def uma_dag(self):
        """Cost of UMA^dagger operation, with ancillary qubit."""
        # No parallelization here
        return 3*self.cnot + self.and_gate

    @property
    def uma_ctrl(self):
        """Cost of controlled UMA operation."""
        nb_cnots = 2 if self.params.algo.parallel_cnots else 3
        return nb_cnots*self.cnot + self.deand + self.toffoli

    def add(self, n=None):
        """Cost of full adder modulo power of two (with ancillary qubits)."""
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        return (n - 2)*(self.maj + self.uma) + 3*self.cnot + self.and_deand

    def add_nomod(self, n=None):
        """Cost of addition (with ancillary qubits), with output carry."""
        if n is None:
            n = self.params.algo.n
        return ((n - 2)*(self.maj + self.uma) + self.maj + 3*self.cnot
                + self.and_deand)

    def add_ctrl(self, n=None):
        """Cost of controlled addition, modulo power of two."""
        if n is None:
            n = self.params.algo.n
        return ((n - 2)*(self.maj + self.uma_ctrl) + 2*self.cnot
                + self.and_deand + 2*self.toffoli)

    def add_ctrl_nomod(self, n=None):
        """Cost of controlled addition, with output carry."""
        if n is None:
            n = self.params.algo.n
        return ((n - 1)*(self.maj + self.uma_ctrl) + self.and_deand
                + self.and_gate + self.toffoli)

    def comparison(self, n=None):
        """Cost of comparison, without use."""
        if n is None:
            n = self.params.algo.n
        return (n - 1)*(self.uma + self.uma_dag) + self.and_deand + 4*self.cnot

    @property
    def semi_classical_maj(self):
        """Cost of MAJ, semi-classical version."""
        return self.and_gate + 0.5*self.cnot + self.gate1

    @property
    def semi_classical_maj_dag(self):
        r"""Cost of MAJ^\dagger, semi-classical version."""
        return self.deand + 0.5*self.cnot + self.gate1

    @property
    def semi_classical_uma(self):
        """Cost of UMA, controlled semi-classical version."""
        return self.deand + 1.5*self.cnot + 0.5*self.gate1

    @property
    def semi_classical_ctrl_maj(self):
        """Cost of semi-classical controlled MAJ."""
        return self.and_gate + 1.5*self.cnot

    @property
    def semi_classical_ctrl_uma(self):
        """Cost of semi-classical controlled UMA, fast version."""
        # Warning: it's not literally a controlled UMA operation.
        return self.deand + 2.5*self.cnot

    @property
    def semi_classical_ctrl_uma_true(self):
        """Cost of semi-classical controlled UMA, slow version."""
        # This one is a true controlled UMA operation.
        return self.gate1 + 0.5*self.cnot + self.deand + self.toffoli

    def semi_classical_add(self, n=None):
        """Semi-classical addition."""
        if not n >= 3:
            raise ValueError("For this adder, n >= 3.")
        return ((n-3)*(self.semi_classical_maj + self.semi_classical_uma)
                + self.toffoli + 2.5*self.cnot + 2*self.gate1)

    def semi_classical_add_nomod(self, n=None):
        """Semi-classical addition, with output carry."""
        if not n >= 2:
            raise ValueError("For this adder, n >= 2")
        return ((n-2)*(self.semi_classical_maj + self.semi_classical_uma)
                + 1.5*self.cnot + 1*self.gate1 + self.and_gate)

    def semi_classical_ctrl_add(self, n=None):
        """Cost of controlled semi-classical addition."""
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        return ((n-2)*(self.semi_classical_ctrl_maj
                       + self.semi_classical_ctrl_uma)
                + 2*self.cnot + 0.5*self.and_deand)

    def semi_classical_ctrl_ctrl_add(self, n=None):
        """Cost of doubly controlled semi-classical addition."""
        return self.and_deand + self.semi_classical_ctrl_add(n)

    def semi_classical_ctrl_add_nomod(self, n=None):
        """Semi-classical controlled addition, with output carry."""
        return ((n-2)*(self.semi_classical_ctrl_maj
                       + self.semi_classical_ctrl_uma)
                + self.semi_classical_ctrl_maj
                + 2*self.cnot + 0.5*self.and_deand)

    def semi_classical_comparison(self, n=None):
        """Semi-classical comparison, without use."""
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        return ((n-1)*(self.semi_classical_maj
                       + self.semi_classical_maj_dag)
                + self.cnot)

    def semi_classical_neg_mod(self, n=None):
        """Modular negation (modulo a classical number)."""
        if n is None:
            n = self.params.algo.n
        return n*self.gate1 + self.semi_classical_add(n)

    def semi_classical_ctrl_neg_mod(self, n=None):
        """Controlled modular negation (modulo a classical number)."""
        if n is None:
            n = self.params.algo.n
        nb_cnots = 1 if self.params.algo.parallel_cnots else n
        return nb_cnots*self.cnot + self.semi_classical_ctrl_add(n)

    def modular_reduce(self, n=None):
        """Cost of modular reduction (standard representation).

        Cost of computing |x> -> |x % p> |x // p>.
        x as n+1 qubits at input, and n at output. p is classically known and
        as n qubits.
        """
        if n is None:
            n = self.params.algo.n
        return ((n-1)*(self.semi_classical_maj
                       + self.semi_classical_ctrl_uma_true)
                + 1.5*self.gate1 + 2*self.cnot + self.and_gate + self.toffoli)

    def add_mod(self, n=None):
        """Cost of modular addition, in standard representation.

        Both numbers are quantum, but classical modulo.
        Directly compatible with Montgomery representation.
        """
        if n is None:
            n = self.params.algo.n
        return (self.add_nomod(n) + self.modular_reduce(n)
                + self.comparison(n) + self.cnot)

    def rotate(self, n=None, k=None):
        """Qubit rotation, to implement multiplication or division by 2^k.

        Assumed free as only relabelling.
        """
        return PhysicalCost(0, 0)

    def _defaul_lookup_sizes(self):
        """Compute default sizes 'w' et 'n' for table lookup."""
        # total window input size
        w = self.params.algo.we + self.params.algo.wm
        # Numbers read < N : despite coset representation normal size OK.
        n = self.params.algo.n
        return w, n

    def lookup(self, w=None, n=None):
        """Cost of table-lookup circuit, address (target) of sizes w (n).

        States initialisation excluded from this function.
        """
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        nb_cnots = 2**w - 2
        nb_cnots += 2**w if self.params.algo.parallel_cnots else 2**w * n/2
        return 2*self.gate1 + nb_cnots*self.cnot + (2**w - 2)*self.and_deand

    def unary_ununairy(self, size=None):
        """Cost of unary representation computation and uncomputation.

        Includes initialisation and destruction of qubits for the unary
        representation (via self.and_deand).
        """
        # first NOT is not counted as |1> can be directly initialized.
        if size is None:
            size = floor((self.params.algo.we + self.params.algo.wm)/2)
        return self.init + 2*(size-1)*self.cnot + (size-1)*self.and_deand

    def unlookup(self, w=None, n=None):
        """Cost of table-lookup uncomputation."""
        # Hadamard gates are merged with preparation/measurement.
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        return (n*self.mesure
                + self.unary_ununairy(floor(w/2))
                # + 2*floor(w/2)*self.gate1  # CZ same cost as CNOT
                + self.lookup(w=ceil(w/2), n=floor(w/2)))

    def look_unlookup(self, w=None, n=None):
        """Cost of table lookup and unlookup."""
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        # No initialisation because we recycle the target from previous steps.
        # (first initialisation neglected).
        # Ancillary qubits are initialised within the subfunctions.
        return self.lookup(w, n) + self.unlookup(w, n)

    def _check_mesure_based_unlookup(self):
        """Check consistency of the deand method with the parameter."""
        if not self.params.algo.mesure_based_unlookup:
            raise ValueError("params.algo.mesure_based_deand must be true!")

    def initialize_coset_reg(self):
        """Coset representation register initialization."""
        # Hadamard gates are merged with preparation/measurement.
        n, m = self.params.algo.n, self.params.algo.c
        return (m*(self.init + self.mesure)
                + m*self.semi_classical_ctrl_add(n+m)
                + 0.5*m*(self.semi_classical_comparison(n+m) + self.gate1))

    def modular_exp_windowed(self):
        """Cost of modular exponentiation, with windowed arithmetics."""
        _, _, _, n, we, wm, c, _, _, _, _ = self.params.algo
        # Factor 2: see arXiv:1905.07682, Fig. 6.
        nb = 2 * (self.ne/we) * (n + c)/wm
        classical_error = PhysicalCost(2**(-c), 0)
        return (nb*(self.add() + self.look_unlookup() + classical_error)
                + 2*self.initialize_coset_reg())

    def modular_exp_controlled(self):
        """Cost of modular exponentiation, with controlled arithmetics."""
        _, _, _, n, _, _, c, _, _, _, _ = self.params.algo
        nb = 2 * self.ne * (n + c)
        classical_error = PhysicalCost(2**(-c), 0)
        return (nb*(self.semi_classical_ctrl_ctrl_add() + classical_error)
                + 2 * self.initialize_coset_reg()
                + self.ne*(n + c)*(2*self.cnot + self.toffoli))

    def modular_exp(self):
        """Modular exponentiation cost, version taken from parameters."""
        if self.params.algo.windowed:
            return self.modular_exp_windowed()
        return self.modular_exp_controlled()

    def factorisation(self):
        """Factorisation resources.

        Fourier transform neglected. Algorithm options taken into account.
        """
        if self.params.algo.algo == 'Shor':
            classical_error = PhysicalCost(1/4, 0)
            return self.modular_exp() + classical_error
        elif self.params.algo.algo == 'Ekera':
            return self.params.algo.s * self.modular_exp()
        else:
            raise ValueError("'self.params.algo.algo' : 'Shor' or 'Ekera'")

    def montgomery_mul_windowed(self, n=None, w=None):
        """Windowed Montgomery multiplication.

        Creates garbage qubits.
        """
        if n is None:
            n = self.params.algo.n
        if w is None:
            w = self.params.algo.wm
        loops = ceil(n/w)
        return (n*self.add_ctrl_nomod(n+1) + loops*self.add(n+w+1)
                + n*self.cnot + loops*self.look_unlookup(w, n+w)
                + loops*self.rotate(n+w, w)
                # Final reduction:
                + self.modular_reduce(n))

    def montgomery_mul_basic(self, n=None):
        """Montgomery multiplication.

        Creates garbage qubits.
        """
        if n is None:
            n = self.params.algo.n
        return (n*self.add_ctrl_nomod(n+1) + n*self.cnot
                + n*self.semi_classical_ctrl_add_nomod(n+2)
                + n*self.rotate(n+2, 1)
                # Réduction finale :
                + self.modular_reduce(n+1))

    def mul(self, n=None):
        """Cost of standard quantum-quantum out of place multiplication."""
        if n is None:
            n = self.params.algo.n
        return n*(self.add_mod(n)
                  + self.rotate(n+1, 1) + self.modular_reduce(n) + self.cnot)

    def square(self, n=None):
        """Cost of squaring."""
        if n is None:
            n = self.params.algo.n
        return self.mul(n) + 2*n*self.cnot

    def montgomery_square_windowed(self, n=None, w=None):
        """Squaring with Montgomery windowed multiplication."""
        if n is None:
            n = self.params.algo.n
        return self.montgomery_mul_windowed(n, w) + 2*n*self.cnot

    def montgomery_square_basic(self, n=None):
        """Squaring with Montgomery multiplication (controlled version)."""
        if n is None:
            n = self.params.algo.n
        return self.montgomery_mul_basic(n) + 2*n*self.cnot

    def kaliski(self, n=None):
        """Out of place modular inversion: quantum version of Kaliski algo.

        Works on number in Montgomery representation: divisions by 2 for
        conversion from pseudo-inversion to Montgomery representation of
        the inverse are included.
        """
        if n is None:
            n = self.params.algo.n
        return (12*n*self.cnot + 2*n*self.toffoli
                + 2*n*(self.multi_and_deand(3) + self.cnot)
                + 2*n*(self.multi_and_deand(3) + 2*self.cnot)
                + 2*n*(self.multi_and_deand(n+1) + self.cnot)
                + 2*n*(self.comparison(n) + self.cnot)
                + 8*n*n*self.fredkin
                + 2*n*(self.add_ctrl(n) + self.and_deand)  # subtraction
                + 2*n*(self.add_ctrl(n) + self.and_deand)  # additions
                + 2*n*n*self.fredkin  # controlled divisions
                + 2*n*(self.rotate(n+1, 1)+self.modular_reduce(n)+self.cnot)
                )

    def div_mod_std(self, n=None):
        """Out of place clean division, with standard multiplication."""
        if n is None:
            n = self.params.algo.n
        return 2*self.kaliski(n) + 2*self.mul(n) + n*self.cnot

    def div_mod_montgomery(self, n=None, w=None):
        """Out of place modular division, with Montgomery representation."""
        if n is None:
            n = self.params.algo.n
        return (2*self.kaliski(n) + 2*self.montgomery_mul_windowed(n, w)
                + n*self.cnot)

    def div_mod_ctrl_std(self, n=None):
        """Out of place controlled modular division, std multiplication."""
        if n is None:
            n = self.params.algo.n
        return 2*self.kaliski(n) + 2*self.mul(n) + n*self.toffoli

    def div_mod_ctrl_montgomery(self, n=None):
        """Out of place controlled modular division, Montgomery mult."""
        # Compared with division, only CNOTs are changed to Toffolis.
        if n is None:
            n = self.params.algo.n
        return (2*self.kaliski(n) + 2*self.montgomery_mul_basic(n)
                + n*self.toffoli)

    def div_add_mod_montgomery(self, n=None, w=None):
        """Out of place controlled modular division and addition of result."""
        return (2*self.kaliski(n) + 2*self.montgomery_mul_windowed(n, w)
                + self.add_mod(n))

    def clean_windowed_mul_montgomery(self, n=None, w=None):
        """Clean Montgomery multiplication, windowed version."""
        if n is None:
            n = self.params.algo.n
        return 2*self.montgomery_mul_windowed(n, w) + n*self.cnot

    def clean_mul_basic_montgomery(self, n=None):
        """Clean Montgomery multiplication, standard version."""
        if n is None:
            n = self.params.algo.n
        return 2*self.montgomery_mul_basic(n) + n*self.cnot

    def clean_mul(self, n=None):
        """Clean multiplication, standard representation and multiplication."""
        if n is None:
            n = self.params.algo.n
        return 2*self.mul(n) + n*self.cnot

    def square_minus(self, n=None):
        """Squaring and subtraction, using standard multiplication."""
        return 2*self.square(n) + self.add_mod(n)

    def montgomery_square_minus_windowed(self, n=None, w=None):
        """Squaring and subtraction, windowed Montgomery version."""
        return 2*self.montgomery_square_windowed(n, w) + self.add_mod(n)

    def montgomery_square_minus_basic(self, n=None):
        """Squaring and subtraction, standard Montgomery version."""
        return 2*self.montgomery_square_basic(n) + self.add_mod(n)

    def elliptic_curve_semi_classical_ctrl_addition_montgomery(self, n=None):
        """Cost of controlled elliptic curve addition, semi-classical."""
        if n is None:
            n = self.params.algo.n
        return (self.semi_classical_ctrl_add(n) + self.semi_classical_add(n)
                + self.div_mod_ctrl_montgomery(n)
                + self.clean_mul_basic_montgomery(n)
                + self.semi_classical_add(n)
                + self.montgomery_square_minus_basic(n)
                + self.clean_mul_basic_montgomery(n)
                + self.add(n)
                + self.div_mod_ctrl_montgomery(n)
                + self.semi_classical_ctrl_add(n) + self.semi_classical_add(n)
                + self.semi_classical_ctrl_add(n)
                + self.semi_classical_ctrl_neg_mod(n)
                )

    def elliptic_curve_semi_classical_ctrl_addition_std(self, n=None):
        """Elliptic curve controlled addition, semi-classical."""
        if n is None:
            n = self.params.algo.n
        return (self.semi_classical_ctrl_add(n) + self.semi_classical_add(n)
                + self.div_mod_ctrl_std(n)
                + self.clean_mul(n)
                + self.semi_classical_add(n)
                + self.square_minus(n)
                + self.clean_mul(n)
                + self.add(n)
                + self.div_mod_ctrl_std(n)
                + self.semi_classical_ctrl_add(n) + self.semi_classical_add(n)
                + self.semi_classical_ctrl_add(n)
                + self.semi_classical_ctrl_neg_mod(n)
                )

    def elliptic_curve_lookup_addition_montgomery(self, n=None,
                                                  wm=None, we=None):
        """Elliptic curve addition from a lookup table."""
        if n is None:
            n = self.params.algo.n
        if wm is None:
            wm = self.params.algo.wm
        if we is None:
            we = self.params.algo.we
        return (self.semi_classical_ctrl_neg_mod(we-1)  # modulo 2**(we-1)
                + self.look_unlookup(we-1, 2*n)
                + 2*self.semi_classical_ctrl_neg_mod(n)
                + 2*self.add_mod(n)
                + self.div_mod_montgomery(n, wm)
                + self.clean_windowed_mul_montgomery(n, wm)
                + self.look_unlookup(we-1, n) + self.add_mod(n)
                + self.montgomery_square_minus_windowed(n, wm)
                + self.clean_windowed_mul_montgomery(n, wm)
                + self.div_mod_montgomery(n, wm)
                + self.look_unlookup(we-1, 2*n)
                + 2*self.semi_classical_ctrl_neg_mod(n)
                + 2*self.add_mod(n)
                + self.semi_classical_ctrl_neg_mod(we-1)
                + self.semi_classical_neg_mod(n)
                )

    def elliptic_curve_lookup_addition_std(self, n=None, we=None):
        """Elliptic curve addition from a lookup table, std representation."""
        if n is None:
            n = self.params.algo.n
        if we is None:
            we = self.params.algo.we
        return (self.semi_classical_ctrl_neg_mod(we-1)
                + self.look_unlookup(we-1, 2*n)
                + 2*self.semi_classical_ctrl_neg_mod(n)
                + 2*self.add_mod(n)
                + self.div_mod_std(n)
                + self.clean_mul(n)
                + self.look_unlookup(we-1, n) + self.add_mod(n)
                + self.square_minus(n)
                + self.clean_mul(n)
                + self.div_mod_std(n)
                + self.look_unlookup(we-1, 2*n)
                + 2*self.semi_classical_ctrl_neg_mod(n)
                + 2*self.add_mod(n)
                + self.semi_classical_ctrl_neg_mod(we-1)
                + self.semi_classical_neg_mod(n)
                )

    def elliptic_curv_mul_windowed_montgomery(self):
        """Elliptic curve scalar multiplication, windowed, Montgomery repr."""
        ne, we = self.ne, self.params.algo.we
        nb = (ne/we)
        return nb*self.elliptic_curve_lookup_addition_montgomery()

    def elliptic_curv_mul_ctrl_montgomery(self):
        """Elliptic curve controlled scalar multiplication."""
        return self.ne*self.elliptic_curve_semi_classical_ctrl_addition_montgomery()

    def elliptic_curv_mul_windowed_std(self):
        """Elliptic curve scalar multiplication, windowed, standard repr."""
        ne, we = self.ne, self.params.algo.we
        nb = (ne/we)
        return nb*self.elliptic_curve_lookup_addition_std()

    def elliptic_curv_mul_ctrl_std(self):
        """Elliptic curve scalar multiplication, standard."""
        return self.ne*self.elliptic_curve_semi_classical_ctrl_addition_std()

    def elliptic_curv_mul(self):
        """Elliptic curve scalar multiplication, autoselection."""
        if USE_MONTGOMERY:
            if self.params.algo.windowed:
                return self.elliptic_curv_mul_windowed_montgomery()
            return self.elliptic_curv_mul_ctrl_montgomery()
        else:
            if self.params.algo.windowed:
                return self.elliptic_curv_mul_windowed_std()
            return self.elliptic_curv_mul_ctrl_std()

    def elliptic_log_compute(self):
        """Elliptic log discrete logarithm computation."""
        # Fourier transform neglected
        if self.params.algo.algo == 'Shor':
            return self.elliptic_curv_mul()
        elif self.params.algo.algo == 'Ekera':
            s = self.params.algo.s
            nb_repeat = 1 if s == 1 else s+1
            return nb_repeat * self.elliptic_curv_mul()
        else:
            raise ValueError("'self.params.algo.algo' must be 'Shor' or "
                             "'Ekera'!")


class ToffoliBasedCode(ErrCorrCode):
    """Base class for codes able to use directly Toffoli gates.

    Cost of Toffoli gate should be fixed in __init__(), otherwise will fall
    back to AND/DEAND + CNOT.
    """

    @property
    def and_gate(self):
        """Cost of AND computation in an ancillary qubit."""
        # Initialisation neglected as we usually reuse clean qubits.
        return self.toffoli

    @property
    def deand(self):
        """Cost of AND uncomputation."""
        return self.toffoli

    def _set_measure_based_deand(self, *_):
        """If 'measure_based_deand' is None, set the correct value."""
        super()._set_measure_based_deand(False)

    def _check_measure_based_deand(self):
        """Check consistancy of the deand method with the parameter."""
        if self.params.algo.mesure_based_deand:
            raise ValueError("params.algo.mesure_based_deand must be false!")

    @property
    def maj(self):
        """Cost of MAJ operation."""
        nb_cnots = 1 if self.params.algo.parallel_cnots else 2
        return self.toffoli + nb_cnots*self.cnot

    @property
    def maj_dag(self):
        """Cost of MAJ^dagger operation."""
        nb_cnots = 1 if self.params.algo.parallel_cnots else 2
        return self.toffoli + nb_cnots*self.cnot

    @property
    def uma(self):
        """Cost of UMA operation."""
        # No parallelisation here
        return self.toffoli + 2*self.cnot

    @property
    def uma_dag(self):
        """Cost of UMA^dagger operation."""
        # No parallelisation here
        return self.toffoli + 2*self.cnot

    @property
    def uma_ctrl(self):
        """Cost of controlled UMA operation."""
        return 2*self.toffoli + 2*self.cnot

    def add(self, n=None):
        """Cost of addition (with Toffoli)."""
        if n is None:  # coset representation
            n = self.params.algo.n + self.params.algo.c
        return (n - 3)*(self.maj + self.uma) + 7*self.cnot + 3*self.toffoli

    def add_nomod(self, n=None):
        """Cost of addition, with output carry."""
        if n is None:
            n = self.params.algo.n
        return ((n - 2)*(self.maj + self.uma) + 3*self.toffoli + 6*self.cnot
                + self.init)

    def add_ctrl(self, n=None):
        """Cost of controlled addition, modulo power of 2."""
        if n is None:
            n = self.params.algo.n
        return ((n - 2)*(self.maj + self.uma_ctrl) + 2*self.cnot
                + 4*self.toffoli)

    def add_ctrl_nomod(self, n=None):
        """Cost of controlled addition, with output carry."""
        if n is None:
            n = self.params.algo.n
        return (n - 1)*(self.maj + self.uma_ctrl) + 4*self.toffoli


# %% Concrete classes, mainly __init__() only implemented
class AliceAndBob2(ToffoliBasedCode):
    """Cat qubits with repetition code operated with lattice surgery.

    Toffoli magical states are prepared through measurement of its stabilizer.
    Preparation is probabilist; gate is teleported.
    """

    def __init__(self, params: Params, k1_k2=1e-5):
        """Create the object representing the repetition code.

        k1_k2 : ratio k1/k2.
        """
        super().__init__(params)
        err = self._single_qubit_err(params, k1_k2)  # 1 qubit logical error
        log_qubits = logical_qubits(params, verb=False)
        self._set_clifford_costs(err, log_qubits)
        # Toffoli : measurement + teleportation
        teleport = self._teleport()
        factory, factory_qubits, nb_factory = self._toffoli_state_mesurement(
            params, teleport.t)
        self._set_toffoli_teleportation(log_qubits, err,
                                        factory, factory_qubits, nb_factory)

    @staticmethod
    def _single_qubit_err(params: Params, k1_k2):
        """Logical error for 1 qubit by cycle."""
        d, n = params.low_level.d, params.low_level.n
        err = (0.092*(16*sqrt(k1_k2))**((d+1)//2)
               + 2*(d**2)*(5.58*sqrt(k1_k2) + 1.68*k1_k2)*exp(-2*n))
        if err > 1:
            raise RuntimeError("Error formula used outside of its domain!")
        return err

    def _set_clifford_costs(self, err, log_qubits):
        """Set the cost of the Clifford operation.

        err : single logical qubit error.
        time : time for fault-tolerant error correction.
        log_qubits : number of logical qubits.
        """
        d, tc = self.params.low_level.d, self.params.low_level.tc
        time = tc*d  # time for fault-tolerant error correction
        err_2 = 1 - (1 - err)**2
        err_3 = 1 - (1 - err)**3
        # Cost of physical gates (0.25 : time for 1 physical gate)
        self.gate1 = PhysicalCost(err, 0.25*tc + time) + (
            (log_qubits-1) * PhysicalCost(err, 0))
        self.cnot = PhysicalCost(err_2, 0.25*tc + 3*time) + (
            (log_qubits-2) * PhysicalCost(err_3, 0))  # CNOT + merge + split
        # CZ are considered as costly as CNOT (not a lot in the algorithm)
        self.init = PhysicalCost(err, time) + (
            (log_qubits-1)*PhysicalCost(err, 0))
        self.mesure = PhysicalCost(err, self.params.low_level.tr + tc) + (
            (log_qubits-1)*PhysicalCost(err, 0))
        # Processor properties
        self.correct_time = time

    def _teleport(self):
        """Cost of teleportation of Toffoli gate."""
        # Not suited for time optimal computation (no use of reaction time).
        return 3*self.cnot + 1.5*(self.cnot | self.gate1)

    @staticmethod
    def _toffoli_state_mesurement(params: Params, final_time):
        """Cost of measurement-based preparation of Toffoli magical state."""
        t_cnot = 0.25*params.low_level.tc
        if params.low_level.d1 == 0:
            d1 = 3
            err_prob = 5.25e-4
            time = 19 * t_cnot
            accept_prob = 8.4e-1
        elif params.low_level.d1 == 1:
            d1 = 3
            err_prob = 5.12e-5
            time = 25 * t_cnot
            accept_prob = 7.45e-1
        elif params.low_level.d1 == 2:
            d1 = 3
            err_prob = 4.07e-5
            time = 31 * t_cnot
            accept_prob = 6.6e-1
        elif params.low_level.d1 == 3:
            d1 = 5
            err_prob = 2.31e-6
            time = 38 * t_cnot
            accept_prob = 4.56e-1
        elif params.low_level.d1 == 4:
            d1 = 5
            err_prob = 3.5e-7
            time = 45 * t_cnot
            accept_prob = 3.62e-1
        elif params.low_level.d1 == 5:
            d1 = 5
            err_prob = 2.68e-7
            time = 52 * t_cnot
            accept_prob = 2.88e-1
        elif params.low_level.d1 == 6:
            d1 = 7
            err_prob = 3.07e-8
            time = 61 * t_cnot
            accept_prob = 1.48e-1
        elif params.low_level.d1 == 7:
            d1 = 7
            err_prob = 7.00e-9
            time = 69 * t_cnot
            accept_prob = 1.05e-1
        elif params.low_level.d1 == 8:
            d1 = 7
            err_prob = 6.00e-9
            time = 77 * t_cnot
            accept_prob = 7.27e-2
        elif params.low_level.d1 == 9:
            d1 = 9
            err_prob = 1.14e-9
            time = 88 * t_cnot
            accept_prob = 2.62e-2
        elif params.low_level.d1 == 10:
            d1 = 9
            err_prob = 1.15e-10
            time = 97 * t_cnot
            accept_prob = 1.54e-2
        elif params.low_level.d1 == 11:
            d1 = 9
            err_prob = 3.68e-11
            time = 106 * t_cnot
            accept_prob = 9.75e-3
        elif params.low_level.d1 == 12:
            d1 = 19
            err_prob = 3.95e-12
            time = 9665 * t_cnot
            accept_prob = 1
        elif params.low_level.d1 == 13:
            d1 = 21
            err_prob = 2.70e-13
            time = 14239 * t_cnot
            accept_prob = 1
        elif params.low_level.d1 == 14:
            d1 = 23
            err_prob = 1.87e-14
            time = 21531 * t_cnot
            accept_prob = 1
        else:
            raise ValueError("'d1' is here used as an index in range(15).")
        nb_qubits = d1 + 3*d1 + 3*(d1-1)
        nb_factory = ceil(time / (final_time * accept_prob))
        nb_qubits_total = nb_qubits * nb_factory
        return PhysicalCost(err_prob, final_time), nb_qubits_total, nb_factory

    def _set_toffoli_teleportation(self, log_qubits, err,
                                   factory, factory_qubits, nb_factory):
        """Set cost of toffoli gate, and number of qubits.

        log_qubits : number of logical qubits.
        err : error by logical qubit and cycle.
        factory : physical cost of magical state factory.
        factory_qubits : number of qubits in all magical state factories.
        nb_factory : number of factories.
        """
        d = self.params.low_level.d
        teleport = self._teleport()
        self.toffoli = PhysicalCost(factory.p, 0) + teleport + (
            (log_qubits-3) * 4.5 * PhysicalCost(err, 0))
        # Processor properties
        nb_horizon_routing_log_qubits = ceil((log_qubits+nb_factory)/2) + 1
        routing_qubits = ((d + d-1)*(nb_horizon_routing_log_qubits)
                          + 2*(3*(log_qubits + 4*nb_factory
                                  + nb_horizon_routing_log_qubits) - 1))
        # d data qubits ; d-1 ancillary qubits
        self.proc_qubits = ((d + d-1)*log_qubits + factory_qubits
                            + routing_qubits)

    def look_unlookup(self, w=None, n=None):
        """Cost of lookup and unlookup (measurement-based or not)."""
        if self.params.algo.mesure_based_unlookup:
            return super().look_unlookup(w=w, n=n)
        if w is None and n is None:
            w, n = self._defaul_lookup_sizes()
        return 2*self.lookup(w, n)

    def _check_mesure_based_unlookup(self):
        """Check consistency of the deand method with the parameter."""
        pass


class NoCorrCode(ToffoliBasedCode):
    """No error correction."""

    def __init__(self, params: Params):
        """Init no correction instance."""
        super().__init__(params)
        err_2 = 1 - (1 - params.low_level.pp)**2
        err_3 = 1 - (1 - params.low_level.pp)**3
        self.gate1 = PhysicalCost(params.low_level.pp, params.low_level.tc)
        self.cnot = PhysicalCost(err_2, params.low_level.tc)
        self.toffoli = PhysicalCost(err_3, params.low_level.tc)
        self.init = PhysicalCost(params.low_level.pp, params.low_level.tc)
        self.mesure = PhysicalCost(params.low_level.pp, params.low_level.tr)
        self.correct_time = float('nan')
        self.proc_qubits = logical_qubits(params, verb=False)


# %% Ancillary functions related with the algorithm
def logical_qubits_exp_mod(params: Params, verb=True):
    """Logical qubits number for modular exponentiation."""
    toffoli_based = params.type in ('alice&bob2', None)
    if toffoli_based and params.algo.windowed:
        res = (3*params.algo.n + 2*params.algo.c + 2*params.algo.we
               + params.algo.wm - 1)
    elif (not toffoli_based) and params.algo.windowed:
        res = 4*params.algo.n + 3*params.algo.c + params.algo.we - 1
    elif not params.algo.windowed:
        res = 3*(params.algo.n + params.algo.c) + 1
    if verb:
        print("Total number of logical qubits:", res)
    return res


def logical_qubits_elliptic_mul(params: Params, verb=True):
    """Logical qubits number for elliptic curve multiplication."""
    # ancilla_free_add = params.type in ('alice&bob2', None)
    ancilla_free_add = False
    if params.algo.windowed:
        if ancilla_free_add:
            res = 8*params.algo.n + params.algo.we + 6
        else:
            res = 9*params.algo.n + params.algo.we + 4
    else:
        if ancilla_free_add:
            res = 8*params.algo.n + 7
        else:
            res = 9*params.algo.n + 6
    if verb:
        print("Total number of logical qubits:", res)
    return res


def logical_qubits(params: Params, verb=True):
    """Logical qubits number.

    Autoselection of the problem depending on parameters.
    """
    if params.algo.prob == 'rsa':
        return logical_qubits_exp_mod(params, verb)
    elif params.algo.prob == 'elliptic_log':
        return logical_qubits_elliptic_mul(params, verb)
    else:
        raise ValueError("Unknown problem type !")
