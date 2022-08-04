#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute resources required for Shor's algorithms (discrete log or factorize).

@author: Élie Gouzien
"""
from math import isnan, isinf
from itertools import product
from functools import reduce
import warnings
from warnings import warn

from tqdm import tqdm

from tools import AlgoOpts, LowLevelOpts, Params, PhysicalCost
from error_correction import ErrCorrCode, logical_qubits
import error_correction


warnings.simplefilter("always", UserWarning)

# Use only TQDM when executed within IPython
try:
    if not __IPYTHON__:
        raise NameError("Artifice")
    IPYTHON = True
except NameError:
    IPYTHON = False
PB_DEF = True  # Default behaviour for progress bars.


# %% Optimisation
# Default parameters (different for each architecture).
DEF_RANGES = {'alice&bob2': dict(d1s=range(15),  # only indexes
                                 d2s=(None,),
                                 ds=range(1, 28, 2),
                                 # cat average photon number, continuous!
                                 ns=range(1, 25, 1),
                                 wes=range(1, 30),
                                 wms=range(1, 30),
                                 cs=range(1, 40)),
              None: dict(d1s=(None,),
                         d2s=(None,),
                         ds=(None,),
                         ns=(None,),
                         wes=range(1, 10),
                         wms=range(1, 10),
                         cs=range(1, 40))}


def _calc_ranges(base_params: Params, **kwargs):
    """Compute iteration ranges for the optimisation.

    kwargs possibles : d1s, d2s, ds, ns, wes, wms, cs
    """
    ranges = DEF_RANGES[base_params.type].copy()
    ranges.update(kwargs)
    if (base_params.type == 'alice&bob2'
            and base_params.algo.prob == 'elliptic_log'):
        if ranges['cs'] != DEF_RANGES[base_params.type]['cs']:
            warn('No coset representation for elliptic_log ; removing the '
                 'exploration !')
        ranges['cs'] = (None,)
    if not base_params.algo.windowed:
        if (ranges['wes'] not in (DEF_RANGES[base_params.type]['wes'], (None,))
                or ranges['wms'] not in (DEF_RANGES[base_params.type]['wms'],
                                         (None,))):
            warn("'wes' or 'wms' range explicitly given while non windowed "
                 "arithmetic circuit used. Removing them!")
        ranges['wes'] = (None,)
        ranges['wms'] = (None,)
    if (base_params.algo.prob == 'elliptic_log'
            and not error_correction.USE_MONTGOMERY):  # magouille temporaire
        if ranges['wms'] not in (DEF_RANGES[base_params.type]['wms'], (None,)):
            warn("Without Montgomery representation 'wm' will not be used; "
                 "removing it, make sure why you set it!")
        ranges['wms'] = (None,)
    return ranges


def iterate(base_params: Params, progress=PB_DEF, **kwargs):
    """Generate iterator on all free parameters of the algorithm.

    progress : show a progress bar?
    Possible kwargs: d1s, d2s, ds, ns, wes, wms, cs
    """
    # pylint: disable=C0103
    ranges = _calc_ranges(base_params, **kwargs)
    iterator = product(ranges['d1s'], ranges['d2s'], ranges['ds'],
                       ranges['ns'], ranges['wes'], ranges['wms'],
                       ranges['cs'])
    if progress:
        nb_iter = reduce(lambda x, y: x*len(y), ranges.values(), 1)
        iterator = tqdm(iterator, total=nb_iter, dynamic_ncols=True)
    for d1, d2, d, n, we, wm, c in iterator:
        if (base_params.algo.prob == 'rsa'  # we and wm have same role.
                and wm is not None and we is not None and wm > we):
            continue
        yield base_params._replace(
            algo=base_params.algo._replace(we=we, wm=wm, c=c),
            low_level=base_params.low_level._replace(d1=d1, d2=d2, d=d, n=n))


def metrique(cost: PhysicalCost, qubits, params: Params, biais=1):
    """Score the quality of resource cost."""
    n = params.low_level.n or 1  # pylint: disable=C0103
    return cost.exp_t * qubits**biais * n


def prepare_ressources(params: Params):
    """Prepare cost and qubit count for given parameter set."""
    err_corr = ErrCorrCode(params)
    if params.algo.prob == 'rsa':
        cost = err_corr.factorisation()
    elif params.algo.prob == 'elliptic_log':
        cost = err_corr.elliptic_log_compute()
    else:
        raise ValueError("params.prob must be 'rsa' or 'elliptic_log'!")
    qubits = err_corr.proc_qubits
    return cost, qubits


def find_best_params(base_params: Params, biais=1, progress=PB_DEF, **kwargs):
    """Find the best parameter set."""
    best = float('inf')
    best_params = None
    for params in iterate(base_params, progress=progress, **kwargs):
        try:
            cost, qubits = prepare_ressources(params)
        except RuntimeError as err:
            if isinstance(err, NotImplementedError):
                raise
            continue
        score = metrique(cost, qubits, params, biais)
        if score < best:
            best = score
            best_params = params
    if best_params is None:
        raise RuntimeError("Optimization didn't converge. "
                           "No parameter allow to end the computation in "
                           "finite time.")
    # Parameters bounds detection.
    ranges = _calc_ranges(base_params, **kwargs)
    for var_name, var_type in [('d1s', 'low_level'),
                               ('d2s', 'low_level'),
                               ('ds', 'low_level'),
                               ('ns', 'low_level'),
                               ('wes', 'algo'),
                               ('wms', 'algo'),
                               ('cs', 'algo')]:
        var_range = ranges[var_name]
        var_val = getattr(getattr(best_params, var_type), var_name[:-1])
        if (var_range != (None,) and (var_val == min(var_range)
                                      or var_val == max(var_range))):
            warn(f"Params : {best_params} ; "
                 f"Variable '{var_name[:-1 ]}={var_val}' reached one of its "
                 "extremities!")
    return best_params


# %% Tables
def unit_format(num, unit, unicode=False):
    """Assemble number and unit, eventually converting it into LaTeX."""
    space = chr(8239)
    num = str(round(num)) if not isinf(num) else "∞" if unicode else r"\infty"
    if not unicode:
        unit = {"µs": r"\micro\second",
                "ms": r"\milli\second",
                "s": r"\second",
                "min": r"\minute",
                "hours": "hours",
                "days": "days"}[unit]
    if unicode:
        return num + space + unit
    return rf"\SI{{{num}}}{{{unit}}}"


def format_time(time, unicode=False):
    """Return formatted time, with correct unity."""
    if time is None:
        return repr(None)
    if isnan(time):
        return "nan"
    if time < 1e-3:
        temps, unit = time*1e6, "µs"
    elif time < 1:
        temps, unit = time*1000, "ms"
    elif time < 60:
        temps, unit = time, "s"
    elif time < 3600:
        temps, unit = time/60, "min"
    elif time < 3600*24:
        temps, unit = time/3600, "hours"
    else:
        temps, unit = time/(3600*24), "days"
    return unit_format(temps, unit, unicode)


def entree_tableau_rsa_alicebob2(params: Params):
    """Give a table line, for RSA and alice&bob2."""
    err_corr = ErrCorrCode(params)
    cost, qubits = prepare_ressources(params)
    return [params.algo.n, err_corr.ne, params.algo.c, params.algo.we,
            params.algo.wm, params.low_level.n, params.low_level.d,
            params.low_level.d1, qubits, format_time(cost.t),
            format_time(cost.exp_t), logical_qubits(params, False)]


def entree_tableau_elliptic_log2(params: Params):
    """Give a table line, for elliptic curve logarithm and Alice&Bob2."""
    err_corr = ErrCorrCode(params)
    cost, qubits = prepare_ressources(params)
    return [params.algo.n, err_corr.ne, params.algo.we, params.algo.wm,
            params.low_level.n, params.low_level.d, params.low_level.d1,
            qubits, format_time(cost.t), format_time(cost.exp_t),
            logical_qubits(params, False)]


def table_shape(largeurs, sep_places, sep="|"):
    """Give the shape of the table (for LaTeX)."""
    liste = [f"S[table-figures-integer={size}]" if size is not None else 'c'
             for size in largeurs]
    for pos in sorted(sep_places, reverse=True):
        liste = liste[:pos] + [sep] + liste[pos:]
    return ''.join(liste)


def _print_tableau(n_values, biais_values, base_params: Params, entree_func,
                   entetes, skip_size, seps, just=30):
    """Compute and display the table."""
    tableau = []
    for n, biais in zip(n_values, biais_values):  # pylint: disable=C0103
        best_params = find_best_params(base_params._replace(
            algo=base_params.algo._replace(n=n)), biais=biais)
        tableau.append(entree_func(best_params))
    # Compute width for each column
    sizes = [max(len(str(ligne[col])) for ligne in tableau)
             if col not in skip_size else None for col in range(len(entetes))]
    # Print the table
    print(r"\begin{tabular}{" + table_shape(sizes, seps) + "}")
    print("\t" + '&'.join(('{'+x+'}').ljust(just) for x in entetes) + r'\\',
          r"\hline")
    for ligne in tableau:
        print("\t" + '&'.join(str(x).ljust(just) for x in ligne) + r'\\')
    print(r'\end{tabular}')


def _print_tableau_rsa_alicebob2(base_params: Params):
    """Table for article, for RSA and alice&bob2."""
    if base_params.type != 'alice&bob2' or base_params.algo.prob != 'rsa':
        warn("Warning, parameters not compatible with the table; "
             "columns might not correspond to what is expected!")
    entetes = ["$n$", '$n_e$', "$m$", "$w_e$", "$w_m$", r"$\abs{\alpha}^2$",
               "$d$", "$d_1$", r"$n_{\text{qubits}}$", "$t$",
               r"$t_{\text{exp}}$", "logical qubits"]
    _print_tableau([6, 8, 16, 128, 256, 512, 829, 1024, 2048],
                   [10] + [1]*8, base_params,
                   entree_tableau_rsa_alicebob2,
                   entetes, skip_size=(9, 10), seps=(2, 4))


def _print_tableau_elliptic_log2(base_params: Params):
    """Table for article, for ECDL and alice&bob2."""
    if (base_params.type != 'alice&bob2'
            or base_params.algo.prob != 'elliptic_log'):
        warn("Warning, parameters not compatible with the table; "
             "columns might not correspond to what is expected!")
    warn("Warning, for small n values, magical state preparation parameters "
         "might not be adapted (too much precision)!")
    entetes = ["$n$", '$n_e$', "$w_e$", "$w_m$", r"$\abs{\alpha}^2$", "$d$",
               "$d_1$", r"$n_{\text{qubits}}$", "$t$", r"$t_{\text{exp}}$",
               "logical qubits"]
    _print_tableau([8, 16, 32, 64, 128, 256, 512], [1]*7, base_params,
                   entree_tableau_elliptic_log2, entetes, skip_size=(9, 10),
                   seps=(2, 8))


def print_tableau(base_params: Params):
    r"""Table for article supplemental material, autoselection of table type.

    To be used with
    \usepackage[table-figures-decimal=0,table-number-alignment=center]{siunitx}
    Incompatible with quantikz
    """
    if base_params.type == 'alice&bob2' and base_params.algo.prob == 'rsa':
        _print_tableau_rsa_alicebob2(base_params)
    elif (base_params.type == 'alice&bob2'
          and base_params.algo.prob == 'elliptic_log'):
        _print_tableau_elliptic_log2(base_params)
    else:
        raise ValueError("No table for those parameters.")


# %% Executable part
if __name__ == '__main__':
    params = Params('alice&bob2',
                    AlgoOpts(prob='elliptic_log', algo='Shor', s=None, n=256,
                             windowed=True, parallel_cnots=True),
                    LowLevelOpts())

    # params = Params('alice&bob2',
    #                 AlgoOpts(n=2048, windowed=True, parallel_cnots=True),
    #                 LowLevelOpts())

    print("\n"*2)
    print("Windowed arithmetics")
    print("====================")
    best_params = find_best_params(params, biais=1)
    best_err_corr = ErrCorrCode(best_params)
    best_cost, best_qubits = prepare_ressources(best_params)
    print("Best case:", best_cost, ";",  best_qubits)

    print("\n"*2)
    print("Controlled arithmetics")
    print("======================")
    best_params_basic = find_best_params(
        params._replace(algo=params.algo._replace(windowed=False)), biais=1)
    best_err_corr_basic = ErrCorrCode(best_params_basic)
    best_cost_basic, best_qubits_basic = prepare_ressources(best_params_basic)
    print("Best controlled case:", best_cost_basic, ";", best_qubits_basic)

    # Table
    print("\n"*2)
    print("Table")
    print("=======")
    print_tableau(params)
