Code for computations in [arXiv:2302.06639](https://arxiv.org/abs/2302.06639)
=============================================================================


Computation of resources for elliptic curve discrete logarithm computation with cat qubits
------------------------------------------------------------------------------------------

Python files for evaluating the cost of elliptic curve discrete logarithm computation with a processor made from cat qubits and operated with lattice surgery.

Manifest:
  * `tools.py` : definition of useful data structures.
  * `error_correction.py` : representation of error correction, and the cost evaluation for circuits.
  * `cout_shor.py` : main file doing the evaluation by optimizing on the parameters.
