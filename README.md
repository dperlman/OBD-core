# OBD-core

The core computational functions for the ordered binomial distribution, shared by
[ordered-binomial-cusps](https://github.com/dperlman/ordered-binomial-cusps) and
[OBDExplorer](https://github.com/dperlman/OBDExplorer).  Everything is in one module,
`obd_core.py`; its docstring holds the definitions, the method and the numerical safeguards.

## Install

```bash
pip install -c https://raw.githubusercontent.com/dperlman/OBD-core/v0.3.3/constraints.txt \
    "obd-core @ git+https://github.com/dperlman/OBD-core.git@v0.3.3"
```

For development, clone it and install editable: `pip install -c constraints.txt -e path/to/OBD-core`.

## Use

```python
import obd_core as core

p, E, F3, S_minus, S_plus, slope_left, slope_right = core.evaluate(100, 45, 56)
```

`core.tie_table(n, both_halves=True)` returns every tie point of `n`, sorted by p*, with the
certified cusp verdict, the exact one-sided slopes and `log10_D`, the log of the slope jump.
`core.E_slopes_at(n, p_array)` gives E and the exact one-sided slopes at arbitrary p (for grids).

Dependencies: numpy, numba, mpmath.  Tests: `pip install -e ".[test]"` then `pytest`.

## One numerical stack

Every environment that runs `obd_core` should have exactly the versions in
[`constraints.txt`](constraints.txt) (numpy, numba, llvmlite, mpmath), so that results agree to
the last bit across environments and numba's compiled-code cache, which is per numba version,
stays valid.  Install with the constraints file, as above.
Importing `obd_core` in an environment that differs warns (`obd_core.pin_mismatches()` lists
the differences).  The pins live in `obd_core.NUMERIC_PINS`; `constraints.txt` is generated from it.
