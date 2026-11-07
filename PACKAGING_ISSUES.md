# Packaging issues observed on Roar

Local correction: core and sampling imports now use `pydoe`, declared as a
mandatory dependency (`>=1.6,<2`) for the sampling and blocked/staged optimizers.
Public package exports are now lazy so the cobalt CLI does not import unrelated
visualization/uncertainty dependencies. The sampling extra remains compatible with existing
installation commands. Package discovery now includes all `pibo_reaxff` subpackages.
Roar diagnostics confirmed pydoe 1.6.0 provides `pydoe` but not `pyDOE`.

## Cobalt CLI fails importing pyDOE

Observed during the Python 3.11 Roar installation: both the base editable install
and a subsequent editable install with the `sampling` extra passed `pip check`,
but running `pibo_calculation/cobalt_pibo.py --help` failed with
`ModuleNotFoundError: No module named 'pyDOE'`.

The import chain is `pibo_reaxff.__init__` -> `core.blocked_pibo` ->
`from pyDOE import lhs`. This eager import makes sampling a requirement even
for entry points that do not need the blocked optimizer, while `pyproject.toml`
declares `pyDOE>=0.3.8` only as an optional sampling extra. Installing the extra
did not resolve the observed failure; the installed distribution version and
its actual import namespace must also be checked. A clean `pip check` validates
declared distribution dependencies, not Python import names or runtime imports.

Correction follow-up:

- Inspect the installed pyDOE distribution version, files and import namespace.
- Decide whether blocked-optimizer imports should be lazy or their dependencies
  should be mandatory; keep the cobalt CLI independently importable where possible.
- Align sampling imports with the supported distribution versions and declare
  tested version constraints rather than relying on the unbounded lower limit.
- Verify package discovery includes required subpackages in a non-editable wheel.
- Add clean-environment base/sampling installation and cobalt `--help` smoke tests.
- Recheck the actual cobalt/LAMMPS path separately; successful imports do not
  establish engine compatibility or scientific correctness.

No scientific results, datasets, credentials or cluster logs are included here.
