"""Load ``jaxrl2/agents/pixel_fpo/fpo_math.py`` by path.

``fpo_math`` only needs NumPy; importing it through the ``jaxrl2`` package would
pull in JAX/Flax via ``jaxrl2/agents/__init__.py``.
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_fpo_math():
    path = ROOT / 'jaxrl2' / 'agents' / 'pixel_fpo' / 'fpo_math.py'
    spec = importlib.util.spec_from_file_location('fpo_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
