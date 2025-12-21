# Energy-Based JEPA package
import importlib
import os
import pkgutil


# Automatically import all modules in this package
def _import_all_modules():
    """Automatically import all Python modules in this package."""
    modules = []
    package_dir = os.path.dirname(__file__)

    for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
        if not is_pkg and not name.startswith("_"):
            try:
                module = importlib.import_module(f".{name}", __package__)
                modules.append(name)
                globals()[name] = module
            except ImportError as e:
                print(f"Warning: Could not import {name}: {e}")

    return modules


# Import all modules and set __all__
__all__ = _import_all_modules()

__version__ = "0.1.0"
