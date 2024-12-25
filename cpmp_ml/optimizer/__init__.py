import importlib

__all__ = ["OptimizerStrategy", "GreedyV1", "GreedyV2", "GreedyModel"]

_importables = {
    "OptimizerStrategy": "cpmp_ml.optimizer",
    "GreedyV1": "cpmp_ml.optimizer",
    "GreedyV2": "cpmp_ml.optimizer",
    "GreedyModel": "cpmp_ml.optimizer",
}

def __getattr__(name):
    """Carga los objetos bajo demanda."""
    if name in _importables:
        module_name = _importables[name]
        module = importlib.import_module(module_name)  # Importa el módulo usando importlib
        return getattr(module, name)  # Obtiene el atributo del módulo
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    """Define los nombres accesibles en el paquete."""
    return list(globals().keys()) + list(_importables.keys())

