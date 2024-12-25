__all__ = ["OptimizerStrategy", "GreedyV1", "GreedyV2", "GreedyModel"]

_importables = {
    "OptimizerStrategy": "cpmp_ml.optimizer",
    "GreedyV1": "cpmp_ml.optimizer",
    "GreedyV2": "cpmp_ml.optimizer",
    "GreedyModel": "cpmp_ml.optimizer",
}

# Marcador para evitar recursión infinita
_imported_modules = {}

def __getattr__(name):
    """Carga los objetos bajo demanda."""
    if name in _importables:
        module_name = _importables[name]
        # Verificar si ya importamos este módulo
        if module_name not in _imported_modules:
            module = __import__(module_name, fromlist=[name])
            _imported_modules[module_name] = module
        else:
            module = _imported_modules[module_name]
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    """Define los nombres accesibles en el paquete."""
    return list(globals().keys()) + list(_importables.keys())
