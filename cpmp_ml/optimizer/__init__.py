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
        # Importa el módulo y obtiene el atributo
        module = __import__(module_name, fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    """Define los nombres accesibles en el paquete."""
    return list(globals().keys()) + list(_importables.keys())
