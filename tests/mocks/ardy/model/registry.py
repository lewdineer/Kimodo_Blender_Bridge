MODELS = {"core": "ARDY-Core-RP-20FPS-Horizon40", "core8": "ARDY-Core-RP-20FPS-Horizon8"}
def resolve_model_name(name, default_family=None, checkpoints_dir=None):
    if name in MODELS:
        return MODELS[name]
    if name in MODELS.values():
        return name
    raise ValueError(f"Unknown model {name!r}. Choose a nickname {list(MODELS)}.")
