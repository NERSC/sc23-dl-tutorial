
def get_data_loader_distributed(params, location, distributed, train):
    if params.data_loader_config.startswith("dali"):
        from .data_loader_dali import get_data_loader
    else:
        from .data_loader import get_data_loader
    return get_data_loader(params, location, distributed, train)

