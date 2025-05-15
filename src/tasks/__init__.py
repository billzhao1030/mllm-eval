from .base_dataset import MetaDataset

from .r2r import R2RDataset
from .rxr import RXRDataset
from .cvdn import CVDNDataset
from .reverie import REVERIEDataset

__all__ = list(MetaDataset.registry.keys())

def load_dataset(name, *args, **kwargs):
    cls = MetaDataset.registry[name]
    return cls(*args, **kwargs)