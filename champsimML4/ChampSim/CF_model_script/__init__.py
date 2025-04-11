from .base_model import MLPrefetchModel
from .cruisefetch_pro import CruiseFetchPro
from .factory import create_model_with_config

# For backward compatibility
Model = CruiseFetchPro
