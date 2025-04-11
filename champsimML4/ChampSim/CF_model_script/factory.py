from .cruisefetch_pro import CruiseFetchPro

def create_model_with_config(config_path=None):
    """Factory function to create a model with specific configuration"""
    return CruiseFetchPro(config_path)
