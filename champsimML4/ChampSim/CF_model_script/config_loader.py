import os
import yaml

def load_config(config_path):
    """Load configuration from a YAML file"""
    try:
        print(f"Attempting to load configuration from: {config_path}")
        
        # Convert to absolute path if not already
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            print(f"Using absolute path: {config_path}")
            
        if not os.path.exists(config_path):
            print(f"No configuration file found at: {config_path}")
            print(f"Current working directory: {os.getcwd()}")
            print("Using default configuration")
            return {}  # Return empty config if file not found
                
        print(f"Found configuration file at: {config_path}")
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            
        if not config or not isinstance(config, dict) or 'model' not in config:
            print(f"Invalid configuration format in {config_path}")
            return {}
                
        print(f"Successfully loaded configuration from {config_path}")
        model_config = config.get('model', {})
        print("Configuration parameters:")
        for key, value in model_config.items():
            print(f"  - {key}: {value}")
        return model_config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {}
