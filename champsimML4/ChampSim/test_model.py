#!/usr/bin/env python3
"""
Test script for CruiseFetchLITE model to verify the fix for the reshape error
"""
import tensorflow as tf
import numpy as np
from model import CruiseFetchLITEModel

def test_model_creation():
    """Test if the model can be created without errors"""
    print("Testing CruiseFetchLITE model creation...")
    
    # Create model instance
    model = CruiseFetchLITEModel()
    
    # Create TensorFlow model
    tf_model = model.create_tf_model()
    
    if tf_model is not None:
        print("✓ Model created successfully!")
        
        # Print model summary
        print("\nModel Summary:")
        tf_model.summary()
        
        # Test with dummy data
        print("\nTesting model with dummy data...")
        batch_size = 2
        history_length = model.config['history_length']
        
        # Create dummy inputs
        cluster_history = np.random.randint(0, model.config['num_clusters'], size=(batch_size, history_length))
        offset_history = np.random.randint(0, model.config['offset_size'], size=(batch_size, history_length))
        pc_input = np.random.randint(0, model.config['num_pcs'], size=(batch_size, 1))
        dpf_input = np.random.random((batch_size, model.config['dpf_history_length'], model.config['num_candidates']))
        
        # Test forward pass
        try:
            outputs = tf_model.predict([cluster_history, offset_history, pc_input, dpf_input], verbose=1)
            print("✓ Forward pass successful!")
            
            # Print output shapes
            print(f"Candidate logits shape: {outputs[0].shape}")
            print(f"Offset logits shape: {outputs[1].shape}")
            
            return True
        except Exception as e:
            print(f"✗ Forward pass failed with error: {e}")
            return False
    else:
        print("✗ Model creation failed!")
        return False

if __name__ == "__main__":
    test_model_creation()
