#!/usr/bin/env python3
"""
Fixed version of the CruiseFetchLITE model with corrected reshape operations
"""
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import os
import pickle
import lzma
from typing import List, Tuple, Dict, Any, Optional

class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass


class FixedCruiseFetchLITEModel(MLPrefetchModel):
    """
    Fixed CruiseFetchLITE Model adapted for ChampSim ML competition
    
    This model integrates the TLITE neural prefetcher with behavioral clustering
    into the competition framework.
    """
    
    def __init__(self):
        super().__init__()
        # Initialize configuration with default values
        self.config = self.get_default_config()
        
        # Initialize TensorFlow model
        self.model = None
        
        # Initialize state variables
        self.clustering_info = {}
        self.page_history = {}
        self.offset_history = {}
        self.last_pc = {}
        self.metadata_manager = None
        
        # Stream management (handle multiple streams by PC)
        self.num_streams = 16
        self.stream_map = {}  # Maps PC -> stream_id
        
        # Statistics tracking
        self.stats = {
            'accesses': 0,
            'prefetches_issued': 0,
            'prefetches_per_instr': {},
        }
    
    def get_default_config(self):
        """Return default configuration for the model"""
        config = {
            # Model architecture parameters
            'pc_embed_size': 32,        # Reduced from 64 for faster inference
            'cluster_embed_size': 16,   # Reduced from 25 for faster inference
            'offset_embed_size': 80,    # cluster_embed_size * num_experts
            'num_experts': 5,           # Reduced from 100 for faster inference
            'history_length': 3,        # Track 3 previous accesses
            'num_pcs': 1024,            # Number of unique PCs to track
            'num_clusters': 512,        # Number of behavioral clusters
            'offset_size': 64,          # Page offsets (use 6 bits)
            'num_candidates': 2,        # Number of candidate pages to consider
            'dpf_history_length': 1,    # Length of DPF vector history
            'offset_bits': 6,           # Number of bits for page offset
            
            # Prefetcher parameters
            'max_prefetches_per_id': 2  # Maximum prefetches per instruction ID
        }
        return config
    
    def load(self, path):
        """Load model from the given path"""
        print(f"Loading CruiseFetchLITE model from {path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Load metadata
            with open(f"{path}_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.config = metadata['config']
                self.clustering_info = metadata['clustering_info']
                self.stream_map = metadata.get('stream_map', {})
                self.stats = metadata.get('stats', self.stats)
            
            # Load TensorFlow model if it exists
            if os.path.exists(f"{path}_model"):
                self.model = self.create_tf_model()
                self.model.load_weights(f"{path}_model").expect_partial()
                print("Successfully loaded TensorFlow model")
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize new model if loading fails
            self.model = self.create_tf_model()
            return False
    
    def save(self, path):
        """Save model to the given path"""
        print(f"Saving CruiseFetchLITE model to {path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save metadata
            metadata = {
                'config': self.config,
                'clustering_info': self.clustering_info,
                'stream_map': self.stream_map,
                'stats': self.stats
            }
            with open(f"{path}_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            # Save TensorFlow model if it exists
            if self.model is not None:
                self.model.save_weights(f"{path}_model")
                print("Successfully saved TensorFlow model")
                
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def create_tf_model(self):
        """Create the TensorFlow model from configuration"""
        try:
            # Define model using Functional API
            # Inputs
            cluster_history_input = tf.keras.layers.Input(shape=(self.config['history_length'],), name='cluster_history', dtype=tf.int32)
            offset_history_input = tf.keras.layers.Input(shape=(self.config['history_length'],), name='offset_history', dtype=tf.int32)
            pc_input = tf.keras.layers.Input(shape=(1,), name='pc', dtype=tf.int32)
            dpf_input = tf.keras.layers.Input(shape=(self.config['dpf_history_length'], self.config['num_candidates']), name='dpf', dtype=tf.float32)
            
            # Embedding layers
            cluster_embedding = tf.keras.layers.Embedding(
                self.config['num_clusters'], 
                self.config['cluster_embed_size'],
                embeddings_regularizer='l1',
                name='cluster_embedding'
            )(cluster_history_input)
            
            offset_embedding = tf.keras.layers.Embedding(
                self.config['offset_size'],
                self.config['offset_embed_size'],
                embeddings_regularizer='l1',
                name='offset_embedding'
            )(offset_history_input)
            
            pc_embedding = tf.keras.layers.Embedding(
                self.config['num_pcs'], 
                self.config['pc_embed_size'],
                embeddings_regularizer='l1',
                name='pc_embedding'
            )(pc_input)
            
            # Reshape and format embeddings
            batch_size = tf.shape(cluster_history_input)[0]
            pc_embedding_flat = tf.reshape(pc_embedding, [batch_size, self.config['pc_embed_size']])
            
            # FIXED: Reshape offset embedding for expert mixing
            # Changed from 4D tensor to 3D tensor to fix dimension mismatch
            offset_embedding_reshaped = tf.reshape(
                offset_embedding, 
                [batch_size, self.config['history_length'] * self.config['num_experts'], self.config['cluster_embed_size']]
            )
            
            # FIXED: Simplified attention mechanism using first cluster embedding as query
            # Changed from 4D tensor to 3D tensor to match attention layer expectations
            query = tf.reshape(
                cluster_embedding[:, 0:1, :],  # Use first cluster embedding as query 
                [batch_size, 1, self.config['cluster_embed_size']]
            )
            
            # Apply attention
            context_offset = tf.keras.layers.Attention(name='context_attention')(
                [query, offset_embedding_reshaped]
            )
            context_offset = tf.reshape(context_offset, [batch_size, self.config['cluster_embed_size']])
            
            # Flatten cluster embedding
            cluster_flat = tf.reshape(cluster_embedding, [batch_size, self.config['history_length'] * self.config['cluster_embed_size']])
            
            # Flatten DPF vectors
            dpf_flat = tf.reshape(dpf_input, [batch_size, self.config['dpf_history_length'] * self.config['num_candidates']])
            
            # Concatenate all features
            combined = tf.keras.layers.Concatenate(name='combined_features')(
                [pc_embedding_flat, cluster_flat, context_offset, dpf_flat]
            )
            
            # Candidate and offset prediction heads
            candidate_logits = tf.keras.layers.Dense(
                self.config['num_candidates'] + 1,  # Add one for "no prefetch" option
                activation=None,
                kernel_regularizer='l1',
                name='candidate_output'
            )(combined)
            
            offset_logits = tf.keras.layers.Dense(
                self.config['offset_size'],
                activation=None,
                kernel_regularizer='l1',
                name='offset_output'
            )(combined)
            
            # Create model
            model = tf.keras.Model(
                inputs=[cluster_history_input, offset_history_input, pc_input, dpf_input],
                outputs=[candidate_logits, offset_logits],
                name='CruiseFetchLITE'
            )
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=[
                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                ]
            )
            
            return model
        except Exception as e:
            print(f"Error creating TensorFlow model: {e}")
            return None
    
    def process_trace_entry(self, instr_id, cycle_count, load_addr, load_ip, llc_hit):
        """Process a single memory access trace entry"""
        # Extract page address (cache line aligned) and offset
        cache_line_addr = load_addr >> 6  # 64-byte cache line
        page_id = cache_line_addr >> self.config['offset_bits']
        offset = cache_line_addr & ((1 << self.config['offset_bits']) - 1)
        
        # Determine stream ID based on load_ip
        stream_id = self.get_stream_id(load_ip)
        
        # Initialize stream history if needed
        if stream_id not in self.page_history:
            self.page_history[stream_id] = [0] * self.config['history_length']
            self.offset_history[stream_id] = [0] * self.config['history_length']
            self.last_pc[stream_id] = 0
        
        # Update stream history
        self.page_history[stream_id] = self.page_history[stream_id][1:] + [page_id]
        self.offset_history[stream_id] = self.offset_history[stream_id][1:] + [offset]
        self.last_pc[stream_id] = load_ip
        
        # Update clustering information
        self.update_clustering_info(page_id, offset)
        
        # Update metadata
        if self.metadata_manager is not None:
            prev_page = self.page_history[stream_id][-2] if len(self.page_history[stream_id]) > 1 else 0
            prev_offset = self.offset_history[stream_id][-2] if len(self.offset_history[stream_id]) > 1 else 0
            if prev_page != 0:
                self.metadata_manager.update_page_access(prev_page, page_id, prev_offset, offset)
        
        # Update statistics
        self.stats['accesses'] += 1
    
    def update_clustering_info(self, page_id, offset):
        """Update clustering information for a page"""
        # Simple clustering approach: use page_id % num_clusters as cluster ID
        if page_id not in self.clustering_info:
            # Check if we have enough data to do behavioral clustering
            if hasattr(self, 'page_transitions') and page_id in self.page_transitions:
                # TODO: Implement proper behavioral clustering
                # For now, just use a simple hash-based assignment
                cluster_id = hash(page_id) % self.config['num_clusters']
            else:
                cluster_id = page_id % self.config['num_clusters']
            
            self.clustering_info[page_id] = cluster_id
    
    def get_stream_id(self, pc):
        """Get stream ID for a PC value"""
        if pc not in self.stream_map:
            # Assign a stream ID based on PC
            stream_id = hash(pc) % self.num_streams
            self.stream_map[pc] = stream_id
        
        return self.stream_map[pc]
    
    def get_cluster_id(self, page_id):
        """Get cluster ID for a page ID"""
        if page_id in self.clustering_info:
            return self.clustering_info[page_id]
        else:
            # Assign new cluster ID
            cluster_id = page_id % self.config['num_clusters']
            self.clustering_info[page_id] = cluster_id
            return cluster_id
    
    def prepare_model_inputs(self, stream_id):
        """Prepare inputs for the model"""
        # Convert page IDs to cluster IDs
        cluster_history = [self.get_cluster_id(page) for page in self.page_history[stream_id]]
        
        # Create dummy DPF vector if metadata manager is not available
        if self.metadata_manager is None:
            dpf_vector = np.zeros(self.config['num_candidates'], dtype=np.float32)
        else:
            dpf_vector = self.metadata_manager.get_dpf_vector(self.page_history[stream_id][-1])
        
        # Format inputs for the model
        inputs = [
            np.array([cluster_history], dtype=np.int32),
            np.array([self.offset_history[stream_id]], dtype=np.int32),
            np.array([[self.last_pc[stream_id] % self.config['num_pcs']]], dtype=np.int32),
            np.array([[dpf_vector]], dtype=np.float32)
        ]
        
        return inputs
    
    def get_candidate_pages(self, trigger_page):
        """Get candidate pages for prefetching"""
        # Simple strategy: prefetch next sequential pages
        # In a real implementation, this would use the metadata manager
        
        # Default to sequential prefetching if no metadata available
        return [(trigger_page + 1, 100), (trigger_page + 2, 50)]
    
    def predict_prefetches(self, stream_id):
        """Make prefetch predictions"""
        if self.model is None:
            return self.default_predictions(stream_id)
        
        try:
            # Prepare inputs
            inputs = self.prepare_model_inputs(stream_id)
            
            # Get predictions
            candidate_logits, offset_logits = self.model.predict(inputs, verbose=0)
            
            # Convert logits to predictions
            candidate_idx = np.argmax(candidate_logits[0])
            offset_idx = np.argmax(offset_logits[0])
            
            # Check if "no prefetch" option was selected
            if candidate_idx == self.config['num_candidates']:
                return []
            
            # Get candidate pages
            trigger_page = self.page_history[stream_id][-1]
            candidate_pages = self.get_candidate_pages(trigger_page)
            
            # Ensure we have enough candidates
            if len(candidate_pages) <= candidate_idx:
                return self.default_predictions(stream_id)
            
            # Get selected candidate page
            selected_page = candidate_pages[candidate_idx][0]
            
            # Compute prefetch address
            prefetch_cache_line = (selected_page << self.config['offset_bits']) | offset_idx
            prefetch_addr = prefetch_cache_line << 6  # Convert to byte address
            
            return [prefetch_addr]
        except Exception as e:
            print(f"Error making prediction: {e}")
            return self.default_predictions(stream_id)
    
    def default_predictions(self, stream_id):
        """Generate default predictions when model is unavailable"""
        # Simple next-line prefetcher
        trigger_page = self.page_history[stream_id][-1]
        trigger_offset = self.offset_history[stream_id][-1]
        
        # Prefetch next two cache lines
        prefetch1 = ((trigger_page << self.config['offset_bits']) | ((trigger_offset + 1) % self.config['offset_size'])) << 6
        prefetch2 = ((trigger_page << self.config['offset_bits']) | ((trigger_offset + 2) % self.config['offset_size'])) << 6
        
        return [prefetch1, prefetch2]
    
    def train(self, data):
        """Train the model on the given data"""
        print("\n=== Using FixedCruiseFetchLITEModel.train from fixed_model.py ===")
        print(f"Model configuration: {self.config}")
        
        # Process the data to extract features
        print(f"Training on {len(data)} samples")
        
        # Initialize model if not already created
        if self.model is None:
            self.model = self.create_tf_model()
            if self.model is None:
                print("Failed to create TensorFlow model, falling back to next-line prefetching")
                return
        
        # Initialize metadata
        self.metadata_manager = DPFMetadataManager(self.config['num_candidates'])
        
        # Initialize page transition tracking
        self.page_transitions = {}
        
        # Process trace data in first pass to build metadata
        print("Building metadata...")
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            if i % 100000 == 0:
                print(f"Processing entry {i}/{len(data)}")
            
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
        
        # Skip full model training if data is too small
        if len(data) < 10000:
            print("Not enough data for full model training, using metadata-only approach")
            return
        
        try:
            # Prepare training data
            print("Preparing training data...")
            X_clusters = []
            X_offsets = []
            X_pcs = []
            X_dpfs = []
            y_candidates = []
            y_offsets = []
            
            # Group by stream to maintain temporal relationships
            stream_data = {}
            
            for instr_id, cycle_count, load_addr, load_ip, llc_hit in data:
                stream_id = self.get_stream_id(load_ip)
                if stream_id not in stream_data:
                    stream_data[stream_id] = []
                
                stream_data[stream_id].append((instr_id, cycle_count, load_addr, load_ip, llc_hit))
            
            # Process each stream
            for stream_id, stream_entries in stream_data.items():
                for i in range(self.config['history_length'], len(stream_entries)-1):
                    # Create history window
                    history_window = stream_entries[i-self.config['history_length']:i]
                    next_entry = stream_entries[i]
                    
                    # Extract features
                    cluster_history = []
                    offset_history = []
                    
                    for entry in history_window:
                        _, _, addr, _, _ = entry
                        cache_line = addr >> 6
                        page_id = cache_line >> self.config['offset_bits']
                        offset = cache_line & ((1 << self.config['offset_bits']) - 1)
                        
                        cluster_id = self.get_cluster_id(page_id)
                        cluster_history.append(cluster_id)
                        offset_history.append(offset)
                    
                    # Get PC from last entry
                    _, _, _, pc, _ = history_window[-1]
                    pc = pc % self.config['num_pcs']
                    
                    # Get target page and offset
                    _, _, next_addr, _, _ = next_entry
                    next_cache_line = next_addr >> 6
                    next_page = next_cache_line >> self.config['offset_bits']
                    next_offset = next_cache_line & ((1 << self.config['offset_bits']) - 1)
                    
                    # Get DPF vector for trigger page
                    trigger_page = page_id
                    if self.metadata_manager is None:
                        dpf_vector = np.zeros(self.config['num_candidates'], dtype=np.float32)
                    else:
                        dpf_vector = self.metadata_manager.get_dpf_vector(trigger_page)
                    
                    # Get candidate index for target page
                    candidate_pages = self.get_candidate_pages(trigger_page)
                    candidate_idx = self.config['num_candidates']  # Default to no-prefetch
                    
                    for j, (candidate_page, _) in enumerate(candidate_pages):
                        if candidate_page == next_page and j < self.config['num_candidates']:
                            candidate_idx = j
                            break
                    
                    # Add to training data
                    X_clusters.append(cluster_history)
                    X_offsets.append(offset_history)
                    X_pcs.append([pc])
                    X_dpfs.append([dpf_vector])
                    y_candidates.append(candidate_idx)
                    y_offsets.append(next_offset)
            
            # Convert to numpy arrays
            X_clusters = np.array(X_clusters, dtype=np.int32)
            X_offsets = np.array(X_offsets, dtype=np.int32)
            X_pcs = np.array(X_pcs, dtype=np.int32)
            X_dpfs = np.array(X_dpfs, dtype=np.float32)
            y_candidates = np.array(y_candidates, dtype=np.int32)
            y_offsets = np.array(y_offsets, dtype=np.int32)
            
            # Train model
            print(f"Training model on {len(X_clusters)} examples...")
            self.model.fit(
                [X_clusters, X_offsets, X_pcs, X_dpfs],
                [y_candidates, y_offsets],
                epochs=5,
                batch_size=64,
                validation_split=0.1,
                verbose=1
            )
            
            print("Model training complete")
        except Exception as e:
            print(f"Error during training: {e}")
            print("Falling back to metadata-based approach")
    
    def generate(self, data):
        """Generate prefetches for the given data"""
        print("\n=== Using FixedCruiseFetchLITEModel.generate from fixed_model.py ===")
        print(f"Model configuration: {self.config}")
        
        prefetches = []
        
        # Initialize metadata manager if needed
        if self.metadata_manager is None:
            self.metadata_manager = DPFMetadataManager(num_candidates=self.config['num_candidates'])
        
        # Process each memory access
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            if i % 100000 == 0:
                print(f"Processing entry {i}/{len(data)}")
            
            # Process trace entry to update internal state
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
            
            # Get stream ID for this access
            stream_id = self.get_stream_id(load_ip)
            
            # Skip if we don't have enough history
            if stream_id not in self.page_history or self.page_history[stream_id][-1] == 0:
                continue
            
            # Generate prefetches
            if self.model is not None:
                predicted_prefetches = self.predict_prefetches(stream_id)
            else:
                predicted_prefetches = self.default_predictions(stream_id)
            
            # Add prefetches to result
            for prefetch_addr in predicted_prefetches[:self.config['max_prefetches_per_id']]:
                prefetches.append((instr_id, prefetch_addr))
                
                # Update statistics
                self.stats['prefetches_issued'] += 1
                if instr_id not in self.stats['prefetches_per_instr']:
                    self.stats['prefetches_per_instr'][instr_id] = 0
                self.stats['prefetches_per_instr'][instr_id] += 1
        
        print(f"Generated {len(prefetches)} prefetches")
        return prefetches


class DPFMetadataManager:
    """
    Simplified DPF metadata manager for CruiseFetchLITE
    """
    
    def __init__(self, num_candidates=4):
        self.num_candidates = num_candidates
        self.page_metadata = {}  # Maps page_id -> metadata
    
    def update_page_access(self, trigger_page, next_page, trigger_offset, next_offset):
        """
        Update metadata for a page access
        
        Args:
            trigger_page: The page that was accessed
            next_page: The page that was accessed next
            trigger_offset: The offset within the trigger page
            next_offset: The offset within the next page
        """
        if trigger_page not in self.page_metadata:
            self.page_metadata[trigger_page] = {
                'next_pages': {},
                'access_count': 0
            }
        
        # Update access count
        self.page_metadata[trigger_page]['access_count'] += 1
        
        # Update next page counts
        if next_page not in self.page_metadata[trigger_page]['next_pages']:
            self.page_metadata[trigger_page]['next_pages'][next_page] = 0
        
        self.page_metadata[trigger_page]['next_pages'][next_page] += 1
    
    def get_candidate_pages(self, trigger_page):
        """
        Get the top N candidate pages for a trigger page
        
        Args:
            trigger_page: The page that was accessed
            
        Returns:
            List of (page_id, confidence) tuples
        """
        if trigger_page not in self.page_metadata:
            # Default to sequential prefetching
            return [(trigger_page + 1, 100), (trigger_page + 2, 50)]
        
        # Get next pages sorted by frequency
        next_pages = self.page_metadata[trigger_page]['next_pages']
        sorted_pages = sorted(next_pages.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N candidates
        return [(page, count) for page, count in sorted_pages[:self.num_candidates]]
    
    def get_dpf_vector(self, trigger_page):
        """
        Get the DPF vector for a trigger page
        
        Args:
            trigger_page: The page that was accessed
            
        Returns:
            Numpy array of confidence values for each candidate page
        """
        candidates = self.get_candidate_pages(trigger_page)
        
        # Create DPF vector
        dpf_vector = np.zeros(self.num_candidates, dtype=np.float32)
        
        # Fill in confidence values
        for i, (_, confidence) in enumerate(candidates):
            if i < self.num_candidates:
                # Normalize confidence
                total_accesses = self.page_metadata.get(trigger_page, {}).get('access_count', 1)
                dpf_vector[i] = min(1.0, confidence / total_accesses)
        
        return dpf_vector


# Replace this with your own model
Model = FixedCruiseFetchLITEModel


def test_model():
    """Test the fixed model"""
    print("Testing fixed CruiseFetchLITE model...")
    
    # Create model instance
    model = FixedCruiseFetchLITEModel()
    
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
    test_model()
