import tensorflow as tf
import numpy as np
import os
import pickle
import threading
from typing import List, Tuple, Dict, Any, Optional

from .base_model import MLPrefetchModel
from .metadata_manager import DPFMetadataManager
from .model_components import create_tf_model, cluster_pages
from .tf_config import configure_tensorflow
from .config_loader import load_config

# Configure TensorFlow
configure_tensorflow()

class CruiseFetchPro(MLPrefetchModel):
    """
    CruiseFetchPro: An enhanced version of CruiseFetchLITE with improved clustering
    and attention mechanisms for more accurate prefetching.
    """
    
    def __init__(self, config_path=None):
        # Initialize configuration with custom values
        if config_path:
            self.config = load_config(config_path)
        else:
            print("Error: No configuration file provided")
            print("Please provide a valid configuration file path")
            exit(1)
        
        # Initialize TensorFlow model
        self.model = None
        
        # Initialize state
        self._initialize_state()
        self._initialize_stats()
        self._initialize_memory_management()
    
    def _initialize_state(self):
        """Initialize state variables"""
        # Clustering information
        self.clustering_info = {}
        
        # History tracking
        self.page_history = {}
        self.offset_history = {}
        self.last_pc = {}
        
        # Metadata manager
        self.metadata_manager = None
        
        # Add transition matrix tracking
        self.offset_transition_matrices = {}  # Maps page_id -> transition matrix
        
        # Stream management (handle multiple streams by PC)
        self.num_streams = 16
        self.stream_map = {}  # Maps PC -> stream_id
        
        # K-means centroids
        self.kmeans_centroids = None
    
    def _initialize_stats(self):
        """Initialize statistics tracking"""
        self.stats = {
            'accesses': 0,
            'prefetches_issued': 0,
            'prefetches_per_instr': {},
        }
        
        # Lock for thread-safe stats updates
        self._stats_lock = threading.Lock()
    
    def _initialize_memory_management(self):
        """Initialize memory management for transition matrices"""
        # Memory management parameters from config
        self.matrices_max_entries = self.config.get('max_matrices', 128000)  
        self.matrices_warning_threshold = self.config.get('warning_threshold', 100000)  
        self.matrices_access_timestamps = {}  
        self.matrices_current_timestamp = 0
    
    def load(self, path):
        """Load model from the given path"""
        print(f"Loading model from {path}...")
        
        # Load TensorFlow model if exists
        if os.path.exists(f"{path}_model"):
            try:
                self.model = tf.keras.models.load_model(f"{path}_model")
                print(f"Loaded TensorFlow model from {path}_model")
            except Exception as e:
                print(f"Error loading TensorFlow model: {e}")
                self.model = None
        
        # Load metadata
        if os.path.exists(f"{path}_metadata.pkl"):
            with open(f"{path}_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.config = metadata['config']
                self.clustering_info = metadata['clustering_info']
                self.stream_map = metadata['stream_map']
                self.stats = metadata['stats']
                
                # Load k-means centroids if available
                if 'kmeans_centroids' in metadata:
                    self.kmeans_centroids = metadata['kmeans_centroids']
                
                print(f"Loaded metadata from {path}_metadata.pkl")
                print(f"Found {len(self.clustering_info)} pages with clustering info")
                print(f"Found {len(self.stream_map)} stream mappings")
        
        # Create metadata manager
        self.metadata_manager = DPFMetadataManager(self.config)
        
        # Initialize memory management
        self.matrices_max_entries = self.config.get('max_matrices', 128000)
        self.matrices_warning_threshold = self.config.get('warning_threshold', 100000)
        
        print(f"Model loaded successfully from {path}")
        return self
    
    def save(self, path):
        """Save model to the given path"""
        print(f"Saving model to {path}...")
        
        # Save TensorFlow model if exists
        if self.model is not None:
            try:
                self.model.save(f"{path}_model")
                print(f"Saved TensorFlow model to {path}_model")
            except Exception as e:
                print(f"Error saving TensorFlow model: {e}")
        
        # Save metadata (clustering info, stream map, etc.)
        metadata = {
            'config': self.config,
            'clustering_info': self.clustering_info,
            'stream_map': self.stream_map,
            'stats': self.stats
        }
        
        # Save k-means centroids if available
        if hasattr(self, 'kmeans_centroids') and self.kmeans_centroids is not None:
            metadata['kmeans_centroids'] = self.kmeans_centroids
        
        with open(f"{path}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
            print(f"Saved metadata to {path}_metadata.pkl")
        
        print(f"Model saved successfully to {path}")
        return self
        
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
        
        # Update offset transition matrix
        if stream_id in self.page_history and len(self.offset_history[stream_id]) >= 2:
            prev_page = self.page_history[stream_id][-2]
            prev_offset = self.offset_history[stream_id][-2]
            curr_page = page_id
            curr_offset = offset
            
            # Initialize matrix if needed
            if prev_page != 0:
                # Update timestamp
                self.matrices_current_timestamp += 1
                
                # Initialize matrix if it doesn't exist
                if prev_page not in self.offset_transition_matrices:
                    # Check if we need to manage size - very relaxed check
                    self._monitor_matrices_usage()
                    
                    # Create new matrix
                    self.offset_transition_matrices[prev_page] = np.zeros(
                        (self.config['offset_size'], self.config['offset_size']), 
                        dtype=np.int32
                    )
                
                # Update timestamp
                self.matrices_access_timestamps[prev_page] = self.matrices_current_timestamp
                
                # Update matrix
                self.offset_transition_matrices[prev_page][prev_offset, curr_offset] += 1
        
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
        if page_id in self.clustering_info:
            # Update offset frequencies for existing page
            self.clustering_info[page_id]['offsets'][offset] += 1
            self.clustering_info[page_id]['total_accesses'] += 1
        else:
            # Initialize offset frequencies for new page
            self.clustering_info[page_id] = {
                'offsets': np.zeros(self.config['offset_size'], dtype=np.int32),
                'total_accesses': 1,
                'cluster_id': -1  # Not assigned yet
            }
            self.clustering_info[page_id]['offsets'][offset] = 1
    
    def get_cluster_id(self, page_id):
        """Get cluster ID for a page ID based on behavioral similarity"""
        # Special case for page_id 0 (sentinel value)
        if page_id == 0:
            return 0  # Use cluster_id 0 for sentinel page_id 0
        
        if page_id in self.clustering_info:
            return self.clustering_info[page_id]['cluster_id']
        
        # If we have transition data for this page, find the closest cluster
        if page_id in self.offset_transition_matrices and hasattr(self, 'kmeans_centroids'):
            matrix = self.offset_transition_matrices[page_id]
            total = matrix.sum()
            if total > 0:
                normalized_matrix = matrix / total
            else:
                normalized_matrix = matrix
            
            feature = normalized_matrix.flatten().reshape(1, -1)
            
            # Find closest centroid
            distances = np.linalg.norm(feature - self.kmeans_centroids, axis=1)
            cluster_id = int(np.argmin(distances))
        elif page_id in self.clustering_info:
            # Even if matrix was removed, return cached cluster ID
            cluster_id = self.clustering_info[page_id]['cluster_id']
            return cluster_id
        else:
            # Fall back to hash-based assignment for new pages
            cluster_id = hash(page_id) % self.config['num_clusters']
        
        # Initialize clustering info for this page
        self.clustering_info[page_id] = {
            'offsets': np.zeros(self.config['offset_size'], dtype=np.int32),
            'total_accesses': 0,
            'cluster_id': cluster_id
        }
        
        return cluster_id
    
    def cluster_pages(self):
        """Apply clustering to pages based on transition matrices"""
        from .model_components import cluster_pages as cluster_pages_func
        
        # Only process if we have enough transition matrices
        if len(self.offset_transition_matrices) < 10:
            print("Not enough transition matrices for meaningful clustering")
            return
            
        # Perform clustering
        clustering_result, centroids = cluster_pages_func(
            self.offset_transition_matrices, 
            self.config
        )
        
        # Update clustering info with results
        if clustering_result:
            self.clustering_info.update(clustering_result)
            
        # Store centroids for future assignments
        if centroids is not None:
            self.kmeans_centroids = centroids
    
    def prepare_model_inputs(self, stream_id):
        """Prepare inputs for the TensorFlow model"""
        # Get history and features
        page_history = self.page_history[stream_id]
        offset_history = self.offset_history[stream_id]
        load_pc = self.last_pc[stream_id]
        
        # Convert page IDs to cluster IDs
        cluster_history = [self.get_cluster_id(page_id) for page_id in page_history]
        
        # Get DPF vector if metadata manager exists
        trigger_page = page_history[-1]
        if self.metadata_manager:
            dpf_vector = self.metadata_manager.get_dpf_vector(trigger_page)
        else:
            # Default vector with sequential bias
            dpf_vector = np.zeros(self.config['num_candidates'], dtype=np.float32)
            dpf_vector[0] = 0.7  # Strong bias for sequential pattern
            dpf_vector[1] = 0.3  # Small probability for stride+2
        
        # Create input arrays with batch dimension
        cluster_history_array = np.array([cluster_history], dtype=np.int32)
        offset_history_array = np.array([offset_history], dtype=np.int32)
        pc_array = np.array([[load_pc]], dtype=np.int32)
        dpf_array = np.array([dpf_vector], dtype=np.float32)
        
        return [cluster_history_array, offset_history_array, pc_array, dpf_array]
    
    def predict_prefetches(self, stream_id):
        """Predict prefetches for a given stream ID"""
        try:
            # Skip if no model or insufficient history
            if self.model is None or self.page_history[stream_id][-1] == 0:
                return self.default_predictions(stream_id)
            
            # Prepare model inputs
            inputs = self.prepare_model_inputs(stream_id)
            
            # Make prediction
            candidate_logits, offset_logits = self.model.predict(inputs, verbose=0)
            
            # Get trigger page and offset
            trigger_page = self.page_history[stream_id][-1]
            trigger_offset = self.offset_history[stream_id][-1]
            
            # Get top-2 candidate indices and their probabilities
            candidate_probs = tf.nn.softmax(candidate_logits).numpy()[0]
            # Exclude the "no prefetch" option (last index) when finding top candidates
            valid_probs = candidate_probs[:-1]  
            top_candidate_indices = np.argsort(valid_probs)[-2:][::-1]  # Get top 2 in descending order
            
            # Get top-2 offset indices
            offset_probs = tf.nn.softmax(offset_logits).numpy()[0]
            top_offset_indices = np.argsort(offset_probs)[-2:][::-1]  # Get top 2 in descending order
            
            # Get candidate pages
            if self.metadata_manager:
                candidate_pages = self.metadata_manager.get_candidate_pages(trigger_page)
            else:
                # Fallback
                candidate_pages = [(trigger_page + 1, 100), (trigger_page + 2, 50)]
            
            # Check for cross-page boundary patterns
            is_near_boundary = False
            if self.metadata_manager:
                is_near_boundary = (trigger_offset >= (64 - self.metadata_manager.boundary_region_size) or 
                                   trigger_offset < self.metadata_manager.boundary_region_size)
            
            # List to store prefetch addresses
            prefetch_addresses = []
            
            # If near boundary, check for cross-page transitions first
            if is_near_boundary and self.metadata_manager and hasattr(self.metadata_manager, 'get_cross_page_predictions'):
                cross_page_predictions = self.metadata_manager.get_cross_page_predictions(trigger_page, trigger_offset)
                prefetch_addresses.extend(cross_page_predictions[:self.config['max_prefetches_per_id']])
            
            # If we still have room for prefetches, use the normal prediction logic
            if len(prefetch_addresses) < self.config['max_prefetches_per_id']:
                # First strategy: Top candidate with top offset
                if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
                    selected_page, confidence = candidate_pages[top_candidate_indices[0]]
                    if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                        # Create prefetch with top candidate page and top offset
                        prefetch_cache_line = (selected_page << self.config['offset_bits']) | top_offset_indices[0]
                        prefetch_addr = prefetch_cache_line << 6
                        
                        # Only add if not already present
                        if prefetch_addr not in prefetch_addresses:
                            prefetch_addresses.append(prefetch_addr)
                
                # Second strategy: Either second candidate page or first candidate with second offset
                if len(prefetch_addresses) < self.config['max_prefetches_per_id']:
                    # Try second candidate page with top offset first (if available)
                    if len(top_candidate_indices) > 1 and top_candidate_indices[1] < len(candidate_pages):
                        selected_page, confidence = candidate_pages[top_candidate_indices[1]]
                        if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                            prefetch_cache_line = (selected_page << self.config['offset_bits']) | top_offset_indices[0]
                            prefetch_addr = prefetch_cache_line << 6
                            # Only add if it's different from existing prefetches
                            if prefetch_addr not in prefetch_addresses:
                                prefetch_addresses.append(prefetch_addr)
                    
                    # If still don't have a second prefetch, try first candidate with second offset
                    if len(prefetch_addresses) < self.config['max_prefetches_per_id'] and len(top_offset_indices) > 1:
                        if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
                            selected_page, confidence = candidate_pages[top_candidate_indices[0]]
                            prefetch_cache_line = (selected_page << self.config['offset_bits']) | top_offset_indices[1]
                            prefetch_addr = prefetch_cache_line << 6
                            # Only add if it's different from existing prefetches
                            if prefetch_addr not in prefetch_addresses:
                                prefetch_addresses.append(prefetch_addr)
            
            # If no prefetches were generated, fall back to default predictions
            if not prefetch_addresses:
                return self.default_predictions(stream_id)
                
            # Limit to max prefetches per ID
            return prefetch_addresses[:self.config['max_prefetches_per_id']]
            
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
        """
        Train the model on the given trace data
        
        Args:
            data: List of (instr_id, cycle_count, load_addr, load_ip, llc_hit) tuples
        """
        print(f"Training model on {len(data)} records...")
        
        # Process trace data to gather training examples
        cluster_history_examples = []
        offset_history_examples = []
        pc_examples = []
        dpf_examples = []
        labels_candidate = []
        labels_offset = []
        
        # Initialize page histories for each stream
        self.page_history = {i: [0] * self.config['history_length'] for i in range(self.num_streams)}
        self.offset_history = {i: [0] * self.config['history_length'] for i in range(self.num_streams)}
        
        # Initialize metadata manager if not exists
        if not self.metadata_manager:
            self.metadata_manager = DPFMetadataManager(self.config)
        
        # First pass: Collect page transition data for clustering
        prev_page = None
        prev_offset = None
        
        for instr_id, cycle_count, load_addr, load_ip, llc_hit in data:
            cache_line_addr = load_addr >> 6  # 64-byte cache line
            page_id = cache_line_addr >> self.config['offset_bits']
            offset = cache_line_addr & ((1 << self.config['offset_bits']) - 1)
            
            # Update transition matrices
            if prev_page is not None:
                self._monitor_matrices_usage()
                
                if prev_page not in self.offset_transition_matrices:
                    # Create new matrix
                    self.offset_transition_matrices[prev_page] = np.zeros(
                        (self.config['offset_size'], self.config['offset_size']), 
                        dtype=np.int32
                    )
                    
                self.matrices_current_timestamp += 1
                self.matrices_access_timestamps[prev_page] = self.matrices_current_timestamp
                self.offset_transition_matrices[prev_page][prev_offset, offset] += 1
                
                # Update DPF metadata
                self.metadata_manager.update_page_access(prev_page, page_id, prev_offset, offset)
            
            prev_page = page_id
            prev_offset = offset
        
        # Perform clustering
        self.cluster_pages()
        
        # Second pass: Collect training examples with cluster information
        prev_page = None
        prev_offset = None
        
        # Reset page histories
        self.page_history = {i: [0] * self.config['history_length'] for i in range(self.num_streams)}
        self.offset_history = {i: [0] * self.config['history_length'] for i in range(self.num_streams)}
        
        for instr_id, cycle_count, load_addr, load_ip, llc_hit in data:
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
            
            # Extract features and labels for training
            cache_line_addr = load_addr >> 6  # 64-byte cache line
            page_id = cache_line_addr >> self.config['offset_bits']
            offset = cache_line_addr & ((1 << self.config['offset_bits']) - 1)
            stream_id = self.get_stream_id(load_ip)
            
            if prev_page is not None and self.page_history[stream_id][-1] != 0:
                # Get input features - using fixed method
                inputs = self.prepare_model_inputs(stream_id)
                
                # Safely extract components
                cluster_history_examples.append(inputs[0][0])
                offset_history_examples.append(inputs[1][0])
                pc_examples.append(inputs[2][0][0])
                dpf_examples.append(inputs[3][0])
                
                # Get target labels
                target_candidates = self.metadata_manager.get_candidate_pages(prev_page)
                
                # Find current page in candidates
                found_idx = -1  # -1 represents "no prefetch"
                for i, (candidate_page, _) in enumerate(target_candidates):
                    if candidate_page == page_id:
                        found_idx = i
                        break
                
                # Add label (index of correct candidate)
                labels_candidate.append(found_idx if found_idx != -1 else self.config['num_candidates'])
                labels_offset.append(offset)
            
            prev_page = page_id
            prev_offset = offset
        
        # Create and train TensorFlow model if we have enough examples
        if len(cluster_history_examples) > 100:
            # Create model
            self.model = create_tf_model(self.config)
            
            if self.model is None:
                print("Failed to create model. Completely aborting training.")
                return self
            
            # Convert to numpy arrays - now with consistent shapes
            cluster_history_array = np.array(cluster_history_examples, dtype=np.int32)
            offset_history_array = np.array(offset_history_examples, dtype=np.int32)
            pc_array = np.array(pc_examples, dtype=np.int32).reshape(-1, 1)
            dpf_array = np.array(dpf_examples, dtype=np.float32)
            
            labels_candidate_array = np.array(labels_candidate, dtype=np.int32)
            labels_offset_array = np.array(labels_offset, dtype=np.int32)
            
            # Train model
            try:
                self.model.fit(
                    [cluster_history_array, offset_history_array, pc_array, dpf_array],
                    [labels_candidate_array, labels_offset_array],
                    epochs=5,
                    batch_size=64,
                    verbose=1
                )
                print("Model trained successfully.")
            except Exception as e:
                print(f"Error during training: {e}")
        else:
            print("Not enough training examples to train the model.")
        
        return self
    
    def generate(self, data):
        """
        Generate prefetches for the given trace data using GPU-accelerated batch processing
        
        Args:
            data: List of (instr_id, cycle_count, load_addr, load_ip, llc_hit) tuples
            
        Returns:
            List of (instr_id, prefetch_addr) tuples
        """
        print("\n=== Using GPU-Accelerated CruiseFetchPro.generate from model.py ===")
        print(f"Model configuration: {self.config}")
        
        if self.model is None:
            print("No model available, using CPU-based fallback processing")
            return self._legacy_generate(data)
        
        # Initialize results list
        prefetches = []
        
        # Process trace entries sequentially for correct state updates
        # But perform model predictions in batches
        batch_size = self.config.get('predict_batch_size', 64)
        
        # Prepare batch containers
        batch_inputs = [[], [], [], []]  # For each of the 4 model input types
        batch_info = []  # Track instruction IDs and stream IDs for each batch entry
        
        # Track number of prefetches per instruction ID
        prefetch_counts = {}
        
        print(f"Processing {len(data)} entries with batch size {batch_size}")
        
        # Process entries and collect into batches
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # Skip if we've reached the maximum prefetches for this ID
            if instr_id in prefetch_counts and prefetch_counts[instr_id] >= self.config['max_prefetches_per_id']:
                continue
                
            # Process the trace entry to update internal state
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
            
            # Get stream_id for this entry
            stream_id = self.get_stream_id(load_ip)
            
            # Skip entries that won't produce meaningful predictions
            if self.page_history[stream_id][-1] == 0:
                # Add default predictions for entries we're skipping
                default_predictions = self.default_predictions(stream_id)
                for pf_addr in default_predictions:
                    if instr_id not in prefetch_counts:
                        prefetch_counts[instr_id] = 0
                    
                    if prefetch_counts[instr_id] < self.config['max_prefetches_per_id']:
                        prefetches.append((instr_id, pf_addr))
                        prefetch_counts[instr_id] += 1
                continue
            
            # Prepare model inputs for this trace entry
            inputs = self.prepare_model_inputs(stream_id)
            
            # Add inputs to batch
            for j in range(4):  # 4 input tensors
                batch_inputs[j].append(inputs[j][0])  # Append without the batch dimension
            
            # Store info needed to process the prediction later
            batch_info.append((instr_id, stream_id))
            
            # When batch is full or at end of data, run batch prediction
            if len(batch_info) >= batch_size or i == len(data) - 1:
                if batch_info:  # Only process if we have any entries
                    # Convert lists to tensors
                    batch_tensors = []
                    for j in range(4):
                        # Convert to appropriate dtype
                        if j < 3:  # First 3 inputs are int32
                            batch_tensors.append(tf.convert_to_tensor(batch_inputs[j], dtype=tf.int32))
                        else:  # Last input is float32
                            batch_tensors.append(tf.convert_to_tensor(batch_inputs[j], dtype=tf.float32))
                    
                    # Run batch prediction with GPU acceleration
                    try:
                        # Use predict method which should use GPU if available
                        batch_results = self.model.predict(batch_tensors, verbose=0, batch_size=len(batch_info))
                        
                        # Process each result in the batch
                        for k, (curr_instr_id, curr_stream_id) in enumerate(batch_info):
                            # Skip if we've already issued max prefetches for this ID
                            if curr_instr_id in prefetch_counts and prefetch_counts[curr_instr_id] >= self.config['max_prefetches_per_id']:
                                continue
                                
                            # Extract this sample's results
                            candidate_logits = batch_results[0][k:k+1]  # Keep batch dimension for tensorflow ops
                            offset_logits = batch_results[1][k:k+1]
                            
                            # Get trigger page and offset
                            trigger_page = self.page_history[curr_stream_id][-1]
                            trigger_offset = self.offset_history[curr_stream_id][-1]
                            
                            # Process prediction to generate prefetches
                            prefetch_list = self._process_batch_prediction(
                                candidate_logits, offset_logits, trigger_page, trigger_offset, curr_stream_id
                            )
                            
                            # Initialize counter if needed
                            if curr_instr_id not in prefetch_counts:
                                prefetch_counts[curr_instr_id] = 0
                                
                            # Add prefetches for this instruction, respecting the per-instruction limit
                            for prefetch_addr in prefetch_list:
                                if prefetch_counts[curr_instr_id] < self.config['max_prefetches_per_id']:
                                    prefetches.append((curr_instr_id, prefetch_addr))
                                    prefetch_counts[curr_instr_id] += 1
                                    
                                    # Update stats
                                    with self._stats_lock:
                                        self.stats['prefetches_issued'] += 1
                                        if curr_instr_id not in self.stats['prefetches_per_instr']:
                                            self.stats['prefetches_per_instr'][curr_instr_id] = 0
                                        self.stats['prefetches_per_instr'][curr_instr_id] += 1
                    
                    except Exception as e:
                        print(f"Error in batch prediction: {e}")
                        # Fall back to default predictions for this batch
                        for k, (curr_instr_id, curr_stream_id) in enumerate(batch_info):
                            # Skip if we've already issued max prefetches for this ID
                            if curr_instr_id in prefetch_counts and prefetch_counts[curr_instr_id] >= self.config['max_prefetches_per_id']:
                                continue
                                
                            # Get default predictions
                            default_preds = self.default_predictions(curr_stream_id)
                            
                            # Initialize counter if needed
                            if curr_instr_id not in prefetch_counts:
                                prefetch_counts[curr_instr_id] = 0
                                
                            # Add default prefetches
                            for prefetch_addr in default_preds:
                                if prefetch_counts[curr_instr_id] < self.config['max_prefetches_per_id']:
                                    prefetches.append((curr_instr_id, prefetch_addr))
                                    prefetch_counts[curr_instr_id] += 1
                    
                    # Reset batch containers for next batch
                    batch_inputs = [[], [], [], []]
                    batch_info = []
        
        return prefetches
    
    def _process_batch_prediction(self, candidate_logits, offset_logits, trigger_page, trigger_offset, stream_id):
        """Process a single batch prediction result to generate prefetches"""
        try:
            # Get top-2 candidate indices and their probabilities
            candidate_probs = tf.nn.softmax(candidate_logits).numpy()[0]
            # Exclude the "no prefetch" option (last index) when finding top candidates
            valid_probs = candidate_probs[:-1]  
            top_candidate_indices = np.argsort(valid_probs)[-2:][::-1]  # Get top 2 in descending order
            
            # Get top-2 offset indices
            offset_probs = tf.nn.softmax(offset_logits).numpy()[0]
            top_offset_indices = np.argsort(offset_probs)[-2:][::-1]  # Get top 2 in descending order
            
            # Get candidate pages
            if self.metadata_manager:
                candidate_pages = self.metadata_manager.get_candidate_pages(trigger_page)
            else:
                # Fallback
                candidate_pages = [(trigger_page + 1, 100), (trigger_page + 2, 50)]
            
            # List to store prefetch addresses for this instruction
            prefetch_addresses = []
            
            # Check for cross-page boundary patterns
            is_near_boundary = False
            if self.metadata_manager:
                is_near_boundary = (trigger_offset >= (64 - self.metadata_manager.boundary_region_size) or 
                                   trigger_offset < self.metadata_manager.boundary_region_size)
            
            # If near boundary, check for cross-page transitions first
            if is_near_boundary and self.metadata_manager and hasattr(self.metadata_manager, 'get_cross_page_predictions'):
                cross_page_predictions = self.metadata_manager.get_cross_page_predictions(trigger_page, trigger_offset)
                prefetch_addresses.extend(cross_page_predictions[:self.config['max_prefetches_per_id']])
            
            # If we still have room for prefetches, use the normal prediction logic
            if len(prefetch_addresses) < self.config['max_prefetches_per_id']:
                # First strategy: Top candidate with top offset
                if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
                    selected_page, confidence = candidate_pages[top_candidate_indices[0]]
                    if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                        # Create prefetch with top candidate page and top offset
                        prefetch_cache_line = (selected_page << self.config['offset_bits']) | top_offset_indices[0]
                        prefetch_addr = prefetch_cache_line << 6
                        
                        # Only add if not already present
                        if prefetch_addr not in prefetch_addresses:
                            prefetch_addresses.append(prefetch_addr)
                
                # Second strategy: Either second candidate page or first candidate with second offset
                if len(prefetch_addresses) < self.config['max_prefetches_per_id']:
                    # Try second candidate page with top offset first (if available)
                    if len(top_candidate_indices) > 1 and top_candidate_indices[1] < len(candidate_pages):
                        selected_page, confidence = candidate_pages[top_candidate_indices[1]]
                        if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                            prefetch_cache_line = (selected_page << self.config['offset_bits']) | top_offset_indices[0]
                            prefetch_addr = prefetch_cache_line << 6
                            # Only add if it's different from existing prefetches
                            if prefetch_addr not in prefetch_addresses:
                                prefetch_addresses.append(prefetch_addr)
                    
                    # If still don't have a second prefetch, try first candidate with second offset
                    if len(prefetch_addresses) < self.config['max_prefetches_per_id'] and len(top_offset_indices) > 1:
                        if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
                            selected_page, confidence = candidate_pages[top_candidate_indices[0]]
                            prefetch_cache_line = (selected_page << self.config['offset_bits']) | top_offset_indices[1]
                            prefetch_addr = prefetch_cache_line << 6
                            # Only add if it's different from existing prefetches
                            if prefetch_addr not in prefetch_addresses:
                                prefetch_addresses.append(prefetch_addr)
            
            # If no prefetches were generated, fall back to default predictions
            if not prefetch_addresses:
                return self.default_predictions(stream_id)
                
            # Limit to max prefetches per ID
            return prefetch_addresses[:self.config['max_prefetches_per_id']]
            
        except Exception as e:
            print(f"Error processing batch prediction: {e}")
            return self.default_predictions(stream_id)
    
    def _legacy_generate(self, data):
        """
        Legacy CPU-based implementation for generating prefetches.
        Used as a fallback when the GPU model is not available.
        
        Args:
            data: List of (instr_id, cycle_count, load_addr, load_ip, llc_hit) tuples
            
        Returns:
            List of (instr_id, prefetch_addr) tuples
        """
        print("Using legacy CPU-based prediction...")
        
        # Group data by PC (creating streams)
        stream_data = {}
        
        for entry in data:
            instr_id, cycle_count, load_addr, load_ip, llc_hit = entry
            stream_id = self.get_stream_id(load_ip)
            
            if stream_id not in stream_data:
                stream_data[stream_id] = []
                
            stream_data[stream_id].append(entry)
        
        # Process each stream separately
        all_prefetches = []
        
        for stream_id, stream_entries in stream_data.items():
            prefetches = self._process_stream(stream_id, stream_entries)
            all_prefetches.extend(prefetches)
            
        return all_prefetches
    
    def _process_stream(self, stream_id, stream_entries):
        """
        Process a single stream of memory accesses.
        Used by the legacy CPU-based implementation.
        """
        stream_prefetches = []
        processed_ids = set()
        
        for instr_id, cycle_count, load_addr, load_ip, llc_hit in stream_entries:
            # Skip if we've reached the maximum prefetches for this ID
            if instr_id in self.stats['prefetches_per_instr'] and \
               self.stats['prefetches_per_instr'][instr_id] >= self.config['max_prefetches_per_id']:
                continue
            
            # Process the memory access
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
            
            # Get stream ID
            stream_id = self.get_stream_id(load_ip)
            
            # Generate prefetches for this access
            predicted_prefetches = self.predict_prefetches(stream_id)
            
            # Add prefetches to output
            for prefetch_addr in predicted_prefetches:
                # Skip if we've reached the maximum prefetches for this ID
                if instr_id in self.stats['prefetches_per_instr'] and \
                   self.stats['prefetches_per_instr'][instr_id] >= self.config['max_prefetches_per_id']:
                    break
                
                # Add prefetch
                stream_prefetches.append((instr_id, prefetch_addr))
                
                # Update stats - need to handle thread safety here
                with self._stats_lock:
                    self.stats['prefetches_issued'] += 1
                    if instr_id not in self.stats['prefetches_per_instr']:
                        self.stats['prefetches_per_instr'][instr_id] = 0
                    self.stats['prefetches_per_instr'][instr_id] += 1
        
        return stream_prefetches
    
    def get_stream_id(self, pc):
        """Get stream ID for a PC value"""
        if pc not in self.stream_map:
            # Assign a stream ID based on PC
            stream_id = hash(pc) % self.num_streams
            self.stream_map[pc] = stream_id
        
        return self.stream_map[pc]
    
    def _monitor_matrices_usage(self):
        """Monitor matrices usage"""
        current_size = len(self.offset_transition_matrices)
        
        # Only perform relaxed LRU cleanup when very close to limit
        if current_size >= self.matrices_max_entries * 0.95:  # 95% threshold
            self._apply_relaxed_matrices_replacement()
            return True
        
        return False

    def _apply_relaxed_matrices_replacement(self):
        """Execute relaxed matrix replacement, only when close to limit"""
        if len(self.offset_transition_matrices) >= int(self.matrices_max_entries * 0.95):
            # Get all matrices sorted by timestamp
            sorted_matrices = sorted(self.matrices_access_timestamps.items(), key=lambda x: x[1])
            
            # Calculate number of matrices to remove (5% of total)
            num_to_remove = int(len(sorted_matrices) * self.config.get('cleanup_percentage', 0.05))
            
            # Delete the least recently used matrices
            for i in range(num_to_remove):
                page_id = sorted_matrices[i][0]
                if page_id in self.offset_transition_matrices:
                    del self.offset_transition_matrices[page_id]
                    del self.matrices_access_timestamps[page_id]
