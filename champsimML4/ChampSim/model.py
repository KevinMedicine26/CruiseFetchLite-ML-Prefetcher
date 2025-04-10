from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import os
import pickle
import lzma
import yaml
from typing import List, Tuple, Dict, Any, Optional

# This model is my Edit virsion for CFpro

# Factory function for creating a model with specific configuration
def create_model_with_config(config_path=None):
    """Factory function to create a model with specific configuration"""
    return CruiseFetchPro(config_path)

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


class CruiseFetchPro(MLPrefetchModel):
    """
    CruiseFetchPro: An enhanced version of CruiseFetchLITE with improved clustering
    and attention mechanisms for more accurate prefetching.
    """
    
    def __init__(self, config_path=None):
        super().__init__()
        # Initialize configuration with custom values
        if config_path:
            self.config = self.load_config(config_path)
        else:
            print("Error: No configuration file provided")
            print("Please provide a valid configuration file path")
            exit(1)
        
        # Initialize TensorFlow model
        self.model = None
        
        # Initialize state variables
        self.clustering_info = {}
        self.page_history = {}
        self.offset_history = {}
        self.last_pc = {}
        self.metadata_manager = None
        
        # Add transition matrix tracking
        self.offset_transition_matrices = {}  # Maps page_id -> transition matrix
        
        # Stream management (handle multiple streams by PC)
        self.num_streams = 16
        self.stream_map = {}  # Maps PC -> stream_id
        
        # Statistics tracking
        self.stats = {
            'accesses': 0,
            'prefetches_issued': 0,
            'prefetches_per_instr': {},
        }
        
        # Memory management parameters from config
        self.matrices_max_entries = self.config.get('max_matrices', 128000)  # 非常宽松的限制(约2GB)
        self.matrices_warning_threshold = self.config.get('warning_threshold', 100000)  # 警告阈值(约1.6GB)
        self.matrices_access_timestamps = {}  # 跟踪访问时间
        self.matrices_current_timestamp = 0  # 时间戳计数器
    
    # Load configuration from YAML file
    def load_config(self, config_path):
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
                    print(f"Loaded k-means centroids with {len(metadata['kmeans_centroids'])} clusters")
        
        # Load page metadata if exists
        if os.path.exists(f"{path}_page_metadata.xz"):
            try:
                with lzma.open(f"{path}_page_metadata.xz", "rb") as f:
                    page_metadata = pickle.load(f)
                    # Create metadata manager with enhanced parameters
                    self.metadata_manager = DPFMetadataManager(
                        num_candidates=self.config.get('num_candidates', 2),
                        positions_tracked=self.config.get('dpf_positions_tracked', 3),
                        max_entries=self.config.get('dpf_max_entries', 1024)
                    )
                    self.metadata_manager.page_metadata = page_metadata
                    print(f"Loaded page metadata with {len(page_metadata)} pages")
            except Exception as e:
                print(f"Error loading page metadata: {e}")
                self.metadata_manager = None
        
        # Initialize other state variables
        self.page_history = {}
        self.offset_history = {}
        self.last_pc = {}
        self.offset_transition_matrices = {}
    
    def save(self, path):
        """Save model to the given path"""
        print(f"Saving model to {path}...")
        
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
            
            # Save k-means centroids if available
            if hasattr(self, 'kmeans_centroids'):
                metadata['kmeans_centroids'] = self.kmeans_centroids
            
            # Save metadata
            with open(f"{path}_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            # Save TensorFlow model if it exists
            if self.model is not None:
                self.model.save(f"{path}_model")
                print("Successfully saved TensorFlow model")
                
            # Save page metadata if exists
            if self.metadata_manager is not None:
                with lzma.open(f"{path}_page_metadata.xz", "wb") as f:
                    pickle.dump(self.metadata_manager.page_metadata, f)
                    print(f"Saved page metadata with {len(self.metadata_manager.page_metadata)} pages")
                
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def create_tf_model(self):
        """Create the TensorFlow model from configuration"""
        try:
            # Define model using Functional API
            # Inputs with properly defined shapes
            cluster_history_input = tf.keras.layers.Input(shape=(self.config['history_length'],), name='cluster_history', dtype=tf.int32)
            offset_history_input = tf.keras.layers.Input(shape=(self.config['history_length'],), name='offset_history', dtype=tf.int32)
            pc_input = tf.keras.layers.Input(shape=(1,), name='pc', dtype=tf.int32)
            
            # Fix dpf_input shape definition to match prepare_model_inputs
            dpf_input = tf.keras.layers.Input(
                shape=(1, self.config['num_candidates']),  # Updated shape to match prepare_model_inputs
                name='dpf', 
                dtype=tf.float32
            )
            
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
            
            # Add positional encoding to cluster embeddings
            positions = tf.range(start=0, limit=self.config['history_length'], delta=1)
            position_embedding = tf.keras.layers.Embedding(
                self.config['history_length'],
                self.config['cluster_embed_size'],
                name='position_embedding'
            )(positions)
            
            # Add position embeddings to the cluster embeddings (broadcasting)
            cluster_embedding_with_pos = cluster_embedding + tf.expand_dims(position_embedding, axis=0)
            
            # Create a learned weighted query from all cluster embeddings
            # This utilizes all history elements instead of just the first one
            cluster_flat = tf.reshape(cluster_embedding_with_pos, [batch_size, -1])
            attention_weights = tf.keras.layers.Dense(
                self.config['history_length'], 
                activation='softmax',
                name='attention_weights'
            )(cluster_flat)
            
            # Apply attention weights for weighted query construction
            attention_weights = tf.reshape(attention_weights, [batch_size, self.config['history_length'], 1])
            weighted_query = tf.reduce_sum(cluster_embedding_with_pos * attention_weights, axis=1)
            query = tf.expand_dims(weighted_query, axis=1)  # Add sequence dimension for attention
            
            # Reshape offset embedding for attention
            offset_embedding_reshaped = tf.reshape(
                offset_embedding, 
                [batch_size, self.config['history_length'] * self.config['num_experts'], self.config['cluster_embed_size']]
            )
            
            # Implement lightweight multi-head attention (2 heads)
            num_heads = 2  # Balance between expressiveness and efficiency
            
            # Create multi-head attention layer
            mha = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=self.config['cluster_embed_size'] // num_heads,
                name='multi_head_attention'
            )
            
            # Apply attention
            attention_output = mha(query, offset_embedding_reshaped)
            context_offset = tf.reshape(attention_output, [batch_size, self.config['cluster_embed_size']])
            
            # Add layer normalization for training stability
            context_offset = tf.keras.layers.LayerNormalization(name='attention_norm')(context_offset)
            
            # Flatten cluster embedding
            cluster_flat = tf.reshape(cluster_embedding_with_pos, [batch_size, self.config['history_length'] * self.config['cluster_embed_size']])
            
            # Flatten DPF vectors with corrected shape
            dpf_flat = tf.reshape(dpf_input, [batch_size, self.config['num_candidates']])
            
            # Concatenate all features
            combined = tf.keras.layers.Concatenate(name='combined_features')(
                [pc_embedding_flat, cluster_flat, context_offset, dpf_flat]
            )
            
            # Ensure combined has a known shape before Dense layer
            combined_shape = combined.get_shape().as_list()
            if None in combined_shape[1:]:
                raise ValueError(f"Combined tensor shape has unknown dimensions: {combined_shape}")
            
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
                name='CruiseFetchPro'
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
        
        # Update offset transition matrix
        if stream_id in self.page_history and len(self.offset_history[stream_id]) >= 2:
            prev_page = self.page_history[stream_id][-2]
            prev_offset = self.offset_history[stream_id][-2]
            curr_page = page_id
            curr_offset = offset
            
            # 初始化矩阵如果需要
            if prev_page != 0:
                # 更新时间戳
                self.matrices_current_timestamp += 1
                
                # 初始化矩阵如果不存在
                if prev_page not in self.offset_transition_matrices:
                    # 检查是否需要管理大小 - 非常宽松的检查
                    self._monitor_matrices_usage()
                    
                    # 创建新矩阵
                    self.offset_transition_matrices[prev_page] = np.zeros(
                        (self.config['offset_size'], self.config['offset_size']), 
                        dtype=np.int32
                    )
                
                # 更新时间戳
                self.matrices_access_timestamps[prev_page] = self.matrices_current_timestamp
                
                # 更新矩阵
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
            # 即使矩阵被移除，也返回缓存的聚类 ID
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
        """Cluster pages based on their offset transition patterns"""
        print("Performing behavioral clustering with k-means...")
        
        # Only cluster if we have enough pages with transition data
        if len(self.offset_transition_matrices) < 10:  # Arbitrary minimum
            print("Not enough pages with transition data for clustering")
            return
        
        # Check if sklearn is available
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Warning: sklearn not available, falling back to simple clustering")
            return
        
        # Prepare data for clustering
        pages = list(self.offset_transition_matrices.keys())
        features = []
        
        for page in pages:
            # Normalize and flatten the transition matrix
            matrix = self.offset_transition_matrices[page]
            total = matrix.sum()
            if total > 0:
                normalized_matrix = matrix / total
            else:
                normalized_matrix = matrix
            features.append(normalized_matrix.flatten())
        
        # Convert to numpy array
        features = np.array(features)
        
        # Apply k-means clustering
        n_clusters = min(self.config['num_clusters'], len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Store centroids for future assignments
        self.kmeans_centroids = kmeans.cluster_centers_
        
        # Assign clusters to pages
        for i, page in enumerate(pages):
            # Check if page exists in clustering_info before assignment
            if page in self.clustering_info:
                self.clustering_info[page]['cluster_id'] = int(clusters[i])
            else:
                # Initialize clustering info for this page if it doesn't exist
                self.clustering_info[page] = {
                    'offsets': np.zeros(self.config['offset_size'], dtype=np.int32),
                    'total_accesses': 1,
                    'cluster_id': int(clusters[i])
                }
            
        print(f"Clustered {len(pages)} pages into {n_clusters} behavioral clusters")
    
    def prepare_model_inputs(self, stream_id):
        """Prepare inputs for the model"""
        # Convert page IDs to cluster IDs
        cluster_history = [self.get_cluster_id(page) for page in self.page_history[stream_id]]
        
        # Create dummy DPF vector if metadata manager is not available
        if self.metadata_manager is not None:
            dpf_vector = self.metadata_manager.get_dpf_vector(self.page_history[stream_id][-1])
        else:
            dpf_vector = np.zeros(self.config['num_candidates'], dtype=np.float32)
        
        # Ensure dpf_vector has the correct shape
        dpf_vector = np.array(dpf_vector, dtype=np.float32)
        if len(dpf_vector) > self.config['num_candidates']:
            # Truncate if too long
            dpf_vector = dpf_vector[:self.config['num_candidates']]
        elif len(dpf_vector) < self.config['num_candidates']:
            # Pad with zeros if too short
            dpf_vector = np.pad(dpf_vector, (0, self.config['num_candidates'] - len(dpf_vector)), 'constant')
        
        # Reshape for model input expectations - ensure consistent shape
        dpf_reshaped = np.reshape(dpf_vector, (1, self.config['num_candidates']))
        
        # Format inputs for the model
        inputs = [
            np.array([cluster_history], dtype=np.int32),
            np.array([self.offset_history[stream_id]], dtype=np.int32),
            np.array([[self.last_pc[stream_id] % self.config['num_pcs']]], dtype=np.int32),
            np.array([dpf_reshaped], dtype=np.float32)  # Changed to ensure consistent shape
        ]
        
        return inputs
    
    def get_candidate_pages(self, trigger_page):
        """Get candidate pages for prefetching"""
        # Delegate to metadata manager if available
        if self.metadata_manager is not None:
            return self.metadata_manager.get_candidate_pages(trigger_page)
        
        # Fallback to sequential prediction
        return [(trigger_page + 1, 100), (trigger_page + 2, 50)]
    
    def predict_prefetches(self, stream_id):
        """Make prefetch predictions with enhanced DPF"""
        if self.model is None or self.page_history[stream_id][-1] == 0:
            return self.default_predictions(stream_id)
        
        try:
            # Prepare inputs
            inputs = self.prepare_model_inputs(stream_id)
            
            # Get predictions
            candidate_logits, offset_logits = self.model.predict(inputs, verbose=0)
            
            # Get trigger page and offset
            trigger_page = self.page_history[stream_id][-1]
            trigger_offset = self.offset_history[stream_id][-1]
            
            # Get top-2 candidate indices and their probabilities
            candidate_probs = tf.nn.softmax(candidate_logits[0]).numpy()
            # Exclude the "no prefetch" option (last index) when finding top candidates
            valid_probs = candidate_probs[:-1]  
            top_candidate_indices = np.argsort(valid_probs)[-2:][::-1]  # Get top 2 in descending order
            
            # Get top-2 offset indices
            offset_probs = tf.nn.softmax(offset_logits[0]).numpy()
            top_offset_indices = np.argsort(offset_probs)[-2:][::-1]  # Get top 2 in descending order
            
            # Get candidate pages
            candidate_pages = self.metadata_manager.get_candidate_pages(trigger_page)
            
            # Check for cross-page boundary patterns
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
                    if self.metadata_manager.should_prefetch(trigger_page, selected_page):
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
                        if self.metadata_manager.should_prefetch(trigger_page, selected_page):
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
            
            # Record prefetch results for adapting position weights
            if self.metadata_manager and hasattr(self.metadata_manager, 'record_prefetch_result'):
                for i, addr in enumerate(prefetch_addresses[:2]):  # Only track first two positions
                    # We don't know hit/miss yet, but we'll use this to mark that we issued a prefetch
                    # Later we could add a proper feedback mechanism
                    self.metadata_manager.record_prefetch_result(i, True)
            
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
            self.model = self.create_tf_model()
            
            if self.model is None:
                print("Failed to create model. Skipping training.")
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
        Generate prefetches for the given trace data
        
        Args:
            data: List of (instr_id, cycle_count, load_addr, load_ip, llc_hit) tuples
            
        Returns:
            List of (instr_id, prefetch_addr) tuples
        """
        print("\n=== Using CruiseFetchPro.generate from model.py ===")
        print(f"Model configuration: {self.config}")
        
        # Process data in streaming fashion
        prefetches = []
        processed_ids = set()
        
        for instr_id, cycle_count, load_addr, load_ip, llc_hit in data:
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
                prefetches.append((instr_id, prefetch_addr))
                
                # Update stats
                self.stats['prefetches_issued'] += 1
                if instr_id not in self.stats['prefetches_per_instr']:
                    self.stats['prefetches_per_instr'][instr_id] = 0
                self.stats['prefetches_per_instr'][instr_id] += 1
        
        print(f"Generated {len(prefetches)} prefetches for {len(data)} memory accesses")
        return prefetches
    
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


class DPFMetadataManager:
    """Manager for delta prefetching metadata"""
    
    def __init__(self, config=None):
        # Use provided config or default values
        if config is None:
            config = {}
            
        self.num_candidates = config.get('num_candidates', 4)
        self.positions_tracked = config.get('dpf_positions_tracked', 3)
        self.max_entries = config.get('dpf_max_entries', 1024)
        self.page_metadata = {}  # Maps page_id -> metadata
        self.access_timestamps = {}  # Track recency of access for replacement
        self.current_timestamp = 0
        
        # Initialize learnable position weights
        self.position_weights = np.array([0.6, 0.3, 0.1][:self.positions_tracked])
        self.position_weights = self.position_weights / np.sum(self.position_weights)
        
        # Add tracking for weight adaptation
        self.position_hit_counters = np.zeros(self.positions_tracked)
        self.position_miss_counters = np.zeros(self.positions_tracked)
        self.weight_update_interval = 1000
        self.updates_counter = 0
        
        # Boundary region configuration
        self.boundary_region_size = config.get('boundary_region_size', 4)
        self.boundary_confidence_threshold = config.get('boundary_confidence_threshold', 3)
        self.config = config  # Store the config for future reference
    
    def update_page_access(self, trigger_page, next_page, trigger_offset, next_offset):
        """Update metadata with a page access"""
        self.current_timestamp += 1
        
        # Track access recency for replacement policy
        self.access_timestamps[trigger_page] = self.current_timestamp
        
        # Initialize metadata if needed
        if trigger_page not in self.page_metadata:
            # Apply replacement policy if needed
            if len(self.page_metadata) >= self.max_entries:
                self._apply_replacement_policy()
                
            self.page_metadata[trigger_page] = {
                'position_successors': [{} for _ in range(self.positions_tracked)],  # Position-based successor tracking
                'offset_transitions': np.zeros((64, 64), dtype=np.int32),
                'last_successors': [0] * self.positions_tracked,  # Last N pages accessed after this one
                'total_accesses': 0,
                'boundary_transitions': {}  # Track cross-page boundary transitions
            }
        
        # Update immediate successor frequency
        position_successors = self.page_metadata[trigger_page]['position_successors'][0]
        if next_page in position_successors:
            position_successors[next_page] += 1
        else:
            position_successors[next_page] = 1
        
        # Update offset transitions
        self.page_metadata[trigger_page]['offset_transitions'][trigger_offset, next_offset] += 1
        
        # Add cross-page boundary detection
        is_boundary_transition = trigger_page != next_page
        is_near_boundary = (trigger_offset >= (64 - self.boundary_region_size) or 
                           trigger_offset < self.boundary_region_size)
        
        if is_boundary_transition and is_near_boundary:
            boundary_key = (trigger_offset, next_page, next_offset)
            
            if boundary_key in self.page_metadata[trigger_page]['boundary_transitions']:
                self.page_metadata[trigger_page]['boundary_transitions'][boundary_key] += 1
            else:
                self.page_metadata[trigger_page]['boundary_transitions'][boundary_key] = 1
        
        # Update total accesses counter
        self.page_metadata[trigger_page]['total_accesses'] += 1
        
        # Update position-based successors for other tracked pages
        for page_id, metadata in self.page_metadata.items():
            # Skip the trigger page
            if page_id == trigger_page:
                continue
                
            # Check if this page has the trigger page in its successor history
            for pos in range(self.positions_tracked - 1):
                if metadata['last_successors'][pos] == trigger_page:
                    # Update the next position successor
                    position_successors = metadata['position_successors'][pos + 1]
                    if next_page in position_successors:
                        position_successors[next_page] += 1
                    else:
                        position_successors[next_page] = 1
        
        # Update successor history for trigger page
        self.page_metadata[trigger_page]['last_successors'] = [next_page] + self.page_metadata[trigger_page]['last_successors'][:-1]
        
        # Apply temporal decay periodically
        self._apply_temporal_decay()
    
    def _apply_temporal_decay(self, decay_factor=0.9, period=1000):
        """Apply temporal decay to all frequency counts periodically"""
        if self.current_timestamp % period == 0:
            for page_id, metadata in self.page_metadata.items():
                # Apply decay to all position successors
                for pos in range(self.positions_tracked):
                    for successor, count in metadata['position_successors'][pos].items():
                        metadata['position_successors'][pos][successor] = max(1, int(count * decay_factor))
    
    def _apply_replacement_policy(self):
        """Apply LRU replacement policy to maintain cache size"""
        # Find least recently used page
        lru_page = min(self.access_timestamps.items(), key=lambda x: x[1])[0]
        
        # Remove it from metadata and timestamps
        if lru_page in self.page_metadata:
            del self.page_metadata[lru_page]
        del self.access_timestamps[lru_page]
    
    def get_dpf_vector(self, trigger_page):
        """Get the DPF vector for a trigger page with position-based weighting"""
        if trigger_page not in self.page_metadata:
            # Default to sequential prediction with exact size
            default_vec = np.zeros(self.num_candidates, dtype=np.float32)
            default_vec[0] = 0.7
            default_vec[1] = 0.3
            return default_vec
        
        # Position weights (higher weight for closer positions)
        position_weights = self.position_weights
        
        # Get candidate pages from all positions with proper weighting
        candidates_scores = {}
        
        for pos in range(self.positions_tracked):
            successors = self.page_metadata[trigger_page]['position_successors'][pos]
            if not successors:
                continue
                
            total_freq = sum(successors.values())
            if total_freq == 0:
                continue
                
            # Calculate normalized scores with position weighting
            for successor, count in successors.items():
                normalized_score = count / total_freq * position_weights[pos]
                if successor in candidates_scores:
                    candidates_scores[successor] += normalized_score
                else:
                    candidates_scores[successor] = normalized_score
        
        # If no candidates found, fallback to sequential
        if not candidates_scores:
            default_vec = np.zeros(self.num_candidates, dtype=np.float32)
            default_vec[0] = 0.7
            default_vec[1] = 0.3
            return default_vec
        
        # Sort by score and create vector
        sorted_candidates = sorted(candidates_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create DPF vector with exact size
        dpf_vector = np.zeros(self.num_candidates, dtype=np.float32)
        
        # Fill with top candidates
        for i, (_, score) in enumerate(sorted_candidates[:self.num_candidates]):
            if i < self.num_candidates:  # Safety check
                dpf_vector[i] = score
        
        # Ensure the vector sums to 1.0
        if np.sum(dpf_vector) > 0:
            dpf_vector = dpf_vector / np.sum(dpf_vector)
            
        return dpf_vector
    
    def get_candidate_pages(self, trigger_page):
        """Get candidate pages with enhanced prediction using offset transitions"""
        if trigger_page not in self.page_metadata:
            # Default to sequential prediction
            return [(trigger_page + 1, 100), (trigger_page + 2, 50)]
        
        # Get position-based successors
        candidates_scores = {}
        
        # Use learned position weights instead of hardcoded values
        position_weights = self.position_weights
        
        # Calculate scores from position-based tracking
        for pos in range(self.positions_tracked):
            successors = self.page_metadata[trigger_page]['position_successors'][pos]
            if not successors:
                continue
                
            weight = position_weights[pos]
            for successor, count in successors.items():
                score = count * weight
                if successor in candidates_scores:
                    candidates_scores[successor] += score
                else:
                    candidates_scores[successor] = score
        
        # Check for boundary transitions when near page boundary
        trigger_offset = None
        # We get the offset from the model when this function is called from predict_prefetches
        # This is a placeholder value that will be replaced with actual offset
        
        # Check for boundary patterns and enhance scores
        if 'boundary_transitions' in self.page_metadata[trigger_page]:
            # Boundary transitions exist for this page
            boundary_transitions = self.page_metadata[trigger_page]['boundary_transitions']
            
            # Find applicable boundary transitions
            for (offset, next_page, next_offset), count in boundary_transitions.items():
                if count >= self.boundary_confidence_threshold:
                    # Add high score for frequently observed boundary transitions
                    if next_page in candidates_scores:
                        candidates_scores[next_page] += count * 5  # Higher weight for boundary transitions
                    else:
                        candidates_scores[next_page] = count * 5
        
        # Get transition matrix confidence
        transition_matrix = self.page_metadata[trigger_page]['offset_transitions']
        matrix_confidence = min(1.0, self.page_metadata[trigger_page]['total_accesses'] / 1000)
        
        # Adjust scores based on transition pattern stability
        # Pages with stable transition patterns get higher confidence
        for successor in list(candidates_scores.keys()):
            if successor in self.page_metadata:
                # Check if successor has a stable pattern
                successor_matrix = self.page_metadata[successor]['offset_transitions']
                pattern_similarity = self._calculate_pattern_similarity(transition_matrix, successor_matrix)
                
                # Adjust score based on pattern similarity
                candidates_scores[successor] *= (1.0 + pattern_similarity * matrix_confidence)
        
        # If no candidates or very low confidence, add sequential predictions
        if not candidates_scores or max(candidates_scores.values()) < 10:
            candidates_scores[trigger_page + 1] = candidates_scores.get(trigger_page + 1, 0) + 100
            candidates_scores[trigger_page + 2] = candidates_scores.get(trigger_page + 2, 0) + 50
        
        # Sort and return top candidates
        sorted_candidates = sorted(candidates_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:self.num_candidates]
    
    def _calculate_pattern_similarity(self, matrix1, matrix2):
        """Calculate similarity between two transition patterns"""
        # Simple cosine similarity between flattened matrices
        flat1 = matrix1.flatten()
        flat2 = matrix2.flatten()
        
        # Normalize
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(flat1, flat2) / (norm1 * norm2)
    
    def should_prefetch(self, trigger_page, candidate_page, confidence_threshold=0.2):
        """Determine if a prefetch should be issued based on confidence"""
        if trigger_page not in self.page_metadata:
            # Default behavior for unknown pages
            return candidate_page in [trigger_page + 1, trigger_page + 2]
        
        # Get position-based confidence
        confidence = 0.0
        total_accesses = self.page_metadata[trigger_page]['total_accesses']
        
        if total_accesses == 0:
            return False
            
        # Calculate confidence from position 0 (immediate successor)
        pos0_successors = self.page_metadata[trigger_page]['position_successors'][0]
        if candidate_page in pos0_successors:
            confidence = pos0_successors[candidate_page] / total_accesses
            
        # Only prefetch if confidence exceeds threshold
        return confidence >= confidence_threshold
        
    def record_prefetch_result(self, position, was_hit):
        """Record the result of a prefetch for weight adaptation"""
        if 0 <= position < len(self.position_hit_counters):
            if was_hit:
                self.position_hit_counters[position] += 1
            else:
                self.position_miss_counters[position] += 1
                
        self.updates_counter += 1
        if self.updates_counter >= self.weight_update_interval:
            self._update_position_weights()
            self.updates_counter = 0
            
    def _update_position_weights(self):
        """Update position weights based on observed hit rates"""
        # Calculate hit rates per position
        total_attempts = self.position_hit_counters + self.position_miss_counters
        hit_rates = np.zeros_like(self.position_weights)
        
        for i in range(len(hit_rates)):
            if total_attempts[i] > 0:
                hit_rates[i] = self.position_hit_counters[i] / total_attempts[i]
            else:
                hit_rates[i] = 0.0
        
        # Apply exponential smoothing to update weights
        alpha = 0.1  # Smoothing factor
        if np.sum(hit_rates) > 0:
            new_weights = hit_rates / np.sum(hit_rates)
            self.position_weights = (1 - alpha) * self.position_weights + alpha * new_weights
            
            # Normalize weights
            self.position_weights = self.position_weights / np.sum(self.position_weights)
            
            print(f"Updated position weights: {self.position_weights}")
    
    def get_cross_page_predictions(self, trigger_page, trigger_offset):
        """Get predictions for cross-page access patterns based on boundary transitions"""
        if trigger_page not in self.page_metadata or 'boundary_transitions' not in self.page_metadata[trigger_page]:
            return []
        
        prefetch_candidates = []
        is_near_boundary = (trigger_offset >= (64 - self.boundary_region_size) or 
                           trigger_offset < self.boundary_region_size)
        
        if is_near_boundary:
            boundary_transitions = self.page_metadata[trigger_page]['boundary_transitions']
            
            # Find applicable boundary transitions for current offset
            for (offset, next_page, next_offset), count in boundary_transitions.items():
                if offset == trigger_offset and count >= self.boundary_confidence_threshold:
                    # Create prefetch with target page and offset
                    prefetch_cache_line = (next_page << 6) | next_offset  # Assuming 6-bit offset
                    prefetch_addr = prefetch_cache_line << 6  # Convert to byte address
                    prefetch_candidates.append((prefetch_addr, count))
        
        # Sort by confidence (count) and return
        sorted_candidates = sorted(prefetch_candidates, key=lambda x: x[1], reverse=True)
        return [addr for addr, _ in sorted_candidates]


# Check if a configuration file exists and use it, otherwise use default config
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cruisefetch_config.yml')
if os.path.exists(config_path):
    print(f"Using configuration from {config_path}")
    Model = create_model_with_config(config_path)
else:
    print("No configuration file found, using default configuration")
    Model = CruiseFetchPro()
