import tensorflow as tf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans

def create_tf_model(config):
    """Create the TensorFlow model from configuration"""
    try:
        print("Creating TensorFlow model...")
        
        # Extract model parameters from config
        history_length = config['history_length']
        num_clusters = config['num_clusters']
        offset_size = config['offset_size']
        num_candidates = config['num_candidates']
        
        # Input layers
        cluster_history_input = tf.keras.layers.Input(
            shape=(history_length,), 
            dtype=tf.int32,
            name='cluster_history'
        )
        
        offset_history_input = tf.keras.layers.Input(
            shape=(history_length,), 
            dtype=tf.int32,
            name='offset_history'
        )
        
        pc_input = tf.keras.layers.Input(
            shape=(1,), 
            dtype=tf.int32,
            name='pc'
        )
        
        dpf_input = tf.keras.layers.Input(
            shape=(num_candidates,), 
            dtype=tf.float32,
            name='dpf_vector'
        )
        
        # PC Embedding
        pc_embedding = tf.keras.layers.Embedding(
            input_dim=config['pc_embed_vocab_size'],
            output_dim=config['pc_embed_size'],
            name='pc_embedding'
        )(pc_input)
        pc_embedding_flat = tf.keras.layers.Flatten(name='pc_embedding_flat')(pc_embedding)
        
        # Cluster Embedding
        cluster_embedding = tf.keras.layers.Embedding(
            input_dim=num_clusters + 1,  # +1 for zero padding
            output_dim=config['cluster_embed_size'],
            name='cluster_embedding'
        )(cluster_history_input)
        
        # Offset Embeddings
        offset_embedding = tf.keras.layers.Embedding(
            input_dim=offset_size,
            output_dim=config['cluster_embed_size'],
            name='offset_embedding'
        )(offset_history_input)
        
        # Positional Encoding for history elements
        if config.get('use_positional_encoding', True):
            # Create learnable position embeddings
            positions = tf.range(start=0, limit=history_length, delta=1)
            position_embedding = tf.keras.layers.Embedding(
                input_dim=history_length,
                output_dim=config['cluster_embed_size'],
                name='position_embedding'
            )(positions)
            
            # Add position embeddings to cluster and offset embeddings
            cluster_embedding = cluster_embedding + position_embedding
            offset_embedding = offset_embedding + position_embedding
        
        # Multi-head attention for cluster history
        if config.get('use_attention', True):
            # Create key and value from combined cluster and offset embeddings
            kv_inputs = tf.keras.layers.Add()([cluster_embedding, offset_embedding])
            
            # Create query from learned combination of history elements
            query_weights = tf.keras.layers.Dense(
                history_length, 
                activation='softmax',
                name='query_weights'
            )(tf.keras.layers.Flatten()(kv_inputs))
            
            # Reshape query weights to broadcast properly
            query_weights = tf.reshape(query_weights, [-1, history_length, 1])
            
            # Apply weights to get query
            query = tf.reduce_sum(kv_inputs * query_weights, axis=1, keepdims=True)
            
            # Apply multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=config.get('num_attention_heads', 2),
                key_dim=config['cluster_embed_size'] // config.get('num_attention_heads', 2),
                name='multi_head_attention'
            )(query, kv_inputs)
            
            # Apply layer normalization
            attention_output = tf.keras.layers.LayerNormalization(
                name='attention_norm'
            )(attention_output)
            
            # Get context vector
            context_offset = tf.keras.layers.Flatten(name='context_vector')(attention_output)
        else:
            # Simpler version without attention
            cluster_flat = tf.keras.layers.Flatten(name='cluster_flat')(cluster_embedding)
            offset_flat = tf.keras.layers.Flatten(name='offset_flat')(offset_embedding)
            combined_history = tf.keras.layers.Concatenate(name='combined_history')([cluster_flat, offset_flat])
            context_offset = tf.keras.layers.Dense(
                config['cluster_embed_size'],
                activation='relu',
                name='context_projection'
            )(combined_history)
        
        # Determine cluster embedding flat size
        if config.get('use_attention', True):
            cluster_flat_size = 0  # Not using flattened cluster directly
        else:
            cluster_flat_size = history_length * config['cluster_embed_size']
            cluster_flat = tf.keras.layers.Flatten(name='cluster_flat')(cluster_embedding)
        
        # Concatenate all features
        if config.get('use_attention', True):
            combined = tf.keras.layers.Concatenate(name='combined_features')(
                [pc_embedding_flat, context_offset, dpf_input]
            )
            
            # Calculate combined dimension
            combined_dim = (
                config['pc_embed_size'] +          # PC embedding
                config['cluster_embed_size'] +     # Context vector from attention
                config['num_candidates']           # DPF vector
            )
        else:
            combined = tf.keras.layers.Concatenate(name='combined_features')(
                [pc_embedding_flat, cluster_flat, context_offset, dpf_input]
            )
            
            # Calculate combined dimension
            combined_dim = (
                config['pc_embed_size'] +                # PC embedding
                cluster_flat_size +                      # Flattened cluster embeddings
                config['cluster_embed_size'] +          # Context vector
                config['num_candidates']                # DPF vector
            )
        
        print(f"Expected combined dimension: {combined_dim}")
        
        # Ensure combined has proper shape by explicitly reshaping if needed
        combined_shape = combined.get_shape().as_list()
        if None in combined_shape[1:]:
            print(f"Warning: Combined shape has undefined dimensions: {combined_shape}")
            print(f"Applying explicit reshape to dimension {combined_dim}")
            combined = tf.reshape(combined, [-1, combined_dim])
        
        # Output heads
        candidate_logits = tf.keras.layers.Dense(
            num_candidates + 1,  # Add one for "no prefetch"
            activation=None,
            kernel_regularizer='l1',
            name='candidate_output'
        )(combined)
        
        offset_logits = tf.keras.layers.Dense(
            offset_size,
            activation=None,
            kernel_regularizer='l1',
            name='offset_output'
        )(combined)
        
        # Create and compile model
        model = tf.keras.Model(
            inputs=[cluster_history_input, offset_history_input, pc_input, dpf_input],
            outputs=[candidate_logits, offset_logits],
            name='CruiseFetchPro'
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
            loss=[
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            ]
        )
        
        print("TensorFlow model created successfully")
        return model
        
    except Exception as e:
        import traceback
        print(f"Error creating TensorFlow model: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def normalize_matrix(matrix_tuple):
    """Normalize a transition matrix for clustering"""
    page_id, matrix = matrix_tuple
    total = matrix.sum()
    if total > 0:
        normalized = matrix / total
    else:
        normalized = matrix
    return page_id, normalized.flatten()

def cluster_pages(offset_transition_matrices, config):
    """
    Cluster pages based on their offset transition patterns
    
    Args:
        offset_transition_matrices: Dictionary mapping page_id to transition matrix
        config: Configuration dictionary
        
    Returns:
        Dictionary with clustering information
    """
    print("Performing memory-efficient parallel behavioral clustering with k-means...")
    
    # Add debugging info
    print(f"Number of matrices available: {len(offset_transition_matrices)}")
    
    if len(offset_transition_matrices) < 10:
        print("Not enough transition matrices to perform meaningful clustering")
        # Return simple hash-based assignments
        clustering_info = {}
        for page_id in offset_transition_matrices.keys():
            cluster_id = hash(page_id) % config['num_clusters']
            clustering_info[page_id] = {
                'offsets': np.zeros(config['offset_size'], dtype=np.int32),
                'total_accesses': 0,
                'cluster_id': cluster_id
            }
        return clustering_info, None
    
    # Extract and normalize transition matrices in parallel
    print("Extracting and normalizing transition matrices...")
    
    matrix_items = list(offset_transition_matrices.items())
    
    try:
        with ProcessPoolExecutor() as executor:
            normalized_matrices = list(executor.map(normalize_matrix, matrix_items))
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        normalized_matrices = [normalize_matrix(item) for item in matrix_items]
    
    # Prepare feature matrix for k-means
    page_ids = [page_id for page_id, _ in normalized_matrices]
    features = np.array([feature for _, feature in normalized_matrices])
    
    # Apply k-means clustering
    print(f"Applying k-means clustering with {config['num_clusters']} clusters...")
    
    try:
        # Use mini-batch k-means for efficiency
        kmeans = KMeans(
            n_clusters=config['num_clusters'],
            random_state=42,
            n_init='auto'
        )
        cluster_labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_
        
        print("Clustering completed successfully")
        
        # Create clustering info dictionary
        clustering_info = {}
        
        # Assign cluster IDs to pages
        for i, page_id in enumerate(page_ids):
            cluster_id = int(cluster_labels[i])
            
            # Initialize or update clustering info
            if page_id not in clustering_info:
                clustering_info[page_id] = {
                    'offsets': np.zeros(config['offset_size'], dtype=np.int32),
                    'total_accesses': 0,
                    'cluster_id': cluster_id
                }
            else:
                clustering_info[page_id]['cluster_id'] = cluster_id
        
        # Calculate cluster distribution
        cluster_counts = np.bincount(cluster_labels, minlength=config['num_clusters'])
        print("Cluster distribution:")
        for i, count in enumerate(cluster_counts):
            print(f"  Cluster {i}: {count} pages ({count/len(page_ids)*100:.2f}%)")
        
        return clustering_info, centroids
        
    except Exception as e:
        print(f"K-means clustering failed: {e}")
        
        # Fall back to hash-based assignments
        print("Falling back to hash-based cluster assignments")
        clustering_info = {}
        for page_id in page_ids:
            cluster_id = hash(page_id) % config['num_clusters']
            clustering_info[page_id] = {
                'offsets': np.zeros(config['offset_size'], dtype=np.int32),
                'total_accesses': 0,
                'cluster_id': cluster_id
            }
        
        return clustering_info, None
