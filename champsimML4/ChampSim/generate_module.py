"""
CruiseFetchLitePro
Generate Module to generate prefetch txt file for ChampSim
currently use TensorFlow cpu parallel
further optimization: use GPU
"""

import numpy as np
import tensorflow as tf
import random
import time
import os
import psutil  # use for monitoring CPU usage


# ============= Set TensorFlow Parallel Configuration =============
# Get CPU cores
CPU_CORES = os.cpu_count() or 8  # If cannot detect, default to 8

# Set TensorFlow internal parallelism
# Control the number of threads within a single operation (e.g., matrix multiplication) 
tf.config.threading.set_intra_op_parallelism_threads(CPU_CORES - 2)  

# Set inter-op parallelism
tf.config.threading.set_inter_op_parallelism_threads(2)

# Enable mixed precision, may further accelerate (but ensure acceptable precision)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

print(f"TensorFlow parallel configuration enabled: {CPU_CORES} CPU cores")
print(f"- Intra-op parallelism threads: {CPU_CORES - 2}")
print(f"- Inter-op parallelism threads: 2")

def predict_prefetches(self, stream_id):
    """Make prefetch predictions with enhanced DPF"""
    if self.model is None or self.page_history[stream_id][-1] == 0:
        return default_predictions(self, stream_id)
    
    try:
        # Prepare inputs
        inputs = self.prepare_model_inputs(stream_id)
        
        # Get predictions using optimized prediction if available
        if hasattr(self, 'optimized_predict'):
            cluster_history, offset_history, pc, dpf = inputs
            candidate_logits, offset_logits = self.optimized_predict(
                cluster_history, offset_history, pc, dpf
            )
        else:
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
            return default_predictions(self, stream_id)
            
        # Limit to max prefetches per ID
        return prefetch_addresses[:self.config['max_prefetches_per_id']]
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return default_predictions(self, stream_id)

def default_predictions(self, stream_id):
    """Generate default predictions when model is unavailable"""
    # Simple next-line prefetcher
    trigger_page = self.page_history[stream_id][-1]
    trigger_offset = self.offset_history[stream_id][-1]
    
    # Prefetch next two cache lines
    prefetch1 = ((trigger_page << self.config.get('offset_bits', 6)) | ((trigger_offset + 1) % self.config.get('offset_size', 64))) << 6
    prefetch2 = ((trigger_page << self.config.get('offset_bits', 6)) | ((trigger_offset + 2) % self.config.get('offset_size', 64))) << 6
    
    return [prefetch1, prefetch2]

def create_cache_key(self, stream_id):
    """Create smart cache key, balance precision and memory usage"""
    # Basic key: recent history and PC value
    recent_pages = tuple(self.page_history[stream_id][-4:])
    recent_offsets = tuple(self.offset_history[stream_id][-4:])
    pc_hash = self.last_pc[stream_id] % 1024  # Limit PC space size
    
    # Add a "pattern fingerprint" to capture longer history features
    if len(self.page_history[stream_id]) > 4:
        full_history_hash = hash(tuple(self.page_history[stream_id])) % 128
    else:
        full_history_hash = 0
        
    return (recent_pages, recent_offsets, pc_hash, full_history_hash)

def optimize_model_for_inference(self):
    """Optimize TensorFlow model for inference (parallel version)"""
    if self.model is None:
        return False
        
    try:
        # Explicitly set TensorFlow thread settings
        num_cores = os.cpu_count() or 8
        tf.config.threading.set_intra_op_parallelism_threads(num_cores - 2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # Efficient definition of input signature
        # Dynamic batch size handling, suit for any batch size
        @tf.function
        def optimized_predict(cluster_history, offset_history, pc, dpf):
            return self.model([cluster_history, offset_history, pc, dpf], training=False)
        
        # Use compiled function to replace model's predict method
        self.optimized_predict = optimized_predict
        
        # Warm-up function to ensure compilation
        # Create small batch test data for warm-up
        hist_len = self.config.get('history_length', 16)
        num_cand = self.config.get('num_candidates', 4)
        
        test_cluster_history = tf.zeros([1, hist_len], dtype=tf.int32)
        test_offset_history = tf.zeros([1, hist_len], dtype=tf.int32)
        test_pc = tf.zeros([1, 1], dtype=tf.int32)
        test_dpf = tf.zeros([1, num_cand], dtype=tf.float32)
        
        # Warm-up function to compile graph
        _ = optimized_predict(test_cluster_history, test_offset_history, test_pc, test_dpf)
        
        print("Model optimized for inference with parallel execution")
        return True
    except Exception as e:
        print(f"Could not optimize model: {e}")
        return False

def _monitor_matrices_usage_optimized(self):
    """Optimized matrix usage monitoring"""
    current_size = len(self.offset_transition_matrices)
    
    # Only perform cleanup check when close to limit
    if current_size >= self.matrices_max_entries * 0.9:  # 90% threshold
        # Batch cleanup, remove more infrequently used matrices
        sorted_matrices = sorted(
            self.matrices_access_timestamps.items(), 
            key=lambda x: x[1]
        )
        
        # Remove 10% of least recently used matrices
        num_to_remove = max(1, int(len(sorted_matrices) * 0.1))
        removed_matrices = []
        
        for i in range(min(num_to_remove, len(sorted_matrices))):
            page_id = sorted_matrices[i][0]
            if page_id in self.offset_transition_matrices:
                del self.offset_transition_matrices[page_id]
                del self.matrices_access_timestamps[page_id]
                removed_matrices.append(page_id)
        
        if removed_matrices:
            print(f"Removed {len(removed_matrices)} matrices to save memory")
        return True
    
    return False

def _process_model_outputs(self, candidate_logits, offset_logits, stream_id):
    """Process model outputs, generate prefetch addresses"""
    # Get trigger page and offset
    trigger_page = self.page_history[stream_id][-1]
    trigger_offset = self.offset_history[stream_id][-1]
    
    # Get candidate indices and probabilities
    candidate_probs = tf.nn.softmax(candidate_logits[0]).numpy()
    valid_probs = candidate_probs[:-1]  
    top_candidate_indices = np.argsort(valid_probs)[-2:][::-1]
    
    # Get offset indices
    offset_probs = tf.nn.softmax(offset_logits[0]).numpy()
    top_offset_indices = np.argsort(offset_probs)[-2:][::-1]
    
    # Get candidate pages
    if self.metadata_manager:
        candidate_pages = self.metadata_manager.get_candidate_pages(trigger_page)
    else:
        # Default sequential prediction
        candidate_pages = [(trigger_page + 1, 100), (trigger_page + 2, 50)]
    
    # Check cross-page boundary pattern
    is_near_boundary = False
    if self.metadata_manager and hasattr(self.metadata_manager, 'boundary_region_size'):
        is_near_boundary = (trigger_offset >= (64 - self.metadata_manager.boundary_region_size) or 
                           trigger_offset < self.metadata_manager.boundary_region_size)
    
    # Prefetch address list
    prefetch_addresses = []
    
    # If close to boundary, first check cross-page conversion
    if is_near_boundary and self.metadata_manager and hasattr(self.metadata_manager, 'get_cross_page_predictions'):
        cross_page_predictions = self.metadata_manager.get_cross_page_predictions(trigger_page, trigger_offset)
        prefetch_addresses.extend(cross_page_predictions[:self.config.get('max_prefetches_per_id', 2)])
    
    # If still space for prefetch, use regular prediction logic
    if len(prefetch_addresses) < self.config.get('max_prefetches_per_id', 2):
        # First strategy: top candidate page and top offset
        if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
            selected_page, confidence = candidate_pages[top_candidate_indices[0]]
            if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                prefetch_cache_line = (selected_page << self.config.get('offset_bits', 6)) | top_offset_indices[0]
                prefetch_addr = prefetch_cache_line << 6
                
                if prefetch_addr not in prefetch_addresses:
                    prefetch_addresses.append(prefetch_addr)
        
        # Second strategy: second candidate page or first candidate page with second offset
        if len(prefetch_addresses) < self.config.get('max_prefetches_per_id', 2):
            # First try second candidate page with top offset
            if len(top_candidate_indices) > 1 and top_candidate_indices[1] < len(candidate_pages):
                selected_page, confidence = candidate_pages[top_candidate_indices[1]]
                if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                    prefetch_cache_line = (selected_page << self.config.get('offset_bits', 6)) | top_offset_indices[0]
                    prefetch_addr = prefetch_cache_line << 6
                    
                    if prefetch_addr not in prefetch_addresses:
                        prefetch_addresses.append(prefetch_addr)
            
            # If still no second prefetch, try first candidate page with second offset
            if len(prefetch_addresses) < self.config.get('max_prefetches_per_id', 2) and len(top_offset_indices) > 1:
                if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
                    selected_page, confidence = candidate_pages[top_candidate_indices[0]]
                    prefetch_cache_line = (selected_page << self.config.get('offset_bits', 6)) | top_offset_indices[1]
                    prefetch_addr = prefetch_cache_line << 6
                    
                    if prefetch_addr not in prefetch_addresses:
                        prefetch_addresses.append(prefetch_addr)
    
    # Record prefetch results to adapt position weights
    if self.metadata_manager and hasattr(self.metadata_manager, 'record_prefetch_result'):
        for i, addr in enumerate(prefetch_addresses[:2]):
            self.metadata_manager.record_prefetch_result(i, True)
    
    # If no prefetch generated, fallback to default prediction
    if not prefetch_addresses:
        return default_predictions(self, stream_id)
    
    # Limit each ID's maximum prefetches
    return prefetch_addresses[:self.config.get('max_prefetches_per_id', 2)]

def get_cpu_utilization():
    """Get current CPU utilization (all cores)"""
    try:
        # Try to use psutil to get CPU utilization
        if 'psutil' in globals():
            return psutil.cpu_percent(interval=0.1, percpu=True)
        return None
    except:
        return None

def generate_optimized_parallel(self, data):
    """
    Optimized version of generate using TensorFlow internal parallelism
    
    Args:
        data: trace of the ram, format as [(instr_id, cycle_count, load_addr, load_ip, llc_hit), ...]
        
    Returns:
        prefetch list, format as [(instr_id, prefetch_addr), ...]
    """
    print("\n=== Using CPU-optimized CruiseFetchPro.generate with TensorFlow internal parallelism ===")
    print(f"Model configuration: {self.config}")
    print(f"CPU Cores: {CPU_CORES}")
    
    # Save original matrix monitoring method
    original_monitor_method = self._monitor_matrices_usage
    # Replace with optimized version
    self._monitor_matrices_usage = lambda: _monitor_matrices_usage_optimized(self)
    
    # Try to optimize model for inference
    model_optimized = False
    if self.model is not None:
        model_optimized = optimize_model_for_inference(self)
        print(f"Model optimization {'succeeded' if model_optimized else 'failed'}")
    
    prefetches = []
    # Increase batch size to better utilize parallelism
    # But still keep batch size within a reasonable range to avoid memory issues
    batch_size = min(1024, len(data) // 10 + 1)  
    print(f"Using batch size: {batch_size}")
    
    # Initialize cache system
    if not hasattr(self, 'prediction_cache'):
        self.prediction_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.max_cache_size = 20000  # limit the use of cache memory
    
    # Batch processing related variables
    batch_inputs = {
        'stream_ids': [],
        'instr_ids': [],
        'cache_keys': [],
        'cache_misses': []  # indices of items to predict
    }
    
    batch_cluster_history = []
    batch_offset_history = []
    batch_pc = []
    batch_dpf = []
    
    # Timer and monitoring variables
    start_time = time.time()
    last_report_time = start_time
    report_interval = 5.0  # report progress every 5 seconds
    last_cpu_check_time = start_time
    cpu_check_interval = 15.0  # check CPU utilization every 15 seconds
    
    try:
        for idx, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # Progress report
            if time.time() - last_report_time > report_interval:
                elapsed = time.time() - start_time
                percent_done = (idx + 1) / len(data) * 100
                records_per_sec = (idx + 1) / elapsed
                eta_seconds = (len(data) - idx - 1) / records_per_sec if records_per_sec > 0 else 0
                
                print(f"Progress: {percent_done:.1f}% ({idx+1}/{len(data)}) - "
                      f"Speed: {records_per_sec:.1f} recs/s - "
                      f"ETA: {eta_seconds/60:.1f} min")
                last_report_time = time.time()
                
                # Periodically check CPU utilization
                if time.time() - last_cpu_check_time > cpu_check_interval:
                    cpu_usage = get_cpu_utilization()
                    if cpu_usage:
                        avg_usage = sum(cpu_usage) / len(cpu_usage)
                        print(f"CPU Utilization: {avg_usage:.1f}% (avg), {cpu_usage}")
                    last_cpu_check_time = time.time()
            
            # Skip instructions that have reached the maximum prefetch count
            if instr_id in self.stats['prefetches_per_instr'] and \
               self.stats['prefetches_per_instr'][instr_id] >= self.config.get('max_prefetches_per_id', 2):
                continue
            
            # Process memory access, update state
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
            
            # Get stream ID
            stream_id = self.get_stream_id(load_ip)
            
            # Create smart cache key
            cache_key = create_cache_key(self, stream_id)
            
            # Add to batch
            batch_inputs['stream_ids'].append(stream_id)
            batch_inputs['instr_ids'].append(instr_id)
            batch_inputs['cache_keys'].append(cache_key)
            
            # Check if cache hit
            if cache_key in self.prediction_cache:
                # Cache hit
                self.cache_stats['hits'] += 1
            else:
                # Cache miss - need prediction
                self.cache_stats['misses'] += 1
                batch_inputs['cache_misses'].append(len(batch_inputs['stream_ids']) - 1)
                
                # Prepare inputs for this item
                cluster_hist = np.array([self.get_cluster_id(page) for page in self.page_history[stream_id]], dtype=np.int32)
                batch_cluster_history.append(cluster_hist)
                batch_offset_history.append(np.array(self.offset_history[stream_id], dtype=np.int32))
                batch_pc.append(np.array([self.last_pc[stream_id] % self.config.get('num_pcs', 4096)], dtype=np.int32))
                
                # Get DPF vector
                if self.metadata_manager is not None:
                    dpf_vector = self.metadata_manager.get_dpf_vector(self.page_history[stream_id][-1])
                else:
                    dpf_vector = np.zeros(self.config.get('num_candidates', 4), dtype=np.float32)
                    
                # Ensure correct shape
                if len(dpf_vector) > self.config.get('num_candidates', 4):
                    dpf_vector = dpf_vector[:self.config.get('num_candidates', 4)]
                elif len(dpf_vector) < self.config.get('num_candidates', 4):
                    dpf_vector = np.pad(dpf_vector, (0, self.config.get('num_candidates', 4) - len(dpf_vector)), 'constant')
                    
                batch_dpf.append(dpf_vector)
            
            # When batch size is reached or end of data is reached
            if len(batch_inputs['stream_ids']) >= batch_size or idx == len(data) - 1:
                if batch_inputs['cache_misses']:
                    # Only predict for cache misses
                    miss_count = len(batch_inputs['cache_misses'])
                    if miss_count > 0 and self.model is not None:
                        try:
                            # Merge all cache misses into tensors
                            # Note: Use tf.convert_to_tensor for faster data transfer
                            model_inputs = [
                                tf.convert_to_tensor(np.stack([batch_cluster_history[i] for i in range(miss_count)]), dtype=tf.int32),
                                tf.convert_to_tensor(np.stack([batch_offset_history[i] for i in range(miss_count)]), dtype=tf.int32),
                                tf.convert_to_tensor(np.stack([batch_pc[i] for i in range(miss_count)]), dtype=tf.int32),
                                tf.convert_to_tensor(np.stack([batch_dpf[i] for i in range(miss_count)]), dtype=tf.float32)
                            ]
                            
                            # Get batch predictions
                            if model_optimized and hasattr(self, 'optimized_predict'):
                                # Use optimized prediction function
                                batch_candidate_logits, batch_offset_logits = self.optimized_predict(*model_inputs)
                            else:
                                # Use standard prediction
                                batch_candidate_logits, batch_offset_logits = self.model.predict(
                                    model_inputs, verbose=0
                                )
                            
                            # Process each prediction
                            for batch_idx, miss_idx in enumerate(batch_inputs['cache_misses']):
                                stream_id = batch_inputs['stream_ids'][miss_idx]
                                cache_key = batch_inputs['cache_keys'][miss_idx]
                                
                                # Process prediction to get prefetch address
                                predicted_prefetches = _process_model_outputs(
                                    self,
                                    batch_candidate_logits[batch_idx:batch_idx+1],
                                    batch_offset_logits[batch_idx:batch_idx+1],
                                    stream_id
                                )
                                
                                # calculate the confidence
                                confidence = float(tf.reduce_max(tf.nn.softmax(batch_candidate_logits[batch_idx])).numpy())
                                
                                # only cache high confidence predictions
                                if confidence > 0.5:  # configurable confidence threshold
                                    # add to cache, use LRU strategy to manage cache size
                                    if len(self.prediction_cache) >= self.max_cache_size:
                                        # randomly remove 1% of cache items to free up space
                                        keys_to_remove = random.sample(
                                            list(self.prediction_cache.keys()), 
                                            max(1, int(len(self.prediction_cache) * 0.01))
                                        )
                                        for key in keys_to_remove:
                                            del self.prediction_cache[key]
                                    
                                    # add to cache, use LRU strategy to manage cache size
                                    self.prediction_cache[cache_key] = predicted_prefetches
                        except Exception as e:
                            print(f"Error in batch prediction: {e}")
                            # Record cache misses, fallback to default prediction
                            for miss_idx in batch_inputs['cache_misses']:
                                stream_id = batch_inputs['stream_ids'][miss_idx]
                                cache_key = batch_inputs['cache_keys'][miss_idx]
                                self.prediction_cache[cache_key] = default_predictions(self, stream_id)
                
                # Process all items in the batch to generate prefetches
                for idx, (stream_id, instr_id, cache_key) in enumerate(zip(
                    batch_inputs['stream_ids'], 
                    batch_inputs['instr_ids'],
                    batch_inputs['cache_keys']
                )):
                    # Get prediction from cache or result
                    if cache_key in self.prediction_cache:
                        predicted_prefetches = self.prediction_cache[cache_key]
                    else:
                        # If not in cache, use default prediction
                        predicted_prefetches = default_predictions(self, stream_id)
                    
                    # Add prefetches to output, maintain original order
                    for prefetch_addr in predicted_prefetches:
                        if instr_id in self.stats['prefetches_per_instr'] and \
                           self.stats['prefetches_per_instr'][instr_id] >= self.config.get('max_prefetches_per_id', 2):
                            break
                        
                        prefetches.append((instr_id, prefetch_addr))
                        
                        # Update statistics
                        self.stats['prefetches_issued'] += 1
                        if instr_id not in self.stats['prefetches_per_instr']:
                            self.stats['prefetches_per_instr'][instr_id] = 0
                        self.stats['prefetches_per_instr'][instr_id] += 1
                
                # Reset batch processing
                batch_inputs = {'stream_ids': [], 'instr_ids': [], 'cache_keys': [], 'cache_misses': []}
                batch_cluster_history = []
                batch_offset_history = []
                batch_pc = []
                batch_dpf = []
    
        # Print cache statistics
        total_cache_accesses = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = 0
        if total_cache_accesses > 0:
            hit_rate = self.cache_stats['hits'] / total_cache_accesses
        
        print(f"Generated {len(prefetches)} prefetches for {len(data)} memory accesses")
        print(f"Cache stats: {self.cache_stats['hits']} hits, {self.cache_stats['misses']} misses, {hit_rate:.2f} hit rate")
        print(f"Cache size: {len(self.prediction_cache)} entries")
        
        if start_time:
            total_time = time.time() - start_time
            records_per_sec = len(data) / total_time
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Average speed: {records_per_sec:.1f} records/second")
            
            # Print final CPU utilization
            final_cpu_usage = get_cpu_utilization()
            if final_cpu_usage:
                avg_usage = sum(final_cpu_usage) / len(final_cpu_usage)
                print(f"Final CPU Utilization: {avg_usage:.1f}% (avg), {final_cpu_usage}")
    
    except Exception as e:
        print(f"Error in generate_optimized_parallel: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original matrix monitoring method
        self._monitor_matrices_usage = original_monitor_method
    
    return prefetches
def generate_optimized(self, data):
    """legacy function, use generate_optimized_parallel instead"""
    return generate_optimized_parallel(self, data)