import numpy as np

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
        
        # Initialize fixed position weights instead of adaptive ones
        self.position_weights = np.array([0.6, 0.3, 0.1][:self.positions_tracked])
        self.position_weights = self.position_weights / np.sum(self.position_weights)
        
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
            # Get successors for this position
            position_successors = self.page_metadata[trigger_page]['position_successors'][pos]
            
            # Apply weighted scores to each candidate based on position
            weight = position_weights[pos]
            for successor, count in position_successors.items():
                if successor in candidates_scores:
                    candidates_scores[successor] += count * weight
                else:
                    candidates_scores[successor] = count * weight
        
        # Create vector with top candidates
        candidates = sorted(candidates_scores.items(), key=lambda x: x[1], reverse=True)[:self.num_candidates]
        
        # Normalize scores
        total_score = sum(score for _, score in candidates)
        if total_score == 0:
            # Default if no candidates found
            default_vec = np.zeros(self.num_candidates, dtype=np.float32)
            default_vec[0] = 0.7
            default_vec[1] = 0.3
            return default_vec
        
        # Create normalized vector
        vector = np.zeros(self.num_candidates, dtype=np.float32)
        for i, (_, score) in enumerate(candidates):
            if i < self.num_candidates:
                vector[i] = score / total_score
        
        return vector
    
    def get_candidate_pages(self, trigger_page):
        """Get candidate pages with enhanced prediction using offset transitions"""
        candidates = []
        
        if trigger_page not in self.page_metadata:
            # Sequential prediction with confidence values
            return [(trigger_page + 1, 0.7), (trigger_page + 2, 0.3)]
        
        # Get all possible candidates from position-based DPF
        for pos in range(self.positions_tracked):
            for successor, count in self.page_metadata[trigger_page]['position_successors'][pos].items():
                # Calculate a confidence score based on position and count
                confidence = count * self.position_weights[pos]
                
                # Check if this candidate is already in the list
                found = False
                for i, (page, conf) in enumerate(candidates):
                    if page == successor:
                        # Update confidence if higher
                        if confidence > conf:
                            candidates[i] = (page, confidence)
                        found = True
                        break
                
                if not found:
                    candidates.append((successor, confidence))
        
        # Sort by confidence and limit to number of candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.num_candidates]
    
    def _calculate_pattern_similarity(self, matrix1, matrix2):
        """Calculate similarity between two transition patterns"""
        # Normalize matrices
        total1 = matrix1.sum()
        total2 = matrix2.sum()
        
        if total1 > 0 and total2 > 0:
            norm1 = matrix1 / total1
            norm2 = matrix2 / total2
            
            # Calculate cosine similarity
            flat1 = norm1.flatten()
            flat2 = norm2.flatten()
            
            # Avoid division by zero
            norm_flat1 = np.linalg.norm(flat1)
            norm_flat2 = np.linalg.norm(flat2)
            
            if norm_flat1 > 0 and norm_flat2 > 0:
                cos_sim = np.dot(flat1, flat2) / (norm_flat1 * norm_flat2)
                return cos_sim
        
        return 0.0
    
    def should_prefetch(self, trigger_page, candidate_page, confidence_threshold=0.2):
        """Determine if a prefetch should be issued based on confidence"""
        # Always prefetch sequential or first page after trigger
        if candidate_page == trigger_page + 1:
            return True
        
        # Check if we have metadata for this transition
        if trigger_page in self.page_metadata:
            for pos in range(self.positions_tracked):
                position_successors = self.page_metadata[trigger_page]['position_successors'][pos]
                if candidate_page in position_successors:
                    freq = position_successors[candidate_page]
                    total = sum(position_successors.values())
                    
                    if total > 0:
                        confidence = freq / total
                        if confidence >= confidence_threshold:
                            return True
        
        # Default policy
        return False
    
    def record_prefetch_result(self, position, was_hit):
        """Record the result of a prefetch for weight adaptation"""
        # Currently not used in the fixed weights approach
        pass
    
    def _update_position_weights(self):
        """Update position weights based on observed hit rates"""
        # Currently not used in the fixed weights approach
        pass
    
    def get_cross_page_predictions(self, trigger_page, trigger_offset):
        """Get predictions for cross-page access patterns based on boundary transitions"""
        predictions = []
        
        # Check if we have metadata for this page
        if trigger_page in self.page_metadata:
            # Check if we're near a page boundary
            is_near_boundary = (trigger_offset >= (64 - self.boundary_region_size) or 
                              trigger_offset < self.boundary_region_size)
            
            if is_near_boundary:
                # Get all boundary transitions for this page
                boundary_transitions = self.page_metadata[trigger_page]['boundary_transitions']
                
                # Find matching transitions for current offset
                matching_transitions = []
                for (trig_off, next_page, next_off), count in boundary_transitions.items():
                    if trig_off == trigger_offset and count >= self.boundary_confidence_threshold:
                        matching_transitions.append((next_page, next_off, count))
                
                # Sort by frequency and generate predictions
                matching_transitions.sort(key=lambda x: x[2], reverse=True)
                
                for next_page, next_off, _ in matching_transitions[:2]:  # Limit to top 2
                    # Generate prefetch address
                    prefetch_cache_line = (next_page << 6) | next_off  # Assuming 6-bit offset
                    prefetch_addr = prefetch_cache_line << 6  # Cache line address to byte address
                    predictions.append(prefetch_addr)
        
        return predictions
