"""
CruiseFetchPro预取器的优化预取生成模块
包含generate方法及其所有相关的辅助函数
"""

import numpy as np
import tensorflow as tf
import random
import time

def predict_prefetches(self, stream_id):
    """Make prefetch predictions with enhanced DPF"""
    if self.model is None or self.page_history[stream_id][-1] == 0:
        return default_predictions(self, stream_id)
    
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
    """创建智能缓存键，平衡精度和内存占用"""
    # 基本键：最近的历史和PC值
    recent_pages = tuple(self.page_history[stream_id][-4:])
    recent_offsets = tuple(self.offset_history[stream_id][-4:])
    pc_hash = self.last_pc[stream_id] % 1024  # 限制PC空间大小
    
    # 添加一个"模式指纹"，捕获更长历史的特征
    if len(self.page_history[stream_id]) > 4:
        full_history_hash = hash(tuple(self.page_history[stream_id])) % 128
    else:
        full_history_hash = 0
        
    return (recent_pages, recent_offsets, pc_hash, full_history_hash)

def optimize_model_for_inference(self):
    """优化TensorFlow模型以加速推理"""
    if self.model is None:
        return False
        
    try:
        # 使用tf.function预编译预测图
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, self.config.get('history_length', 16)], dtype=tf.int32),
            tf.TensorSpec(shape=[None, self.config.get('history_length', 16)], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[None, self.config.get('num_candidates', 4)], dtype=tf.float32)
        ])
        def optimized_predict(cluster_history, offset_history, pc, dpf):
            return self.model([cluster_history, offset_history, pc, dpf], training=False)
        
        # 使用编译后的函数替换模型的predict方法
        self.optimized_predict = optimized_predict
        print("Model optimized for inference")
        return True
    except Exception as e:
        print(f"Could not optimize model: {e}")
        return False

def _monitor_matrices_usage_optimized(self):
    """优化的矩阵使用监控"""
    current_size = len(self.offset_transition_matrices)
    
    # 仅当接近限制时执行清理检查
    if current_size >= self.matrices_max_entries * 0.9:  # 90%阈值
        # 批量清理，一次删除更多不常用矩阵
        sorted_matrices = sorted(
            self.matrices_access_timestamps.items(), 
            key=lambda x: x[1]
        )
        
        # 一次删除10%的最不常用矩阵
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
    """处理模型输出，生成预取地址"""
    # 获取触发页和偏移量
    trigger_page = self.page_history[stream_id][-1]
    trigger_offset = self.offset_history[stream_id][-1]
    
    # 获取候选索引和概率
    candidate_probs = tf.nn.softmax(candidate_logits[0]).numpy()
    valid_probs = candidate_probs[:-1]  
    top_candidate_indices = np.argsort(valid_probs)[-2:][::-1]
    
    # 获取偏移量索引
    offset_probs = tf.nn.softmax(offset_logits[0]).numpy()
    top_offset_indices = np.argsort(offset_probs)[-2:][::-1]
    
    # 获取候选页
    if self.metadata_manager:
        candidate_pages = self.metadata_manager.get_candidate_pages(trigger_page)
    else:
        # 默认顺序预测
        candidate_pages = [(trigger_page + 1, 100), (trigger_page + 2, 50)]
    
    # 检查跨页边界模式
    is_near_boundary = False
    if self.metadata_manager and hasattr(self.metadata_manager, 'boundary_region_size'):
        is_near_boundary = (trigger_offset >= (64 - self.metadata_manager.boundary_region_size) or 
                           trigger_offset < self.metadata_manager.boundary_region_size)
    
    # 预取地址列表
    prefetch_addresses = []
    
    # 如果靠近边界，首先检查跨页转换
    if is_near_boundary and self.metadata_manager and hasattr(self.metadata_manager, 'get_cross_page_predictions'):
        cross_page_predictions = self.metadata_manager.get_cross_page_predictions(trigger_page, trigger_offset)
        prefetch_addresses.extend(cross_page_predictions[:self.config.get('max_prefetches_per_id', 2)])
    
    # 如果仍有空间进行预取，使用常规预测逻辑
    if len(prefetch_addresses) < self.config.get('max_prefetches_per_id', 2):
        # 第一策略：顶级候选页与顶级偏移量
        if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
            selected_page, confidence = candidate_pages[top_candidate_indices[0]]
            if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                prefetch_cache_line = (selected_page << self.config.get('offset_bits', 6)) | top_offset_indices[0]
                prefetch_addr = prefetch_cache_line << 6
                
                if prefetch_addr not in prefetch_addresses:
                    prefetch_addresses.append(prefetch_addr)
        
        # 第二策略：第二候选页或第一候选页与第二偏移量
        if len(prefetch_addresses) < self.config.get('max_prefetches_per_id', 2):
            # 首先尝试第二候选页与顶级偏移量
            if len(top_candidate_indices) > 1 and top_candidate_indices[1] < len(candidate_pages):
                selected_page, confidence = candidate_pages[top_candidate_indices[1]]
                if not self.metadata_manager or self.metadata_manager.should_prefetch(trigger_page, selected_page):
                    prefetch_cache_line = (selected_page << self.config.get('offset_bits', 6)) | top_offset_indices[0]
                    prefetch_addr = prefetch_cache_line << 6
                    
                    if prefetch_addr not in prefetch_addresses:
                        prefetch_addresses.append(prefetch_addr)
            
            # 如果仍然没有第二个预取，尝试第一候选页与第二偏移量
            if len(prefetch_addresses) < self.config.get('max_prefetches_per_id', 2) and len(top_offset_indices) > 1:
                if len(top_candidate_indices) > 0 and top_candidate_indices[0] < len(candidate_pages):
                    selected_page, confidence = candidate_pages[top_candidate_indices[0]]
                    prefetch_cache_line = (selected_page << self.config.get('offset_bits', 6)) | top_offset_indices[1]
                    prefetch_addr = prefetch_cache_line << 6
                    
                    if prefetch_addr not in prefetch_addresses:
                        prefetch_addresses.append(prefetch_addr)
    
    # 记录预取结果以适应位置权重
    if self.metadata_manager and hasattr(self.metadata_manager, 'record_prefetch_result'):
        for i, addr in enumerate(prefetch_addresses[:2]):
            self.metadata_manager.record_prefetch_result(i, True)
    
    # 如果没有生成预取，回退到默认预测
    if not prefetch_addresses:
        return default_predictions(self, stream_id)
    
    # 限制每个ID的最大预取数
    return prefetch_addresses[:self.config.get('max_prefetches_per_id', 2)]

def generate_optimized(self, data):
    """
    优化版的generate方法，使用批处理和智能缓存进行加速
    
    Args:
        data: 内存访问轨迹，格式为[(instr_id, cycle_count, load_addr, load_ip, llc_hit), ...]
        
    Returns:
        预取列表，格式为[(instr_id, prefetch_addr), ...]
    """
    print("\n=== Using optimized CruiseFetchPro.generate ===")
    print(f"Model configuration: {self.config}")
    
    # 保存原始矩阵监控方法
    original_monitor_method = self._monitor_matrices_usage
    # 替换为优化版
    self._monitor_matrices_usage = lambda: _monitor_matrices_usage_optimized(self)
    
    # 尝试优化模型以加速推理
    model_optimized = False
    if self.model is not None:
        model_optimized = optimize_model_for_inference(self)
        print(f"Model optimization {'succeeded' if model_optimized else 'failed'}")
    
    prefetches = []
    batch_size = min(64, len(data) // 10 + 1)  # 动态批处理大小，不超过数据集大小的10%
    print(f"Using batch size: {batch_size}")
    
    # 初始化缓存系统
    if not hasattr(self, 'prediction_cache'):
        self.prediction_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.max_cache_size = 20000  # 缓存大小限制
    
    # 批处理相关变量
    batch_inputs = {
        'stream_ids': [],
        'instr_ids': [],
        'cache_keys': [],
        'cache_misses': []  # 需要预测的项目索引
    }
    
    batch_cluster_history = []
    batch_offset_history = []
    batch_pc = []
    batch_dpf = []
    
    # 计时器
    start_time = time.time()
    last_report_time = start_time
    report_interval = 10.0  # 每10秒报告一次进度
    
    try:
        for idx, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # 进度报告
            if start_time and last_report_time and time.time() - last_report_time > report_interval:
                elapsed = time.time() - start_time
                percent_done = (idx + 1) / len(data) * 100
                records_per_sec = (idx + 1) / elapsed
                eta_seconds = (len(data) - idx - 1) / records_per_sec if records_per_sec > 0 else 0
                
                print(f"Progress: {percent_done:.1f}% ({idx+1}/{len(data)}) - "
                      f"Speed: {records_per_sec:.1f} recs/s - "
                      f"ETA: {eta_seconds/60:.1f} min")
                last_report_time = time.time()
            
            # 跳过已达最大预取次数的指令
            if instr_id in self.stats['prefetches_per_instr'] and \
               self.stats['prefetches_per_instr'][instr_id] >= self.config.get('max_prefetches_per_id', 2):
                continue
            
            # 处理内存访问，更新状态
            self.process_trace_entry(instr_id, cycle_count, load_addr, load_ip, llc_hit)
            
            # 获取流ID
            stream_id = self.get_stream_id(load_ip)
            
            # 创建智能缓存键
            cache_key = create_cache_key(self, stream_id)
            
            # 添加到批处理
            batch_inputs['stream_ids'].append(stream_id)
            batch_inputs['instr_ids'].append(instr_id)
            batch_inputs['cache_keys'].append(cache_key)
            
            # 检查缓存中是否存在
            if cache_key in self.prediction_cache:
                # 缓存命中
                self.cache_stats['hits'] += 1
            else:
                # 缓存未命中 - 需要进行预测
                self.cache_stats['misses'] += 1
                batch_inputs['cache_misses'].append(len(batch_inputs['stream_ids']) - 1)
                
                # 准备此项目的输入
                cluster_hist = np.array([self.get_cluster_id(page) for page in self.page_history[stream_id]], dtype=np.int32)
                batch_cluster_history.append(cluster_hist)
                batch_offset_history.append(np.array(self.offset_history[stream_id], dtype=np.int32))
                batch_pc.append(np.array([self.last_pc[stream_id] % self.config.get('num_pcs', 4096)], dtype=np.int32))
                
                # 获取DPF向量
                if self.metadata_manager is not None:
                    dpf_vector = self.metadata_manager.get_dpf_vector(self.page_history[stream_id][-1])
                else:
                    dpf_vector = np.zeros(self.config.get('num_candidates', 4), dtype=np.float32)
                    
                # 确保正确形状
                if len(dpf_vector) > self.config.get('num_candidates', 4):
                    dpf_vector = dpf_vector[:self.config.get('num_candidates', 4)]
                elif len(dpf_vector) < self.config.get('num_candidates', 4):
                    dpf_vector = np.pad(dpf_vector, (0, self.config.get('num_candidates', 4) - len(dpf_vector)), 'constant')
                    
                batch_dpf.append(dpf_vector)
            
            # 当批处理达到大小或结束时进行处理
            if len(batch_inputs['stream_ids']) >= batch_size or idx == len(data) - 1:
                if batch_inputs['cache_misses']:
                    # 仅对缓存未命中的项目进行模型预测
                    miss_count = len(batch_inputs['cache_misses'])
                    if miss_count > 0 and self.model is not None:
                        try:
                            # 如果批量太大，分割成更小的批次处理
                            max_sub_batch = 32  # TensorFlow通常在较小批次上更稳定
                            for sub_batch_start in range(0, miss_count, max_sub_batch):
                                sub_batch_end = min(sub_batch_start + max_sub_batch, miss_count)
                                sub_batch_indices = batch_inputs['cache_misses'][sub_batch_start:sub_batch_end]
                                
                                if not sub_batch_indices:
                                    continue
                                
                                # 准备子批次的模型输入
                                sub_batch_indices_in_original = [batch_inputs['cache_misses'][i] for i in range(sub_batch_start, sub_batch_end)]
                                
                                model_inputs = [
                                    np.stack([batch_cluster_history[i-sub_batch_start] for i in range(sub_batch_start, sub_batch_end)]),
                                    np.stack([batch_offset_history[i-sub_batch_start] for i in range(sub_batch_start, sub_batch_end)]),
                                    np.stack([batch_pc[i-sub_batch_start] for i in range(sub_batch_start, sub_batch_end)]),
                                    np.stack([batch_dpf[i-sub_batch_start] for i in range(sub_batch_start, sub_batch_end)])
                                ]
                                
                                # 批量获取预测结果
                                if model_optimized and hasattr(self, 'optimized_predict'):
                                    # 使用优化的预测函数
                                    batch_candidate_logits, batch_offset_logits = self.optimized_predict(*model_inputs)
                                else:
                                    # 使用标准预测
                                    batch_candidate_logits, batch_offset_logits = self.model.predict(
                                        model_inputs, verbose=0
                                    )
                                
                                # 处理每个预测结果
                                for batch_idx, miss_idx in enumerate(sub_batch_indices_in_original):
                                    stream_id = batch_inputs['stream_ids'][miss_idx]
                                    cache_key = batch_inputs['cache_keys'][miss_idx]
                                    
                                    # 处理预测以获取预取地址
                                    predicted_prefetches = _process_model_outputs(
                                        self,
                                        batch_candidate_logits[batch_idx:batch_idx+1],
                                        batch_offset_logits[batch_idx:batch_idx+1],
                                        stream_id
                                    )
                                    
                                    # 计算预测置信度
                                    confidence = np.max(tf.nn.softmax(batch_candidate_logits[batch_idx]).numpy())
                                    
                                    # 只缓存高置信度预测
                                    if confidence > 0.5:  # 可配置的置信度阈值
                                        # 添加到缓存，使用LRU策略管理缓存大小
                                        if len(self.prediction_cache) >= self.max_cache_size:
                                            # 随机删除1%的缓存项以释放空间
                                            keys_to_remove = random.sample(
                                                list(self.prediction_cache.keys()), 
                                                max(1, int(len(self.prediction_cache) * 0.01))
                                            )
                                            for key in keys_to_remove:
                                                del self.prediction_cache[key]
                                        
                                        # 添加到缓存
                                        self.prediction_cache[cache_key] = predicted_prefetches
                        except Exception as e:
                            print(f"Error in batch prediction: {e}")
                            # 记录缓存未命中的项目，回退到默认预测
                            for miss_idx in batch_inputs['cache_misses']:
                                stream_id = batch_inputs['stream_ids'][miss_idx]
                                cache_key = batch_inputs['cache_keys'][miss_idx]
                                self.prediction_cache[cache_key] = default_predictions(self, stream_id)
                
                # 处理批处理中的所有项目以生成预取
                for idx, (stream_id, instr_id, cache_key) in enumerate(zip(
                    batch_inputs['stream_ids'], 
                    batch_inputs['instr_ids'],
                    batch_inputs['cache_keys']
                )):
                    # 从缓存或结果获取预测
                    if cache_key in self.prediction_cache:
                        predicted_prefetches = self.prediction_cache[cache_key]
                    else:
                        # 如果不在缓存中，使用默认预测
                        predicted_prefetches = default_predictions(self, stream_id)
                    
                    # 添加预取到输出，保持原始顺序
                    for prefetch_addr in predicted_prefetches:
                        if instr_id in self.stats['prefetches_per_instr'] and \
                           self.stats['prefetches_per_instr'][instr_id] >= self.config.get('max_prefetches_per_id', 2):
                            break
                        
                        prefetches.append((instr_id, prefetch_addr))
                        
                        # 更新统计信息
                        self.stats['prefetches_issued'] += 1
                        if instr_id not in self.stats['prefetches_per_instr']:
                            self.stats['prefetches_per_instr'][instr_id] = 0
                        self.stats['prefetches_per_instr'][instr_id] += 1
                
                # 重置批处理
                batch_inputs = {'stream_ids': [], 'instr_ids': [], 'cache_keys': [], 'cache_misses': []}
                batch_cluster_history = []
                batch_offset_history = []
                batch_pc = []
                batch_dpf = []
    
        # 打印缓存统计信息
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
    
    except Exception as e:
        print(f"Error in generate_optimized: {e}")
    
    finally:
        # 恢复原始矩阵监控方法
        self._monitor_matrices_usage = original_monitor_method
    
    return prefetches