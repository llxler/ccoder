import json
import difflib

# ================= 配置项 =================
# 比较的目标字段，通常寻找 file1 此字段比 file2 好的区间
TARGET_FIELD = 'prompt_res' 
# 原始字段名，用于计算 Baseline
RAW_FIELD = 'raw_res'
# 最小区间长度
MIN_LENGTH = 200
# =========================================

def load_data_as_dict(file_path):
    """读取JSON并转换为 {id: item} 的字典格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['id']: item for item in data}
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

def calculate_metrics(pred, gt):
    """计算单个样本的 EM 和 ES"""
    if pred is None: pred = ""
    if gt is None: gt = ""
    
    # EM (Exact Match)
    em = 1.0 if pred.strip() == gt.strip() else 0.0
    
    # ES (Edit Similarity)
    es = difflib.SequenceMatcher(None, pred, gt).ratio()
    
    return em, es

def find_best_diff_sequence(file1_path, file2_path, min_len=MIN_LENGTH):
    print(f"正在加载文件...")
    print(f"File 1 (优): {file1_path}")
    print(f"File 2 (劣): {file2_path}")
    
    dict1 = load_data_as_dict(file1_path)
    dict2 = load_data_as_dict(file2_path)

    if not dict1 or not dict2:
        return

    # 对齐 ID
    common_ids = sorted(list(set(dict1.keys()) & set(dict2.keys())))
    n = len(common_ids)
    
    if n < min_len:
        print(f"错误: 公共 ID 数量 ({n}) 小于要求的最小间隔 ({min_len})。")
        return

    print(f"开始分析 {n} 条对齐数据...")

    diff_scores = []
    details = [] 

    for curr_id in common_ids:
        item1 = dict1[curr_id]
        item2 = dict2[curr_id]
        
        # Ground Truth
        gt = item1.get('gt', "")
        
        # 1. 获取 File 1 (Prompt) 结果
        pred1 = item1.get(TARGET_FIELD, "")
        em1, es1 = calculate_metrics(pred1, gt)

        # 2. 获取 File 2 (Prompt) 结果
        pred2 = item2.get(TARGET_FIELD, "")
        em2, es2 = calculate_metrics(pred2, gt)

        # 3. 获取 Raw (原始) 结果 - 默认从 File 1 获取
        raw = item1.get(RAW_FIELD, "")
        raw_em, raw_es = calculate_metrics(raw, gt)

        # 计算得分 (EM + ES)
        score1 = em1 + es1
        score2 = em2 + es2
        
        # 优化目标：寻找 File 1 综合得分比 File 2 高的区域
        # Diff > 0 表示 File 1 更好
        diff = score1 - score2
        
        diff_scores.append(diff)
        details.append({
            "id": curr_id,
            "em1": em1, "es1": es1,        # File 1
            "em2": em2, "es2": es2,        # File 2
            "raw_em": raw_em, "raw_es": raw_es # Raw
        })

    # === 核心算法：带长度限制的最大子段和 ===
    prefix_sum = [0.0] * (n + 1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + diff_scores[i]

    max_gain = -float('inf')
    best_start_idx = -1
    best_end_idx = -1
    
    min_prefix_val = float('inf')
    min_prefix_idx = -1

    for j in range(min_len, n + 1):
        valid_i = j - min_len
        if prefix_sum[valid_i] < min_prefix_val:
            min_prefix_val = prefix_sum[valid_i]
            min_prefix_idx = valid_i
            
        current_gain = prefix_sum[j] - min_prefix_val
        
        if current_gain > max_gain:
            max_gain = current_gain
            best_start_idx = min_prefix_idx
            best_end_idx = j - 1

    # === 输出统计结果 ===
    if best_start_idx != -1:
        start_real_id = common_ids[best_start_idx]
        end_real_id = common_ids[best_end_idx]
        length = best_end_idx - best_start_idx + 1
        
        # 提取区间数据
        subset = details[best_start_idx : best_end_idx + 1]
        
        # 计算平均值辅助函数
        def calc_avg(key): return sum(d[key] for d in subset) / length

        avg_em1 = calc_avg('em1')
        avg_es1 = calc_avg('es1')
        avg_em2 = calc_avg('em2')
        avg_es2 = calc_avg('es2')
        avg_raw_em = calc_avg('raw_em')
        avg_raw_es = calc_avg('raw_es')

        print("\n" + "="*80)
        print(f"【分析结果】 最佳区间分析报告")
        print("="*80)
        print(f"区间 ID 范围 : {start_real_id} ~ {end_real_id}")
        print(f"区间样本数量 : {length}")
        print(f"区间累计净胜分: {max_gain:.4f} (File1 - File2)")
        print("-" * 80)
        print(f"{'Metric':<10} | {'Raw (原始)':<15} | {'File 1 (本模型)':<18} | {'File 2 (对比)':<18} | {'本模型 vs 对比':<15}")
        print("-" * 80)
        print(f"{'Avg EM':<10} | {avg_raw_em:.2%}        | {avg_em1:.2%}           | {avg_em2:.2%}           | {avg_em1-avg_em2:+.2%}")
        print(f"{'Avg ES':<10} | {avg_raw_es:.4f}         | {avg_es1:.4f}            | {avg_es2:.4f}            | {avg_es1-avg_es2:+.4f}")
        print("-" * 80)
        
        # 简单的文字总结
        print("说明:")
        print("1. Raw:      该区间内 raw_res 的原始表现。")
        print("2. File 1:   你提供的第一个 Json 文件在该区间的表现。")
        print("3. File 2:   你提供的第二个 Json 文件在该区间的表现。")
        print("4. 最后一列: 正数表示 File 1 优于 File 2。")

    else:
        print("未找到满足条件的区间。")

# ================= 运行入口 =================
if __name__ == "__main__":
    # 请替换为你的实际文件路径
    # json1: 你希望胜出的文件
    # json2: 用来对比的文件
    json_file_1 = "/home/sub4-wy/lxl/ccoder/results/codellama7b/c_codellama7b_result.json"
    json_file_2 = "/home/sub4-wy/lxl/ccoder/results_rag/codellama7b/c_codellama7b_result.json"

    find_best_diff_sequence(json_file_1, json_file_2, min_len=200)