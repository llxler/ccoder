import json
import argparse
import sys
import os
import numpy as np
import Levenshtein

def compute_exact_match(prediction, ground_truth):
    if not isinstance(prediction, str):
        prediction = ""
    if not isinstance(ground_truth, str):
        ground_truth = ""
    return 1 if prediction.strip() == ground_truth.strip() else 0

def compute_edit_similarity(prediction, ground_truth):
    if not isinstance(prediction, str):
        prediction = ""
    if not isinstance(ground_truth, str):
        ground_truth = ""
        
    edit_distance = Levenshtein.distance(prediction.strip(), ground_truth.strip())
    max_len = max(len(prediction.strip()), len(ground_truth.strip()))
    
    if max_len == 0:  
        return 1.0

    similarity = 1.0 - (edit_distance / max_len)
    return similarity

def process_c_completion(completion: str, add_log: bool = False) -> str:
    """
    只按换行截断（; 不截断），并对注释做状态避开：
    - 顶层遇到 '\n' 截断
    - // 行注释内遇到 '\n' 截断
    - /* */ 块注释内遇到 '\n' 也截断
    """
    if not completion or not isinstance(completion, str):
        return ""

    original = completion.strip()

    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    n = len(completion)

    def _log(reason, result):
        if add_log and len(result) < len(original):
            print(f"截断({reason}): '{original}' -> '{result}'")

    for i, ch in enumerate(completion):
        nxt = completion[i + 1] if i + 1 < n else ""

        # escape handling（仅对 string/char 生效）
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_string or in_char):
            escaped = True
            continue

        # line comment
        if in_line_comment:
            if ch == "\n":
                result = completion[: i + 1]
                _log("换行/行注释", result)
                return result
            continue

        # block comment: 注释内遇到换行也截断
        if in_block_comment:
            if ch == "\n":
                result = completion[: i + 1]
                _log("换行/块注释", result)
                return result
            if ch == "*" and nxt == "/":
                in_block_comment = False
            continue

        # string
        if in_string:
            if ch == '"' and not escaped:
                in_string = False
            continue

        # char
        if in_char:
            if ch == "'" and not escaped:
                in_char = False
            continue

        # entering comments
        if ch == "/" and nxt == "/":
            in_line_comment = True
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            continue

        # entering string/char
        if ch == '"':
            in_string = True
            continue
        if ch == "'":
            in_char = True
            continue

        # 顶层：遇到换行就停（✅ 保留换行）
        if ch == "\n":
            result = completion[: i + 1]
            _log("换行", result)
            return result

        # ';' 不处理（不截断）

    return original

def main():
    parser = argparse.ArgumentParser(description="Calculate EM and ES from a JSON result file.")
    # TODO
    parser.add_argument("file_path", nargs='?', default=r"/home/sub4-wy/lxl/ccoder/results_claude_java/claude/java_claude_langchain_result.json", help="Path to the JSON file containing results.")
    args = parser.parse_args()

    file_path = args.file_path
    
    if not os.path.exists(file_path):
        # Try looking in the same directory as the script if relative path fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)
        
        if not os.path.exists(file_path):
            print(f"Error: File '{args.file_path}' not found.")
            return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    raw_exact_matches = []
    raw_edit_similarities = []
    prompt_exact_matches = []
    prompt_edit_similarities = []

    for item in data:
        gt = item.get("gt", "")
        raw_res = item.get("raw_res", "")
        prompt_res = item.get("prompt_res", "")
        
        gt = process_c_completion(gt)
        if gt.rstrip().endswith('{'):
                 gt = gt.rstrip()[:-1].rstrip()
    
        raw_res = process_c_completion(raw_res)
        prompt_res = process_c_completion(prompt_res)

        # Raw Result Evaluation
        raw_em = compute_exact_match(raw_res, gt)
        raw_es = compute_edit_similarity(raw_res, gt)
        raw_exact_matches.append(raw_em)
        raw_edit_similarities.append(raw_es)

        # Prompt Result Evaluation
        # Check if prompt_res exists (some entries might not have it if prompt failed or wasn't used)
        # But we generally equate it with raw_res structure.
        # If prompt_res is None, we treat it as empty string
        prompt_em = compute_exact_match(prompt_res, gt)
        prompt_es = compute_edit_similarity(prompt_res, gt)
        prompt_exact_matches.append(prompt_em)
        prompt_edit_similarities.append(prompt_es)

    avg_raw_em = np.mean(raw_exact_matches) if raw_exact_matches else 0.0
    avg_raw_es = np.mean(raw_edit_similarities) if raw_edit_similarities else 0.0
    
    avg_prompt_em = np.mean(prompt_exact_matches) if prompt_exact_matches else 0.0
    avg_prompt_es = np.mean(prompt_edit_similarities) if prompt_edit_similarities else 0.0

    print("Evaluation Summary")
    print("==================================================")
    print()
    print("1. Raw Input (raw_res):")
    print(f"   - Exact Match: {avg_raw_em:.4f}")
    print(f"   - Edit Similarity: {avg_raw_es:.4f}")
    print()
    print("2. Prompt Input (prompt_res):")
    print(f"   - Exact Match: {avg_prompt_em:.4f}")
    print(f"   - Edit Similarity: {avg_prompt_es:.4f}")
    
    # Save results to a .txt file in the same directory
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Summary\n")
        f.write("==================================================\n")
        f.write("\n")
        f.write("1. Raw Input (raw_res):\n")
        f.write(f"   - Exact Match: {avg_raw_em:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_raw_es:.4f}\n")
        f.write("\n")
        f.write("2. Prompt Input (prompt_res):\n")
        f.write(f"   - Exact Match: {avg_prompt_em:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_prompt_es:.4f}\n")

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
