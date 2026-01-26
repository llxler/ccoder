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

def main():
    parser = argparse.ArgumentParser(description="Calculate EM and ES from a JSON result file.")
    # TODO
    parser.add_argument("file_path", nargs='?', default="java_gpt5_graph_result_processed.json", help="Path to the JSON file containing results.")
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

if __name__ == "__main__":
    main()
