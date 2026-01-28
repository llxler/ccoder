#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
import Levenshtein
from openai import OpenAI
import concurrent.futures

# Import constants and path logic from utils, but we might need to override some
from utils import DS_BASE_DIR, BASE_DIR

# Initial Configuration
MODEL_API_NAME = "google/gemini-3-pro-preview"
MODEL_SHORT_NAME = "gemini" # Used for file naming

# TODO: Change this
FILE_PREFIX = "c"
RAGMETHOD = "graph"

API_KEY = os.getenv("OPENAI_API_KEY")

# Construct paths 
# We assume the prompts might serve as a base, or we might need to point to a specific prompt file.
# For this script, I'll define the result directories based on the new model name.
RESULT_DIR = os.path.join(BASE_DIR, f"results_{MODEL_SHORT_NAME}_{FILE_PREFIX}/{MODEL_SHORT_NAME}")
# Ensure directory exists
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR, exist_ok=True)

DS_FILE = os.path.join(DS_BASE_DIR, f"{FILE_PREFIX}_metadata.jsonl")
# Default prompt file - user might need to change this if they have a specific one
PT_FILE = os.path.join(DS_BASE_DIR, f"{FILE_PREFIX}_{RAGMETHOD}_prompt.jsonl") 

EVAL_FILE = os.path.join(RESULT_DIR, f"{FILE_PREFIX}_{MODEL_SHORT_NAME}_{RAGMETHOD}_eval.txt")
RESULT_FILE = os.path.join(RESULT_DIR, f"{FILE_PREFIX}_{MODEL_SHORT_NAME}_{RAGMETHOD}_result.json")
IMP_FILE = os.path.join(RESULT_DIR, f"{FILE_PREFIX}_{MODEL_SHORT_NAME}_{RAGMETHOD}_improved.json")

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    return {}

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

def get_openai_client():

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )

def generate_single_completion(client, prompt, model_name):
    # System prompt to guide the model to behave like a completion engine
    system_prompt = "You are a code completion engine. You must complete the code starting exactly where the user input ends. Do NOT repeat the user input. Output only the completion code. Keep the output concise, only generation one statement or block."
    
    try:
        # Single API call to reduce latency and thinking time
        # print(f"DEBUG PROMPT: {prompt[:100]!r}...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # max_tokens=128,
            stop=["\n\n"],
            # Removed extra reasoning loop and parameters
        )
        
        content = response.choices[0].message.content
        # print(f"DEBUG RAW: {content!r}")
        
        # Post-processing to remove prompt overlap if present
        # This occurs if the model repeats the last few lines of the prompt
        # Simple heuristic: Check if content starts with a significant suffix of prompt
        
        # Taking last 200 chars of prompt to check alignment
        prompt_suffix = prompt[-200:] if len(prompt) > 200 else prompt
        
        # We can try to find the longest common substring between prompt_suffix end and content start
        # But a simpler way: iterate backwards from prompt end
        
        # Simplified overlap check for common case
        # If content starts with the last non-whitespace line of prompt? 
        # Or if prompt ends with `lv_obj_t *page = ` and content starts with `lv_obj_t *page = `
        
        # Let's clean up whitespace for comparison
        clean_content = content.lstrip()
        
        # Check against lines from the prompt (reverse order)
        prompt_lines = prompt.split('\n')
        # Check up to last 5 lines
        for i in range(1, min(6, len(prompt_lines))):
            suffix = "\n".join(prompt_lines[-i:])
            if not suffix.strip(): continue
            
            if content.strip().startswith(suffix.strip()):
                # Found overlap
                # Remove it from content
                # We need to be careful matching exact string in content
                idx = content.find(suffix.strip())
                if idx != -1:
                    # Remove up to the end of that match
                    content = content[idx + len(suffix.strip()):]
                    break
        
        # If model repeated the very last partial line (which doesn't have \n)
        last_line = prompt_lines[-1]
        if last_line.strip() and content.strip().startswith(last_line.strip()):
             idx = content.find(last_line.strip())
             if idx != -1:
                content = content[idx + len(last_line.strip()):]

        return content

    except Exception as e:
        print(f"Error calling API: {e}")
        return ""

def generate_completion_batch(client, prompts, model_name, max_workers=4):
    results = []
    # Using ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {executor.submit(generate_single_completion, client, prompt, model_name): prompt for prompt in prompts}
        
        # We need to maintain order, so we'll wait for all and then map back or just append in order if we map carefully
        # Actually easier is to just map them 
        futures = [executor.submit(generate_single_completion, client, p, model_name) for p in prompts]
        
        for future in futures:
            try:
                res = future.result()
                print(f"GPT Output: {res!r}") # Debug output
                
                # Preprocess step similar to original logic
                processed_text = process_c_completion(res, add_log=True)
                results.append(processed_text)
            except Exception as e:
                print(f"Batch generation error: {e}")
                results.append("")
                
    return results

def process_c_completion(completion, add_log=False):
    if not completion:
        return ""
    
    original = completion.strip()
    
    in_string = False        
    in_char = False          
    in_line_comment = False  
    in_block_comment = False 
    escaped = False          
    
    for i, char in enumerate(completion):
        if char == '\\' and not escaped:
            escaped = True
            continue
        
        if escaped:
            escaped = False
            continue
            
        if in_line_comment:
            if char == '\n':
                in_line_comment = False
            continue
            
        if in_block_comment:
            if char == '*' and i+1 < len(completion) and completion[i+1] == '/':
                in_block_comment = False
                i += 1
            continue
            
        if in_string:
            if char == '"':
                in_string = False
            continue
            
        if in_char:
            if char == '\'':
                in_char = False
            continue
            
        if char == '/' and i+1 < len(completion):
            if completion[i+1] == '/':
                in_line_comment = True
                i += 1
                continue
            elif completion[i+1] == '*':
                in_block_comment = True
                i += 1
                continue
                
        if char == '"':
            in_string = True
            continue
            
        if char == '\'':
            in_char = True
            continue
            
        if char == ';':
            result = completion[:i+1].strip()  
            if add_log and len(result) < len(original):
                print(f"截断: '{original}' -> '{result}'")
            return result
        
        if char == '{':
            result = completion[:i].strip()
            if add_log and len(result) < len(original):
                print(f"截断 (brace): '{original}' -> '{result}'")
            return result
    
    return original

def compute_exact_match(prediction, ground_truth):
    return 1 if prediction.strip() == ground_truth.strip() else 0

def compute_edit_similarity(prediction, ground_truth):
    edit_distance = Levenshtein.distance(prediction.strip(), ground_truth.strip())
    max_len = max(len(prediction.strip()), len(ground_truth.strip()))
    
    if max_len == 0:  
        return 1.0

    similarity = 1.0 - (edit_distance / max_len)
    return similarity

def main():
    global PT_FILE
    parser = argparse.ArgumentParser(description=f"Evaluate {MODEL_SHORT_NAME} model code completion")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (concurrency level for API)")
    parser.add_argument("--prompt_file", type=str, default=PT_FILE, help="Path to prompt file")
    
    args = parser.parse_args()
    
    # Update PT_FILE if argument provided
    if args.prompt_file:
        PT_FILE = args.prompt_file
        
    config = load_config()
    
    if not os.path.exists(DS_FILE):
        print(f"Error: Dataset file not found {DS_FILE}")
        return


    os.makedirs(RESULT_DIR, exist_ok=True)
    
    dataset = load_jsonl(DS_FILE)
    prompts = load_jsonl(PT_FILE)
    
    # Testing: only use the first item TODO
    # print("TEST MODE: Processing first 5 items.")
    dataset = dataset[3250:] # Limit dataset to 5 items
    
    prompt_dict = {item.get("id", ""): item.get("prompt", "") for item in prompts}
    dataset_dict = {item.get("id", ""): item for item in dataset}
    
    client = get_openai_client()
    
    results = []
    improved_samples = [] 
    raw_exact_matches = []
    raw_edit_similarities = []
    prompt_exact_matches = []
    prompt_edit_similarities = []
    
    batch_size = args.batch_size
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size  
    
    print(f"Total samples: {num_samples}, Batch size (concurrency): {batch_size}")
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        
        current_batch = dataset[i:i+batch_size]
        batch_ids = [sample.get("id", "") for sample in current_batch]
        # batch_inputs = [sample.get("input", "") for sample in current_batch] # Original used input?
        # Actually original used generation from 'prompts' logic
        # Looking at original: 
        # batch_inputs = [sample.get("input", "") for sample in current_batch]
        # raw_preds = generate_completion_batch(model, tokenizer, batch_inputs, ...)
        
        # We need raw inputs from dataset
        batch_inputs = [sample.get("input", "") for sample in current_batch]
        batch_gts = [sample.get("gt", "") for sample in current_batch]
        
        batch_prompts = [prompt_dict.get(id_, "") for id_ in batch_ids]
        
        # 1. Generate based on raw input TODO: if need raw_preds
        # raw_preds = generate_completion_batch(client, batch_inputs, MODEL_API_NAME, max_workers=batch_size)
        # else just empty for now
        raw_preds = []
        for ii in range(batch_size):
            raw_preds.append("")
        
        # 2. Generate based on enhanced detailed prompt (if available)
        valid_prompts_indices = [idx for idx, p in enumerate(batch_prompts) if p]
        valid_prompts = [batch_prompts[idx] for idx in valid_prompts_indices]
        valid_prompt_ids = [batch_ids[idx] for idx in valid_prompts_indices]
        
        prompt_preds_map = {}
        
        if valid_prompts:
            prompt_preds = generate_completion_batch(client, valid_prompts, MODEL_API_NAME, max_workers=batch_size)
            for k, pid in enumerate(valid_prompt_ids):
                prompt_preds_map[pid] = prompt_preds[k]
        
        for j, sample_id in enumerate(batch_ids):
            gt = batch_gts[j]
            raw_pred = raw_preds[j]
            prompt_pred = prompt_preds_map.get(sample_id, "")
            
            raw_em = compute_exact_match(raw_pred, gt)
            raw_es = compute_edit_similarity(raw_pred, gt)
            
            raw_exact_matches.append(raw_em)
            raw_edit_similarities.append(raw_es)
            
            if prompt_pred:
                prompt_em = compute_exact_match(prompt_pred, gt)
                prompt_es = compute_edit_similarity(prompt_pred, gt)
                
                if raw_em == 0 and prompt_em == 1:
                    sample_data = dataset_dict.get(sample_id, {})
                    improved_sample = {
                        "id": sample_id,
                        "pkg": sample_data.get("pkg", ""),
                        "fpath": sample_data.get("fpath", ""),
                        "input": sample_data.get("input", ""),
                        "raw_res": raw_pred,
                        "prompt_res": prompt_pred,
                        "gt": gt
                    }
                    improved_samples.append(improved_sample)
                
                prompt_exact_matches.append(prompt_em)
                prompt_edit_similarities.append(prompt_es)
            
            result = {
                "id": sample_id,
                "raw_res": raw_pred,
                "prompt_res": prompt_pred,
                "gt": gt
            }
            
            results.append(result)
            
        # Optional: Save intermediate results
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    avg_raw_exact_match = np.mean(raw_exact_matches) if raw_exact_matches else 0.0
    avg_raw_edit_similarity = np.mean(raw_edit_similarities) if raw_edit_similarities else 0.0
    avg_prompt_exact_match = np.mean(prompt_exact_matches) if prompt_exact_matches else 0.0
    avg_prompt_edit_similarity = np.mean(prompt_edit_similarities) if prompt_edit_similarities else 0.0
    
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. Raw Input (raw_res):\n")
        f.write(f"   - Exact Match: {avg_raw_exact_match:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_raw_edit_similarity:.4f}\n\n")
        
        f.write("2. Prompt Input (prompt_res):\n")
        f.write(f"   - Exact Match: {avg_prompt_exact_match:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_prompt_edit_similarity:.4f}\n")
    
    if improved_samples:
        with open(IMP_FILE, "w", encoding="utf-8") as f:
            json.dump(improved_samples, f, ensure_ascii=False, indent=2)
        print(f"Found {len(improved_samples)} improved samples, saved to {IMP_FILE}")
    
    print(f"Evaluation complete. Summary: {EVAL_FILE}, Details: {RESULT_FILE}")

if __name__ == "__main__":
    main()
