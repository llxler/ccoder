import json
import os

def process_c_completion(completion, add_log=False):
    """
    单行补全版后处理（适合 C/Java 风格单行语句补全）：
    - 避开字符串/字符/注释
    - 顶层遇到：
        1) ';'  → 截断并返回（包含 ';'）
        2) '\n' → 截断并返回（不包含换行）
    """
    if not completion:
        return ""

    original = completion.strip()

    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    for i, ch in enumerate(completion):
        nxt = completion[i + 1] if i + 1 < len(completion) else ""

        # escape（仅对 string/char 生效）
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_string or in_char):
            escaped = True
            continue

        # line comment
        if in_line_comment:
            if ch == "\n":
                # 单行补全：到换行就结束（不包含换行）
                result = completion[:i].strip()
                if add_log and len(result) < len(original):
                    print(f"截断(换行): '{original}' -> '{result}'")
                return result
            continue

        # block comment
        if in_block_comment:
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

        # 单行：遇到换行就停（不含换行）
        if ch == "\n":
            result = completion[:i].strip()
            if add_log and len(result) < len(original):
                print(f"截断(换行): '{original}' -> '{result}'")
            return result

        # 单行语句：遇到 ';' 就停（含 ';'）
        if ch == ";":
            result = completion[: i + 1].strip()
            if add_log and len(result) < len(original):
                print(f"截断(;): '{original}' -> '{result}'")
            return result

    return original

def apply_processing(file_path):
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    modified_count = 0
    for item in data:
        if 'raw_res' in item:
             old_raw = item['raw_res']
             new_raw = process_c_completion(old_raw)
             if old_raw != new_raw:
                 item['raw_res'] = new_raw
                 modified_count += 1
        
        if 'prompt_res' in item:
             old_prompt = item['prompt_res']
             new_prompt = process_c_completion(old_prompt)
             if old_prompt != new_prompt:
                 item['prompt_res'] = new_prompt
                 modified_count += 1

        if 'gt' in item:
             old_gt = item['gt']
             new_gt = process_c_completion(old_gt)
             if old_gt != new_gt:
                 item['gt'] = new_gt
                 modified_count += 1
        
        

    # Generate new file path in the same directory
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    new_file_path = os.path.join(dir_name, f"{name}_processed{ext}")

    print(f"Saving changes to {new_file_path}. Modified {modified_count} fields.")
    
    with open(new_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    target_file = r"D:\Desktop\ccoder\results_gemini_java\gemini\java_gemini_langchain_result.json"
    if os.path.exists(target_file):
        apply_processing(target_file)
    else:
        print(f"File not found: {target_file}")
