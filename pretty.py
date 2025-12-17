import json
import os
import re
from pathlib import Path

PATH_H_RE = re.compile(r"//\s*(.+\.h)\s*$")
PATH_C_RE = re.compile(r"//\s*path:\s*(.+\.c)\s*$")

def clean_line(line: str) -> str:
    # 去 LLM token
    line = line.replace("<s>", "")
    return line.rstrip("\n")

def process_prompt(prompt: str):
    lines = prompt.splitlines()

    files = {}
    cur_path = None
    cur_buf = []

    def flush():
        nonlocal cur_path, cur_buf
        if cur_path and cur_buf:
            files[cur_path] = "\n".join(cur_buf).strip() + "\n"
        cur_path = None
        cur_buf = []

    for raw in lines:
        line = clean_line(raw).strip()

        # 命中 .h
        m_h = PATH_H_RE.match(line)
        if m_h:
            flush()
            cur_path = m_h.group(1)
            cur_buf = []
            continue

        # 命中 .c
        m_c = PATH_C_RE.match(line)
        if m_c:
            flush()
            cur_path = m_c.group(1)
            cur_buf = []
            continue

        # 没有进入文件前 → 直接丢弃（这是你之前出 bug 的根源）
        if not cur_path:
            continue

        # 正常收集
        cur_buf.append(raw)

    flush()
    return files


def main():
    
    # JSONL_FILE = "/home/sub4-wy/lxl/ccoder/CEval/c_codellama7b_prompt_real.jsonl"
    JSONL_FILE = "/home/sub4-wy/lxl/ccoder/CEval/c_codellama7b_prompt.jsonl"
    
    OUT_DIR = "output_langchain_src"
    # OUT_DIR = "output_src"
    
    id_want = 100
    
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            
            if idx != id_want:
                continue
            
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if not prompt:
                continue

            file_map = process_prompt(prompt)

            for path, content in file_map.items():
                # 保留文件名
                out_path = Path(OUT_DIR) / Path(path).name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(out_path, "w", encoding="utf-8") as wf:
                    wf.write(content)

                print(f"[OK] {out_path}")
            break

if __name__ == "__main__":
    main()
