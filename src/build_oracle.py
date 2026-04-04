
import json
import re
import os
import argparse
from pathlib import Path


def find_header_in_repo(header_name: str, repo_path: str) -> str | None:
    """在仓库中查找头文件的实际路径"""
    basename = os.path.basename(header_name)
    for root, _, files in os.walk(repo_path):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_path)
            if rel.endswith(header_name) or fname == basename:
                return full
    return None


def extract_macro_defs_from_header(header_path: str, macro_names: set) -> list[str]:
    """从头文件中提取指定宏名的 #define 定义（包括多行宏）"""
    if not os.path.isfile(header_path):
        return []

    try:
        content = open(header_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []

    defs = []
    for name in macro_names:
        # 匹配 #define NAME ... （包括多行 \ 续行）
        pattern = re.compile(
            rf"^[ \t]*#define\s+{re.escape(name)}\b[^\n]*(?:\\\n[^\n]*)*",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            defs.append(m.group().rstrip())
    return defs


def get_graph_identifiers(graph_data: dict) -> set:
    """获取代码图中所有已有的标识符集合"""
    idents = set()
    for fpath_key, file_info in graph_data.items():
        if isinstance(file_info, dict):
            idents.update(file_info.keys())
    return idents


def supplement_macros(
    input_code: str,
    pkg: str,
    repo_dir: str,
    graph_data: dict,
) -> str:
    """
    从 #include 头文件中提取 input 代码引用但代码图中不存在的宏定义。
    返回补充的宏定义文本（可能为空字符串）。
    """
    # 1. 从 input 提取大写风格标识符 → 可能是宏
    used_macros = set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", input_code))

    # 过滤常见的非宏大写词
    COMMON_NON_MACROS = {
        "NULL", "TRUE", "FALSE", "EOF", "FILE",
        "INT", "CHAR", "VOID", "LONG", "SHORT",
        "UINT", "SIZE", "MAX", "MIN",
    }
    used_macros -= COMMON_NON_MACROS

    if not used_macros:
        return ""

    # 2. 过滤掉代码图中已有的标识符
    graph_idents = get_graph_identifiers(graph_data) if graph_data else set()
    missing_macros = used_macros - graph_idents

    if not missing_macros:
        return ""

    # 3. 从 #include 头文件中查找
    repo_path = os.path.join(repo_dir, pkg)
    local_includes = re.findall(r'#include\s+"([^"]+)"', input_code)

    all_defs = []
    seen = set()
    for header in local_includes:
        hpath = find_header_in_repo(header, repo_path)
        if not hpath:
            continue
        defs = extract_macro_defs_from_header(hpath, missing_macros)
        for d in defs:
            if d not in seen:
                seen.add(d)
                all_defs.append(d)

    if not all_defs:
        return ""

    return "// Macro definitions from headers:\n" + "\n".join(all_defs)


def build_oracle_prompt(
    graph_prompt_text: str,
    macro_supplement: str,
) -> str:
    """将宏补充合并到 graph_prompt 的跨文件上下文部分"""
    if not macro_supplement:
        return graph_prompt_text

    parts = graph_prompt_text.split("<s>")

    if len(parts) >= 3:
        # parts: ['', cross-file, path, input]
        # 在跨文件上下文后追加宏定义
        parts[1] = parts[1].rstrip() + "\n\n" + macro_supplement + "\n"
        return "<s>".join(parts)
    else:
        # 没有跨文件部分，在最前面添加
        return "<s> " + macro_supplement + "\n" + graph_prompt_text


def main():
    parser = argparse.ArgumentParser(
        description="在 ONLY_DEF=False 的 graph_prompt 基础上补充宏定义，生成 Oracle prompt"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="metadata JSONL 文件路径（含 id, pkg, fpath, input, gt）",
    )
    parser.add_argument(
        "--graph_prompt",
        required=True,
        help="graph_prompt JSONL 文件路径（ONLY_DEF=False 版本）",
    )
    parser.add_argument(
        "--repo_dir",
        required=True,
        help="仓库根目录（包含各 repo 子目录）",
    )
    parser.add_argument(
        "--graph_dir",
        required=True,
        help="代码图 JSON 目录",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 Oracle prompt JSONL 路径",
    )
    args = parser.parse_args()

    # 加载 metadata
    samples = {}
    with open(args.metadata, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples[obj["id"]] = obj

    # 加载 graph_prompt
    graph_prompts = {}
    with open(args.graph_prompt, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            graph_prompts[obj["id"]] = obj["prompt"]

    # 加载代码图
    all_graphs: dict[str, dict] = {}
    graph_dir = Path(args.graph_dir)
    for gf in sorted(graph_dir.iterdir()):
        if gf.suffix == ".json":
            pkg = gf.stem
            with open(gf, encoding="utf-8") as f:
                all_graphs[pkg] = json.load(f)

    # 构建 Oracle prompt
    results = []
    stats = {"total": 0, "supplemented": 0, "total_macro_defs": 0}

    for sid in sorted(samples.keys()):
        s = samples[sid]
        gp = graph_prompts.get(sid, "")
        graph = all_graphs.get(s["pkg"], {})

        macro_supplement = supplement_macros(
            input_code=s["input"],
            pkg=s["pkg"],
            repo_dir=args.repo_dir,
            graph_data=graph,
        )

        oracle = build_oracle_prompt(gp, macro_supplement)

        stats["total"] += 1
        if macro_supplement:
            stats["supplemented"] += 1
            stats["total_macro_defs"] += macro_supplement.count("#define")

        results.append({"id": sid, "prompt": oracle})

    # 输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ 生成 {stats['total']} 条 Oracle prompt → {args.output}")
    print(f"  补充了宏定义的样本: {stats['supplemented']} ({stats['supplemented']/max(stats['total'],1)*100:.1f}%)")
    print(f"  补充的宏定义总数: {stats['total_macro_defs']}")


if __name__ == "__main__":
    main()