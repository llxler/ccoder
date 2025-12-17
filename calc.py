import json
from difflib import SequenceMatcher
from pathlib import Path

# ======================
# 配置
# ======================
JSON_PATH = "/home/sub4-wy/lxl/ccoder/results_rag/codellama7b/c_codellama7b_result.json"

# ======================
# 指标函数
# ======================
def exact_match(pred: str, gt: str) -> int:
    if pred is None:
        pred = ""
    if gt is None:
        gt = ""
    return int(pred.strip() == gt.strip())


def edit_similarity(pred: str, gt: str) -> float:
    if pred is None:
        pred = ""
    if gt is None:
        gt = ""
    return SequenceMatcher(None, pred.strip(), gt.strip()).ratio()


# ======================
# 主逻辑
# ======================
def main():
    path = Path(JSON_PATH)
    assert path.exists(), f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    raw_em = 0
    prompt_em = 0
    raw_es_sum = 0.0
    prompt_es_sum = 0.0

    raw_better_ids = []

    for item in data:
        _id = item["id"]
        raw = item.get("raw_res", "")
        prompt = item.get("prompt_res", "")
        gt = item.get("gt", "")

        # EM
        raw_em += exact_match(raw, gt)
        prompt_em += exact_match(prompt, gt)

        # ES
        raw_es = edit_similarity(raw, gt)
        prompt_es = edit_similarity(prompt, gt)

        raw_es_sum += raw_es
        prompt_es_sum += prompt_es

        # raw 优于 prompt
        if raw_es > prompt_es:
            raw_better_ids.append(_id)

    # ======================
    # 输出结果
    # ======================
    print("=" * 60)
    print(f"Total samples: {total}")
    print()
    print("Exact Match (EM)")
    print(f"  raw_res    : {raw_em}/{total} = {raw_em / total:.4f}")
    print(f"  prompt_res : {prompt_em}/{total} = {prompt_em / total:.4f}")
    print()
    print("Edit Similarity (ES)")
    print(f"  raw_res    : {raw_es_sum / total:.4f}")
    print(f"  prompt_res : {prompt_es_sum / total:.4f}")
    print()
    print(f"raw_res ES > prompt_res ES: {len(raw_better_ids)} samples")
    print("IDs:", raw_better_ids)
    print("=" * 60)


if __name__ == "__main__":
    main()
