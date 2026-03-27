"""
三层分层抽样：语言 → 难度 bucket → repo 约束
- 语言层: 按 C/Java 比例分配总量 (C:333 + Java:71 = 404)
- 难度层: Java 用 external_api_calls_unique, C 用 #include 数量作代理
  - 分桶: [0,4), [4,8), [8,12), [12,+∞)
- Repo层: 同层内按 repo 比例分配, 小 repo 保护 (>=3条保底1个), 大 repo 上限 (≤40%)
- largest remainder method 保证总数精确
- 固定种子无放回抽样
"""

import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
TOTAL = 350         # 总抽样量
C_QUOTA = 250       # C 语言
JAVA_QUOTA = 100    # Java 语言

DIFFICULTY_BUCKETS = [(0, 4, "[0,4)"), (4, 8, "[4,8)"), (8, 12, "[8,12)"), (12, float("inf"), "[12,+∞)")]


def get_difficulty_bucket(value):
    for lo, hi, name in DIFFICULTY_BUCKETS:
        if lo <= value < hi:
            return name
    return "[12,+∞)"


def count_includes(code):
    """统计 C 代码中的 #include 数量 (作为 external_api_calls_unique 的代理)"""
    return len(re.findall(r'^\s*#include\s+[<"]', code, re.MULTILINE))


def largest_remainder_alloc(weights, total):
    """最大余数法分配，保证总和精确等于 total"""
    raw = {k: w / sum(weights.values()) * total for k, w in weights.items()}
    floored = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floored.values())
    # 按小数部分从大到小排序
    sorted_keys = sorted(raw.keys(), key=lambda k: raw[k] - floored[k], reverse=True)
    for i in range(remainder):
        floored[sorted_keys[i]] += 1
    return floored


def repo_allocate(repo_groups, stratum_quota, max_repo_pct=0.40):
    """
    在一个 (language, difficulty) stratum 内按 repo 比例分配。
    - 按比例分配 (largest remainder)
    - 大 repo 上限: 单 repo 不超过 max_repo_pct * stratum_quota
    - 总量严格等于 stratum_quota
    - 如果 repo 数 > quota，按大小排序取 top repos
    """
    if stratum_quota <= 0:
        return {}

    repo_sizes = {repo: len(recs) for repo, recs in repo_groups.items()}
    n_repos = len(repo_sizes)

    if n_repos == 0:
        return {}

    # 如果 repo 数超过 quota，只保留最大的 repos
    if n_repos > stratum_quota:
        sorted_repos = sorted(repo_sizes.keys(), key=lambda r: -repo_sizes[r])
        selected_repos = sorted_repos[:stratum_quota]
        repo_sizes = {r: repo_sizes[r] for r in selected_repos}

    # 用 largest remainder 分配
    alloc = largest_remainder_alloc(repo_sizes, stratum_quota)

    # 大 repo 上限
    max_per_repo = max(1, int(stratum_quota * max_repo_pct))
    excess = 0
    for repo in alloc:
        if alloc[repo] > max_per_repo:
            excess += alloc[repo] - max_per_repo
            alloc[repo] = max_per_repo

    # 确保不超过 repo 实际大小
    for repo in alloc:
        if alloc[repo] > repo_sizes[repo]:
            excess += alloc[repo] - repo_sizes[repo]
            alloc[repo] = repo_sizes[repo]

    # 把多出来的补给还有余量的 repo
    if excess > 0:
        for repo in sorted(alloc.keys(), key=lambda r: -repo_sizes[r]):
            if excess <= 0:
                break
            can_add = min(repo_sizes[repo] - alloc[repo], max_per_repo - alloc[repo])
            add = min(can_add, excess)
            if add > 0:
                alloc[repo] += add
                excess -= add

    return alloc


def kl_divergence(p_counts, q_counts):
    """KL(P||Q)"""
    all_keys = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    eps = 1e-10
    kl = 0.0
    for k in all_keys:
        p = p_counts.get(k, 0) / p_total + eps
        q = q_counts.get(k, 0) / q_total + eps
        kl += p * math.log(p / q)
    return kl


def verify_distribution(original, sampled, label):
    """验证三个维度的分布一致性"""
    print(f"\n  [{label}] 分布验证 (KL散度越小越好):")

    for dim_name, bucket_fn in [("pkg", lambda r: r["pkg"]),
                                 ("difficulty", lambda r: r["_difficulty_bucket"]),
                                 ("pkg×difficulty", lambda r: (r["pkg"], r["_difficulty_bucket"]))]:
        orig_c = Counter(bucket_fn(r) for r in original)
        samp_c = Counter(bucket_fn(r) for r in sampled)
        kl = kl_divergence(orig_c, samp_c)
        all_keys = set(orig_c) | set(samp_c)

        print(f"\n    {dim_name}: KL={kl:.6f} (共 {len(all_keys)} 类)")

        if dim_name != "pkg×difficulty":  # 联合太多不逐行打印
            orig_total = len(original)
            samp_total = len(sampled)
            for k in sorted(all_keys, key=lambda x: -orig_c.get(x, 0)):
                o_pct = orig_c.get(k, 0) / orig_total * 100
                s_pct = samp_c.get(k, 0) / samp_total * 100
                print(f"      {str(k):<40} 原始:{o_pct:5.1f}%  抽样:{s_pct:5.1f}%  差异:{s_pct-o_pct:+.1f}%")


def stratified_sample_3layer(meta_path, output_path, difficulty_values, lang_quota, lang_label):
    """
    三层分层抽样主函数
    meta_path: metadata jsonl 路径
    difficulty_values: dict {(pkg, fpath) -> external_api_calls_unique}
    lang_quota: 该语言的目标抽样数
    """
    random.seed(SEED)

    records = []
    with open(meta_path) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)

    # 为每条记录附加难度 bucket
    for r in records:
        key = (r["pkg"], r["fpath"])
        diff_val = difficulty_values.get(key, 0)
        r["_difficulty_value"] = diff_val
        r["_difficulty_bucket"] = get_difficulty_bucket(diff_val)

    # 第 2 层: 按难度 bucket 分配 quota
    diff_counts = Counter(r["_difficulty_bucket"] for r in records)
    diff_quotas = largest_remainder_alloc(diff_counts, lang_quota)

    print(f"\n{'='*70}")
    print(f"[{lang_label}] 源文件: {meta_path}")
    print(f"总数据量: {total}, 目标抽取: {lang_quota}")
    print(f"\n  难度 bucket 分配:")
    for bucket in ["[0,4)", "[4,8)", "[8,12)", "[12,+∞)"]:
        orig = diff_counts.get(bucket, 0)
        quota = diff_quotas.get(bucket, 0)
        print(f"    {bucket}: 原始 {orig} ({orig/total*100:.1f}%) → 抽取 {quota}")

    # 第 3 层: 在每个难度 bucket 内按 repo 分配
    sampled = []
    print(f"\n  详细分配 (difficulty → repo):")

    for bucket in ["[0,4)", "[4,8)", "[8,12)", "[12,+∞)"]:
        bucket_recs = [r for r in records if r["_difficulty_bucket"] == bucket]
        stratum_quota = diff_quotas.get(bucket, 0)

        if stratum_quota == 0 or not bucket_recs:
            continue

        repo_groups = defaultdict(list)
        for r in bucket_recs:
            repo_groups[r["pkg"]].append(r)

        repo_alloc = repo_allocate(repo_groups, stratum_quota)

        print(f"\n    {bucket} (quota={stratum_quota}):")
        for repo in sorted(repo_alloc.keys(), key=lambda r: -len(repo_groups[r])):
            n = repo_alloc[repo]
            if n > 0:
                selected = random.sample(repo_groups[repo], n)
                sampled.extend(selected)
                print(f"      {repo:<40} {len(repo_groups[repo]):>4} → {n}")

    # 按原始 id 排序
    sampled.sort(key=lambda x: x["id"])

    # 写出 (移除临时字段)
    with open(output_path, "w") as f:
        for r in sampled:
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\n  实际抽取: {len(sampled)} / {total} ({len(sampled)/total*100:.1f}%)")
    print(f"  输出: {output_path}")

    verify_distribution(records, sampled, lang_label)
    return sampled


def load_java_difficulty(tasks_path, meta_path):
    """从 AutoCodeEval_java_tasks_repo.jsonl 加载 Java 难度值"""
    tasks = {}
    with open(tasks_path) as f:
        for line in f:
            t = json.loads(line)
            tasks[(t["pkg"], t["fpath"])] = t["external_api_calls_unique"]
    return tasks


def compute_c_difficulty(meta_path):
    """从 C 源码中计算 #include 数量作为 external_api_calls_unique 的代理"""
    difficulty = {}
    with open(meta_path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["pkg"], r["fpath"])
            difficulty[key] = count_includes(r["input"])
    return difficulty


if __name__ == "__main__":
    base = Path(__file__).parent

    # ===== C: #include 数量作为难度代理 =====
    c_difficulty = compute_c_difficulty(str(base / "c_metadata.jsonl"))
    stratified_sample_3layer(
        meta_path=str(base / "c_metadata.jsonl"),
        output_path=str(base / "c_metadata_sample10.jsonl"),
        difficulty_values=c_difficulty,
        lang_quota=C_QUOTA,
        lang_label="C",
    )

    # ===== Java: 使用论文的 external_api_calls_unique =====
    java_difficulty = load_java_difficulty(
        str(base / "AutoCodeEval_java_tasks_repo.jsonl"),
        str(base / "java_metadata.jsonl"),
    )
    stratified_sample_3layer(
        meta_path=str(base / "java_metadata.jsonl"),
        output_path=str(base / "java_metadata_sample10.jsonl"),
        difficulty_values=java_difficulty,
        lang_quota=JAVA_QUOTA,
        lang_label="Java",
    )
