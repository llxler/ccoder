#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate CEval with an agent-style code completion workflow.

Compared with `evaluation_gpt5.py`, this script:
1. uses the raw dataset `input` directly instead of any RAG prompt file;
2. lets the model act like a standard repository-aware agent through tool calls;
3. evaluates the final completion with exact match and edit similarity.

The agent can optionally inspect a local repository root such as `c_repo/` or
`java_repo/`. Repository paths are configurable via CLI flags.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "CEval"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 180

SYSTEM_PROMPT_TEMPLATE = """You are a repository-aware code completion agent.

Your job is to continue the user's code prefix at the cursor.

Rules:
- Output only the completion code.
- Do not repeat the user's prompt.
- Do not add explanations, markdown fences, or commentary.
- Keep the completion limited to the immediate next statement, declaration fragment, or block header.
- Use tools only when they help you inspect the repository and resolve ambiguity.
- If tools are available, prefer checking the current file context before guessing.

Current language: {language}
Current repository root: {repo_root}
Current target file: {target_file}
"""


@dataclass(frozen=True)
class Sample:
    id: int
    pkg: str
    fpath: str
    input: str
    gt: str


@dataclass
class AgentConfig:
    model: str
    base_url: str
    api_key: str
    language: str
    repo_root: Path
    max_steps: int
    max_output_tokens: int
    timeout_seconds: int
    reasoning_effort: Optional[str]
    allow_tools: bool


class ResponsesAPIClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def create_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self.base_url}/responses",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib_request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Responses API HTTP {exc.code}: {error_body}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Responses API request failed: {exc}") from exc


class RepoTools:
    def __init__(self, repo_root: Path, sample: Sample) -> None:
        self.repo_root = repo_root
        self.sample = sample
        self.target_file = self._resolve_target_file()

    def specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "get_target_file_context",
                "description": (
                    "Read the current target file around the cursor. "
                    "Use this first when you need the local context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "before_lines": {
                            "type": "integer",
                            "description": "How many lines before the cursor to show.",
                            "minimum": 20,
                            "maximum": 300,
                        },
                        "after_lines": {
                            "type": "integer",
                            "description": "How many lines after the cursor to show.",
                            "minimum": 0,
                            "maximum": 160,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file from the repository by relative path and line range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "File path relative to repository root.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based starting line number.",
                            "minimum": 1,
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based ending line number.",
                            "minimum": 1,
                        },
                    },
                    "required": ["relative_path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "search_code",
                "description": "Search repository code text with ripgrep-like behavior.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Literal or regex code search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matches to return.",
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_dir",
                "description": "List files and directories under a repository path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "Directory path relative to repository root.",
                        },
                        "max_entries": {
                            "type": "integer",
                            "description": "Maximum entries to return.",
                            "minimum": 1,
                            "maximum": 200,
                        },
                    },
                    "additionalProperties": False,
                },
            },
        ]

    def invoke(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_target_file_context":
            return self.get_target_file_context(
                before_lines=int(arguments.get("before_lines", 80)),
                after_lines=int(arguments.get("after_lines", 40)),
            )
        if name == "read_file":
            return self.read_file(
                relative_path=str(arguments.get("relative_path", "")),
                start_line=int(arguments.get("start_line", 1)),
                end_line=int(arguments.get("end_line", 200)),
            )
        if name == "search_code":
            return self.search_code(
                query=str(arguments.get("query", "")),
                max_results=int(arguments.get("max_results", 10)),
            )
        if name == "list_dir":
            return self.list_dir(
                relative_path=str(arguments.get("relative_path", ".")),
                max_entries=int(arguments.get("max_entries", 100)),
            )
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def _resolve_target_file(self) -> Optional[Path]:
        candidates = [self.repo_root / self.sample.fpath]
        if self.sample.pkg:
            candidates.append(self.repo_root / self.sample.pkg / self.sample.fpath)

        for path in candidates:
            if path.exists() and path.is_file():
                return path.resolve()

        target_name = Path(self.sample.fpath).name
        if not self.repo_root.exists():
            return None

        try:
            for path in self.repo_root.rglob(target_name):
                try:
                    if path.is_file() and str(path).endswith(self.sample.fpath):
                        return path.resolve()
                except OSError:
                    continue
        except OSError:
            return None

        return None

    def _ensure_repo_path(self, relative_path: str) -> Path:
        path = (self.repo_root / relative_path).resolve()
        repo_root_resolved = self.repo_root.resolve()
        if path == repo_root_resolved or repo_root_resolved in path.parents:
            return path
        raise ValueError(f"Path escapes repository root: {relative_path}")

    def list_dir(self, relative_path: str = ".", max_entries: int = 100) -> Dict[str, Any]:
        if not self.repo_root.exists():
            return {"ok": False, "error": f"Repository root does not exist: {self.repo_root}"}

        max_entries = max(1, min(max_entries, 200))
        try:
            path = self._ensure_repo_path(relative_path or ".")
            if not path.exists():
                return {"ok": False, "error": f"Path does not exist: {relative_path}"}
            if not path.is_dir():
                return {"ok": False, "error": f"Path is not a directory: {relative_path}"}

            entries = []
            for idx, entry in enumerate(sorted(path.iterdir(), key=lambda item: item.name)):
                if idx >= max_entries:
                    break
                entries.append(
                    {
                        "name": entry.name,
                        "relative_path": str(entry.relative_to(self.repo_root)),
                        "type": "dir" if entry.is_dir() else "file",
                    }
                )

            truncated = len(list(path.iterdir())) > max_entries
            return {
                "ok": True,
                "path": str(path.relative_to(self.repo_root)),
                "entries": entries,
                "truncated": truncated,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def read_file(self, relative_path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
        if not self.repo_root.exists():
            return {"ok": False, "error": f"Repository root does not exist: {self.repo_root}"}

        start_line = max(1, start_line)
        end_line = max(start_line, min(end_line, start_line + 399))

        try:
            path = self._ensure_repo_path(relative_path)
            if not path.exists() or not path.is_file():
                return {"ok": False, "error": f"File does not exist: {relative_path}"}

            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start_idx = start_line - 1
            end_idx = min(len(lines), end_line)
            sliced = lines[start_idx:end_idx]

            return {
                "ok": True,
                "path": str(path.relative_to(self.repo_root)),
                "start_line": start_line,
                "end_line": end_idx,
                "content": format_numbered_lines(sliced, start_line),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def search_code(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        if not self.repo_root.exists():
            return {"ok": False, "error": f"Repository root does not exist: {self.repo_root}"}
        if not query.strip():
            return {"ok": False, "error": "Empty query"}

        max_results = max(1, min(max_results, 50))
        rg = shutil.which("rg")

        try:
            if rg is not None:
                cmd = [
                    rg,
                    "-n",
                    "--no-heading",
                    "--color",
                    "never",
                    "-m",
                    str(max_results),
                    query,
                    str(self.repo_root),
                ]
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                raw_lines = completed.stdout.splitlines()
            else:
                raw_lines = self._python_search_code(query, max_results)

            matches = []
            for line in raw_lines[:max_results]:
                parsed = parse_search_line(line, self.repo_root)
                if parsed is not None:
                    matches.append(parsed)

            return {
                "ok": True,
                "query": query,
                "matches": matches,
                "count": len(matches),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _python_search_code(self, query: str, max_results: int) -> List[str]:
        pattern = re.compile(query)
        collected: List[str] = []
        for path in self.repo_root.rglob("*"):
            if len(collected) >= max_results:
                break
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for idx, line in enumerate(lines, start=1):
                if pattern.search(line):
                    collected.append(f"{path}:{idx}:{line}")
                    if len(collected) >= max_results:
                        break
        return collected

    def get_target_file_context(self, before_lines: int = 80, after_lines: int = 40) -> Dict[str, Any]:
        if self.target_file is None:
            return {
                "ok": False,
                "error": f"Could not resolve target file for {self.sample.fpath}",
            }

        before_lines = max(20, min(before_lines, 300))
        after_lines = max(0, min(after_lines, 160))

        try:
            file_text = self.target_file.read_text(encoding="utf-8", errors="replace")
            cursor_index = locate_prompt_in_text(file_text, self.sample.input)
            if cursor_index is None:
                return {
                    "ok": False,
                    "error": "Could not align the raw prompt with the target file.",
                    "target_file": str(self.target_file.relative_to(self.repo_root)),
                }

            content, cursor_line = render_cursor_context(
                file_text=file_text,
                cursor_index=cursor_index,
                before_lines=before_lines,
                after_lines=after_lines,
            )

            return {
                "ok": True,
                "target_file": str(self.target_file.relative_to(self.repo_root)),
                "cursor_line": cursor_line,
                "content": content,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def load_jsonl(file_path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            samples.append(
                Sample(
                    id=int(obj["id"]),
                    pkg=str(obj.get("pkg", "")),
                    fpath=str(obj.get("fpath", "")),
                    input=str(obj.get("input", "")),
                    gt=str(obj.get("gt", "")),
                )
            )
    return samples


def save_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CEval with an agent workflow.")
    parser.add_argument("--lang", choices=["c", "java"], default="c", help="Dataset language.")
    parser.add_argument("--dataset_file", type=str, default="", help="Optional dataset jsonl path.")
    parser.add_argument("--repo_root", type=str, default="", help="Repository root for agent tools.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Responses API model.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="Responses API base URL.")
    parser.add_argument("--api_key_env", type=str, default=DEFAULT_API_KEY_ENV, help="API key env var name.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of parallel samples.")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum agent turns.")
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Responses API max_output_tokens.",
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["minimal", "low", "medium", "high"],
        default="medium",
        help="Reasoning effort for compatible models.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing result file.")
    parser.add_argument(
        "--disable_tools",
        action="store_true",
        help="Disable repository tools and run raw-prompt agent only.",
    )
    return parser.parse_args()


def resolve_default_dataset(lang: str) -> Path:
    return DATASET_DIR / f"{lang}_metadata.jsonl"


def resolve_repo_root(lang: str, raw_repo_root: str) -> Path:
    repo_name = raw_repo_root or f"{lang}_repo"
    raw_path = Path(repo_name)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        PROJECT_ROOT / raw_path,
        PROJECT_ROOT.parent / raw_path,
        DATASET_DIR / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def build_result_paths(language: str, model_name: str) -> Dict[str, Path]:
    model_short = sanitize_name(model_name)
    result_dir = PROJECT_ROOT / f"results_agent_{language}" / model_short
    return {
        "result_dir": result_dir,
        "result_file": result_dir / f"{language}_{model_short}_agent_result.json",
        "eval_file": result_dir / f"{language}_{model_short}_agent_eval.txt",
    }


def load_existing_results(result_file: Path) -> Dict[int, Dict[str, Any]]:
    if not result_file.exists():
        return {}
    try:
        with result_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {int(item["id"]): item for item in data}
    except Exception:
        return {}


def format_numbered_lines(lines: Sequence[str], start_line: int) -> str:
    if not lines:
        return ""
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(
        f"{start_line + idx:>{width}}| {line}"
        for idx, line in enumerate(lines)
    )


def parse_search_line(line: str, repo_root: Path) -> Optional[Dict[str, Any]]:
    match = re.match(r"^(.*?):(\d+):(.*)$", line)
    if not match:
        return None
    abs_path = Path(match.group(1))
    try:
        relative_path = str(abs_path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        relative_path = str(abs_path)
    return {
        "relative_path": relative_path,
        "line": int(match.group(2)),
        "text": match.group(3),
    }


def locate_prompt_in_text(file_text: str, prompt_text: str) -> Optional[int]:
    if not prompt_text:
        return None

    direct_idx = file_text.rfind(prompt_text)
    if direct_idx != -1:
        return direct_idx + len(prompt_text)

    prompt_lines = prompt_text.splitlines()
    for count in (12, 10, 8, 6, 4, 3, 2, 1):
        if len(prompt_lines) < count:
            continue
        suffix = "\n".join(prompt_lines[-count:])
        idx = file_text.rfind(suffix)
        if idx != -1:
            return idx + len(suffix)

    for width in (1000, 800, 600, 400, 300, 200, 120, 80):
        if len(prompt_text) < width:
            continue
        suffix = prompt_text[-width:]
        idx = file_text.rfind(suffix)
        if idx != -1:
            return idx + len(suffix)

    return None


def render_cursor_context(file_text: str, cursor_index: int, before_lines: int, after_lines: int) -> tuple[str, int]:
    lines = file_text.splitlines()
    prefix = file_text[:cursor_index]
    cursor_line_index = prefix.count("\n")
    line_start = prefix.rfind("\n")
    column = cursor_index if line_start == -1 else cursor_index - line_start - 1

    start_line_index = max(0, cursor_line_index - before_lines)
    end_line_index = min(len(lines), cursor_line_index + after_lines + 1)

    rendered: List[str] = []
    width = len(str(end_line_index))

    for idx in range(start_line_index, end_line_index):
        line = lines[idx]
        if idx == cursor_line_index:
            split_at = max(0, min(column, len(line)))
            line = f"{line[:split_at]}<CURSOR>{line[split_at:]}"
        rendered.append(f"{idx + 1:>{width}}| {line}")

    return "\n".join(rendered), cursor_line_index + 1


def remove_prompt_overlap(prompt: str, completion: str) -> str:
    if not completion:
        return ""

    content = completion
    prompt_lines = prompt.splitlines()
    for line_count in range(min(6, len(prompt_lines)), 0, -1):
        suffix = "\n".join(prompt_lines[-line_count:]).strip()
        if suffix and content.strip().startswith(suffix):
            start = content.find(suffix)
            if start != -1:
                content = content[start + len(suffix):]
                break

    last_line = prompt_lines[-1].strip() if prompt_lines else ""
    if last_line and content.strip().startswith(last_line):
        start = content.find(last_line)
        if start != -1:
            content = content[start + len(last_line):]

    return content


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    match = re.match(r"^```[^\n]*\n(.*)\n```$", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def line_needs_continuation(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped:
        return False
    if stripped.endswith(("=", "(", "[", ".", "->", "+", "-", "*", "/", "%", "&", "|", "^", "?", ":", "\\")):
        return True
    return bool(re.search(r"\b(new|return|case)\s*$", stripped))


def truncate_completion(completion: str) -> str:
    if not completion:
        return ""

    original = completion.strip()
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    for idx, char in enumerate(completion):
        next_char = completion[idx + 1] if idx + 1 < len(completion) else ""

        if escaped:
            escaped = False
            continue

        if char == "\\" and (in_string or in_char):
            escaped = True
            continue

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
            continue

        if in_string:
            if char == "\"":
                in_string = False
            continue

        if in_char:
            if char == "'":
                in_char = False
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            continue

        if char == "\"":
            in_string = True
            continue

        if char == "'":
            in_char = True
            continue

        if char == ";":
            return completion[: idx + 1].strip()

        if char == "{":
            line_start = completion.rfind("\n", 0, idx) + 1
            current_line_prefix = completion[line_start:idx]
            if current_line_prefix.strip():
                return completion[: idx + 1].strip()
            if line_start > 0:
                return completion[: line_start - 1].strip()
            return completion[: idx + 1].strip()

    lines = [line.rstrip() for line in original.splitlines() if line.strip()]
    if not lines:
        return ""

    collected = [lines[0]]
    for line in lines[1:]:
        if line_needs_continuation(collected[-1]):
            collected.append(line)
            continue
        break

    return "\n".join(collected).strip()


def postprocess_completion(prompt: str, raw_output: str) -> str:
    text = strip_code_fences(raw_output)
    text = remove_prompt_overlap(prompt, text)
    return truncate_completion(text)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip() == ground_truth.strip() else 0.0


def compute_edit_similarity(prediction: str, ground_truth: str) -> float:
    pred = prediction.strip()
    gt = ground_truth.strip()
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(pred, gt)
    return 1.0 - (distance / max_len)


def build_initial_payload(
    config: AgentConfig,
    sample: Sample,
    tool_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config.model,
        "input": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(
                    language=config.language,
                    repo_root=str(config.repo_root),
                    target_file=sample.fpath,
                ),
            },
            {
                "role": "user",
                "content": sample.input,
            },
        ],
        "max_output_tokens": config.max_output_tokens,
        "parallel_tool_calls": False,
    }
    if tool_specs:
        payload["tools"] = tool_specs
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    return payload


def build_follow_up_payload(
    config: AgentConfig,
    previous_response_id: str,
    tool_outputs: List[Dict[str, Any]],
    tool_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config.model,
        "previous_response_id": previous_response_id,
        "input": tool_outputs,
        "max_output_tokens": config.max_output_tokens,
        "parallel_tool_calls": False,
    }
    if tool_specs:
        payload["tools"] = tool_specs
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    return payload


def extract_response_text(response: Dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(content["text"])
        elif item.get("type") in {"output_text", "text"} and item.get("text"):
            chunks.append(item["text"])
    return "\n".join(chunks).strip()


def extract_function_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in response.get("output", []):
        if item.get("type") == "function_call":
            calls.append(
                {
                    "call_id": item.get("call_id") or item.get("id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                }
            )
    return calls


def parse_function_arguments(raw_arguments: Any) -> Dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        return json.loads(raw_arguments)
    return {}


def run_agent_completion(
    client: ResponsesAPIClient,
    config: AgentConfig,
    sample: Sample,
) -> Dict[str, Any]:
    repo_tools = RepoTools(config.repo_root, sample)
    tool_specs = repo_tools.specs() if config.allow_tools else []

    response = client.create_response(build_initial_payload(config, sample, tool_specs))
    tool_trace: List[Dict[str, Any]] = []
    raw_output = ""
    error_message = ""

    for step in range(1, config.max_steps + 1):
        function_calls = extract_function_calls(response)
        if not function_calls:
            raw_output = extract_response_text(response)
            break

        tool_outputs = []
        for call in function_calls:
            call_name = call.get("name") or ""
            call_id = call.get("call_id") or ""
            try:
                arguments = parse_function_arguments(call.get("arguments"))
                tool_result = repo_tools.invoke(call_name, arguments)
            except Exception as exc:
                arguments = {}
                tool_result = {"ok": False, "error": str(exc)}

            tool_trace.append(
                {
                    "step": step,
                    "tool": call_name,
                    "arguments": arguments,
                    "ok": bool(tool_result.get("ok", False)),
                }
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        response = client.create_response(
            build_follow_up_payload(
                config=config,
                previous_response_id=response["id"],
                tool_outputs=tool_outputs,
                tool_specs=tool_specs,
            )
        )
    else:
        raw_output = extract_response_text(response)
        error_message = f"Reached max_steps={config.max_steps} before the agent stopped."

    completion = postprocess_completion(sample.input, raw_output)
    exact_match = compute_exact_match(completion, sample.gt)
    edit_similarity = compute_edit_similarity(completion, sample.gt)

    return {
        "id": sample.id,
        "pkg": sample.pkg,
        "fpath": sample.fpath,
        "agent_res": completion,
        "raw_agent_output": raw_output,
        "gt": sample.gt,
        "exact_match": exact_match,
        "edit_similarity": edit_similarity,
        "tool_calls": len(tool_trace),
        "tool_trace": tool_trace,
        "error": error_message,
    }


def evaluate_samples(
    samples: List[Sample],
    client: ResponsesAPIClient,
    config: AgentConfig,
    batch_size: int,
    result_file: Path,
    existing_results: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results = dict(existing_results)
    pending_samples = [sample for sample in samples if sample.id not in results]

    if not pending_samples:
        return [results[sample.id] for sample in samples if sample.id in results]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
        future_to_sample = {
            executor.submit(run_agent_completion, client, config, sample): sample
            for sample in pending_samples
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_sample),
            total=len(future_to_sample),
            desc="Evaluating",
        ):
            sample = future_to_sample[future]
            try:
                results[sample.id] = future.result()
            except Exception as exc:
                results[sample.id] = {
                    "id": sample.id,
                    "pkg": sample.pkg,
                    "fpath": sample.fpath,
                    "agent_res": "",
                    "raw_agent_output": "",
                    "gt": sample.gt,
                    "exact_match": 0.0,
                    "edit_similarity": 0.0,
                    "tool_calls": 0,
                    "tool_trace": [],
                    "error": str(exc),
                }
            save_json(result_file, [results[s.id] for s in samples if s.id in results])

    return [results[sample.id] for sample in samples if sample.id in results]


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_eval_summary(
    eval_file: Path,
    args: argparse.Namespace,
    config: AgentConfig,
    dataset_file: Path,
    results: List[Dict[str, Any]],
) -> None:
    exact_matches = [float(item["exact_match"]) for item in results]
    edit_similarities = [float(item["edit_similarity"]) for item in results]
    tool_calls = [int(item["tool_calls"]) for item in results]
    error_count = sum(1 for item in results if item.get("error"))
    used_tools = sum(1 for item in results if int(item["tool_calls"]) > 0)

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    with eval_file.open("w", encoding="utf-8") as handle:
        handle.write("Evaluation Summary\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Language: {config.language}\n")
        handle.write(f"Dataset: {dataset_file}\n")
        handle.write(f"Repository Root: {config.repo_root}\n")
        handle.write(f"Model: {config.model}\n")
        handle.write(f"Base URL: {config.base_url}\n")
        handle.write(f"Batch Size: {args.batch_size}\n")
        handle.write(f"Max Steps: {config.max_steps}\n")
        handle.write(f"Max Output Tokens: {config.max_output_tokens}\n")
        handle.write(f"Reasoning Effort: {config.reasoning_effort}\n")
        handle.write(f"Tools Enabled: {config.allow_tools}\n")
        handle.write(f"Samples Evaluated: {len(results)}\n\n")
        handle.write(f"Exact Match: {average(exact_matches):.4f}\n")
        handle.write(f"Edit Similarity: {average(edit_similarities):.4f}\n")
        handle.write(f"Samples Using Tools: {used_tools}\n")
        handle.write(f"Average Tool Calls: {average(tool_calls):.4f}\n")
        handle.write(f"Failures: {error_count}\n")


def main() -> None:
    args = parse_args()
    dataset_file = Path(args.dataset_file).resolve() if args.dataset_file else resolve_default_dataset(args.lang)
    repo_root = resolve_repo_root(args.lang, args.repo_root)

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    if not dataset_file.exists():
        raise SystemExit(f"Dataset file not found: {dataset_file}")

    result_paths = build_result_paths(args.lang, args.model)
    existing_results = {} if args.overwrite else load_existing_results(result_paths["result_file"])

    samples = load_jsonl(dataset_file)
    if args.limit > 0:
        samples = samples[: args.limit]

    config = AgentConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        language=args.lang,
        repo_root=repo_root,
        max_steps=args.max_steps,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        reasoning_effort=args.reasoning_effort,
        allow_tools=not args.disable_tools,
    )

    client = ResponsesAPIClient(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )

    started_at = time.time()
    results = evaluate_samples(
        samples=samples,
        client=client,
        config=config,
        batch_size=args.batch_size,
        result_file=result_paths["result_file"],
        existing_results=existing_results,
    )
    elapsed = time.time() - started_at

    save_json(result_paths["result_file"], results)
    write_eval_summary(
        eval_file=result_paths["eval_file"],
        args=args,
        config=config,
        dataset_file=dataset_file,
        results=results,
    )

    print(f"Evaluation complete in {elapsed:.1f}s")
    print(f"Results: {result_paths['result_file']}")
    print(f"Summary: {result_paths['eval_file']}")


if __name__ == "__main__":
    main()

# python src/evaluation_agent.py --lang c --repo_root c_repo --model gpt-5
# python src/evaluation_agent.py --lang java --repo_root java_repo --model gpt-5
