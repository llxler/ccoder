# CONSTANT for settings

MAX_HOP = None
ONLY_DEF = True

ENABLE_DOCSTRING = True
LAST_K_LINES = 1

import os
MODEL = "codellama7b"
FILE = "c"
DS_BASE_DIR = os.path.abspath("../CEval")
DS_REPO_DIR = os.path.join(DS_BASE_DIR, f"{FILE}_repo")
DS_FILE = os.path.join(DS_BASE_DIR, f"{FILE}_metadata.jsonl")
DS_GRAPH_DIR = os.path.join(DS_BASE_DIR, f"{FILE}_graph")
PT_FILE = os.path.join(DS_BASE_DIR, f"{FILE}_{MODEL}_prompt.jsonl")
BASE_DIR = os.path.abspath("../")
RESULT_DIR = os.path.join(BASE_DIR, f"results/{MODEL}")
# RESULT_DIR = os.path.join(BASE_DIR, f"results_rag/{MODEL}") # 为了langchain_rag
EVAL_FILE = os.path.join(RESULT_DIR, f"{FILE}_{MODEL}_eval.txt")
RESULT_FILE = os.path.join(RESULT_DIR, f"{FILE}_{MODEL}_result.json")
IMP_FILE = os.path.join(RESULT_DIR, f"{FILE}_{MODEL}_improved.json")

