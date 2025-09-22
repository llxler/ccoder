# Ccoder

## Prepare Environment

```
conda create -n Ccoder python=3.10
pip install -r requirements.txt
```

## CEval Dataset

Each sample in CEval (`CEval/c_metadata.jsonl`) contains the following fields:

- `id`: task id
- `pkg`: the repository it belongs to
- `fpath`: the file path where the code to be completed
- `input`: the content before the cursor position to be completed.
- `gt`: the ground truth of the code line to be completed.

Merge c_repo

` cd CEval && cat c_repo.zip.part_* > c_repo.zip`
`unzip c_repo.zip -d <your_path>`

## Quickstart
### Repo-specific Context Graph
During offline preprocessing, we build a repo-specific context graph for each repository in the datasets:

```
cd src && python preprocess.py
```

### Code Completion
In real-time code completion, we generate the prompts for querying code language models (LMs):

```
cd src && python main.py --model $MODEL --file $OUT_FILE --c_dataset $DATASET
```
