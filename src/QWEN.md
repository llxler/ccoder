# Ccoder 项目上下文

## 项目概述
Ccoder 是一个基于上下文感知提示的 C 语言代码补全系统。该系统通过解析 C 代码库来构建特定仓库的上下文图，并将这些图用于为代码语言模型生成智能提示。它专门设计用于 C 代码补全任务，并与 CEval 数据集集成用于评估。

## 架构
系统包含几个关键组件：

1. **CParser** (`src/cfile_parse.py`): 使用 libclang 解析 C 文件并提取 AST 信息，包括函数、结构体、变量、枚举、联合体及其关系
2. **CProjectParser** (`src/preprocess.py`): 协调整个 C 项目的解析并构建上下文图
3. **CGenerator** (`src/generator.py`): 获取相关上下文并为代码补全生成提示
4. **CProjectSearcher** (`src/node_prompt.py`): 管理项目中符号之间的交叉引用和关系
5. **CModelTokenizer** (`src/tokenizer.py`): 处理不同模型的分词和提示截断
6. **Utils** (`src/utils.py`): 包含配置常量和路径定义

## 主要特性
- **特定仓库的上下文图**: 构建 C 仓库中代码关系的知识图谱
- **基于 AST 的解析**: 使用 libclang 进行精确的 C 代码解析
- **上下文感知提示**: 在提示中包含相关的函数定义、包含文件和相关代码
- **多模型支持**: 支持各种代码模型 (CodeLlama, GPT-3.5, GPT-4, StarCoder, DeepSeekCoder 等)
- **智能截断**: 管理提示长度以适应模型的 token 限制
- **批量处理**: 高效处理大型数据集并带有超时机制

## 主要入口点
- `src/preprocess.py`: 为数据集仓库构建上下文图
- `src/main.py`: 使用上下文图生成补全提示
- 配置常量在 `src/utils.py` 中定义

## 数据流
1. **预处理阶段**: 使用 `CParser` 解析每个 C 仓库以提取符号和关系，将结果存储为 `CEval/c_graph` 目录中的 `.json` 图文件
2. **生成阶段**: 对于每个补全请求，从图中检索相关上下文，并构建一个提示，将当前代码上下文与项目中的相关定义结合起来

## 依赖项
- libclang 用于 C 解析 (需要 CONDA_PREFIX/lib/libclang.so)
- Hugging Face Transformers 库用于模型分词器
- 用于模型集成的各种 ML/AI 库 (PyTorch 等)
- NetworkX 用于图操作
- clang Python 绑定

## 项目结构
- `CEval/`: 包含数据集、解析的仓库 (`c_repo`) 和生成的上下文图 (`c_graph`)
- `src/`: Ccoder 系统的源代码
- `results/`: 生成结果的输出目录 (动态创建)

## 配置
配置主要通过 `src/utils.py` 中的常量处理。主要可配置方面：
- 模型选择 (codellama7b, gpt35, gpt4 等)
- 上下文检索的最大跳数 (MAX_HOP)
- 仅包含定义 (ONLY_DEF = True)
- 启用文档字符串包含 (ENABLE_DOCSTRING = True)
- 包含的最后行数 (LAST_K_LINES = 1)
- 数据集和输出路径

## 构建和运行
### 环境设置:
```bash
conda create -n Ccoder python=3.10
pip install -r requirements.txt
```

### 准备 CEval 数据集:
```bash
# 如果数据集分成多个部分:
cd CEval && cat c_repo.zip.part_* > c_repo.zip
unzip c_repo.zip -d <your_path>
```

### 预处理阶段:
```bash
cd src && python preprocess.py
```
这为数据集中的每个仓库构建特定仓库的上下文图。

### 代码补全阶段:
```bash
cd src && python main.py --model $MODEL --file $OUT_FILE --c_dataset $DATASET
```
其中:
- `--model`: 要使用的代码模型 (deepseekcoder, codegen, codegen25, santacoder, starcoder, codellama, gpt35, gpt4)
- `--file`: 输出提示文件路径
- `--c_dataset`: C 语言数据集文件路径 (可选，如果不指定则使用默认值)
- `--timeout`: 处理每个样本的超时时间 (默认 30s)
- `--batch_size`: 保存结果的批量大小 (默认 100)

## 使用模式
- 运行 `python preprocess.py` 为数据集中的所有仓库构建上下文图
- 运行 `python main.py --model <model_name> --file <output_file>` 生成补全提示
- 系统处理 C 文件 (.c, .h) 并提取包含文件以在提示中提供相关上下文
- 批量处理自动定期保存结果以防止数据丢失
- 超时处理防止在复杂文件上挂起

## 重要限制
- 仅处理 C 语言文件 (.c 和 .h 扩展名)
- 需要 config.yaml 文件用于模型特定配置 (在 tokenizer.py 中引用但可能需要创建)
- 具有超时机制以防止在复杂或大型文件上挂起
- 使用批量处理以高效处理大型数据集
- 依赖于适当的 libclang 库路径配置

## 开发约定
- 代码库遵循模块化结构，在解析、生成和分词逻辑之间有明确分离
- 实现了错误处理，即使单个文件失败也能继续处理
- 系统采用数据驱动方法，数据集和结果使用 JSONL 格式
- 路径管理在 utils 模块中集中处理