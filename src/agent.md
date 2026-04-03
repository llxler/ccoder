# `evaluation_agent.py` 设计说明

## 1. 背景

`evaluation_agent.py` 的目标，是把原来基于 RAG prompt 的评测方式，改成基于标准 Agent 的评测方式。

核心变化不是“换一个模型名字”，而是把推理过程从：

- 先离线检索
- 再把检索结果拼进 prompt
- 最后一次性生成答案

改成：

- 只给模型原始补全前缀 `input`
- 允许模型像 agent 一样按需查看仓库
- 让模型自己决定是否读文件、搜代码、看目标文件上下文
- 最后再输出补全结果并进行统一评测

这样更贴近 rebuttal 里说的观点：`RAG` 不应该被默认视为唯一的上下文增强方式，标准的 repository-aware agent 也可以完成同类任务，而且检索行为是动态的、按需发生的。

## 2. 与 `evaluation_gpt5.py` 的本质区别

`evaluation_gpt5.py` 的评测对象，本质上还是“模型 + prompt 工程”：

- 原始数据来自 `metadata.jsonl`
- 增强输入来自 `*_graph_prompt.jsonl` 或其他 RAG prompt 文件
- 模型只是在收到 prompt 后直接生成

`evaluation_agent.py` 则不再依赖任何 `prompt.jsonl`：

- 数据源只使用 `CEval/c_metadata.jsonl` 或 `CEval/java_metadata.jsonl`
- 真正输入给模型的用户内容只有样本中的 `input`
- 仓库上下文不再提前拼进 prompt，而是通过工具调用按需获取

因此，这个脚本评测的是：

- 模型在“原始代码前缀 + agent工具”条件下的补全能力

而不是：

- 模型在“人工或规则构造好的增强 prompt”条件下的补全能力

## 3. 总体架构

整个脚本可以分成六层：

1. 参数与运行配置
2. 数据样本表示
3. Responses API 封装
4. 本地仓库工具集
5. Agent 循环推理
6. 后处理与评测

对应代码结构如下：

- `Sample` / `AgentConfig`
- `ResponsesAPIClient`
- `RepoTools`
- payload 构造函数
- `run_agent_completion`
- `postprocess_completion` / `compute_exact_match` / `compute_edit_similarity`
- `evaluate_samples` / `write_eval_summary` / `main`

## 4. 设计目标

这版 agent 的设计主要围绕四个目标：

### 4.1 只使用原始 prompt

脚本只读取样本中的：

- `input`
- `gt`
- `fpath`
- `pkg`

其中真正发给模型的“问题”只有 `input`。

这样可以保证评测更干净：模型拿到的不是预先检索好的 RAG 上下文，而是一个更接近真实 IDE 补全场景的原始前缀。

### 4.2 让模型按需检索

agent 不会一上来把整个仓库内容塞给模型，而是提供少量必要工具：

- `get_target_file_context`
- `read_file`
- `search_code`
- `list_dir`

模型如果觉得当前前缀足够，就可以直接输出答案；如果不够，再自己决定是否查上下文。

这和 RAG 的最大区别在于：

- RAG 是“先检索，再生成”
- Agent 是“边判断，边检索，边生成”

### 4.3 让上下文获取尽量贴近真实开发行为

工具集的设计模仿开发者补全时最常见的动作：

- 先看当前文件光标附近
- 再读相关文件
- 再搜索符号或 API 用法
- 必要时查看目录结构

所以这版 agent 不是一个“泛化智能体”，而是一个非常聚焦的代码补全 agent。

### 4.4 保证评测可重复、结果可落盘

脚本最终仍然回到标准自动评测：

- Exact Match
- Edit Similarity

同时支持：

- 中间结果实时保存
- 已完成样本断点续跑
- 批量并发评测

## 5. 配置层设计

`AgentConfig` 负责把所有运行时选项集中管理，包括：

- `model`
- `base_url`
- `api_key`
- `language`
- `repo_root`
- `max_steps`
- `max_output_tokens`
- `timeout_seconds`
- `reasoning_effort`
- `allow_tools`

这样做有两个好处：

1. agent 逻辑不需要到处读环境变量或命令行参数
2. 后面如果想切换模型、切换 repo、关闭工具，都只需要改配置对象

## 6. 为什么不用 `openai` SDK，而是直接调 HTTP

`ResponsesAPIClient` 直接通过 `urllib` 调 `/responses` 接口，而不是依赖 `openai` Python SDK。

这样设计有几个考虑：

### 6.1 减少环境依赖

在一些运行环境里，`openai` 包不一定已经装好，但标准库一定存在。直接走 HTTP 可以让脚本更容易迁移。

### 6.2 更容易保持接口显式

Agent 模式的关键是：

- 发送 `tools`
- 读取 `function_call`
- 回填 `function_call_output`

直接构造 JSON payload，逻辑会更透明，便于理解 agent 的真实运行流程。

### 6.3 方便兼容不同 base URL

脚本支持 `--base_url`，所以既可以连官方 API，也可以连兼容 Responses API 的中转服务。

## 7. 工具层设计

`RepoTools` 是这个 agent 的核心。它把“模型可以做什么”限制在几个和代码补全直接相关的动作上。

### 7.1 `get_target_file_context`

这是最重要的工具。

它做的事是：

- 找到当前样本对应的目标文件
- 尝试把 `input` 对齐到文件中的真实光标位置
- 把光标前后一定范围内的代码返回给模型
- 用 `<CURSOR>` 标记当前位置

这是最贴近 IDE 补全的局部上下文。

为什么单独做这个工具，而不是只保留 `read_file`？

因为补全任务最先需要的，通常不是“读整个文件”，而是“看光标附近”。

### 7.2 `read_file`

这个工具提供按相对路径、按行号读取文件的能力。

它适合以下场景：

- 模型已经知道自己要看哪个文件
- 需要读取某个定义附近的代码
- 需要确认另一个文件里的接口签名

### 7.3 `search_code`

这个工具相当于给模型提供了一个轻量版 `rg`。

优先使用 `ripgrep`，如果环境中没有 `rg`，则回退到 Python 正则搜索。

作用是：

- 搜某个符号定义
- 搜某个 API 的用法
- 搜某个类/结构体/宏的出现位置

### 7.4 `list_dir`

这个工具不是最常用，但很有必要。

当模型不确定文件在哪里，或者想先理解项目结构时，它可以列目录，而不是盲猜路径。

### 7.5 工具边界控制

工具层还做了路径安全控制：

- 所有路径都必须落在 `repo_root` 下
- 不允许路径逃逸

这是为了让 agent 行为稳定，也避免工具语义过宽。

## 8. 为什么还保留 `pkg` 和 `fpath`

虽然你要求“包怎么处理不用太管，按标准 agent 来”，但代码里仍然保留了 `pkg` 和 `fpath` 的轻量使用：

- 优先尝试 `repo_root / fpath`
- 再尝试 `repo_root / pkg / fpath`
- 最后按文件名在仓库里兜底搜索

原因是评测样本通常知道目标文件，只是不保证仓库实际摆放形式完全一致。这里的设计不是依赖 `pkg` 做复杂路由，而只是把它当作一个辅助定位信号。

## 9. Agent 循环的设计

`run_agent_completion` 是整套方法的核心。

它的逻辑是：

1. 构造初始 payload
2. 把系统提示词和原始 `input` 发给模型
3. 检查模型回复中是否包含 `function_call`
4. 如果有，就在本地执行工具
5. 把工具结果作为 `function_call_output` 送回模型
6. 重复上述过程，直到模型不再调用工具，而是直接输出文本
7. 对文本做后处理，再算指标

这就是一个标准的 tool-using agent loop。

### 9.1 为什么要有 `max_steps`

Agent 不是一次性生成，而是多轮循环，所以必须设置上限，避免：

- 模型反复调用工具
- 陷入无效搜索
- 单个样本耗时过长

`max_steps` 的作用就是给 agent 一个清晰的预算。

### 9.2 为什么关闭 `parallel_tool_calls`

这里显式设置了 `parallel_tool_calls = False`。

原因是这不是一个通用搜索 agent，而是一个评测脚本。我们更希望：

- 工具调用顺序可解释
- 调试时更容易复现
- 单步 trace 更清晰

所以选择了串行工具调用，而不是追求更复杂的并行检索。

### 9.3 为什么记录 `tool_trace`

每个样本都会保存：

- 调用了哪些工具
- 每一步传了什么参数
- 工具是否成功

这非常重要，因为 agent 评测不仅要看最终分数，还要看：

- 模型到底有没有利用仓库上下文
- 常见失败是不是因为工具选错了
- 是否存在过度搜索或无效搜索

也就是说，这份 trace 是后续分析 agent 行为的基础。

## 10. Prompt 设计思路

系统提示词有两个核心任务：

### 10.1 把任务限定为“补全”

提示词明确要求模型：

- 只输出补全代码
- 不重复用户输入
- 不输出解释
- 不输出 markdown

这样做是为了让最终输出尽量与 `gt` 的格式一致。

### 10.2 把工具使用限定为“按需”

提示词没有要求模型必须调用工具，而是强调：

- 只有在需要时再使用工具
- 优先看目标文件局部上下文

这能避免 agent 退化成“每题都先搜一遍仓库”的低效模式。

## 11. 后处理设计

Agent 输出通常比普通 completion 更容易出现格式噪声，所以 `postprocess_completion` 做了三层清理。

### 11.1 去掉 markdown fence

有些模型会输出：

```text
```java
...
```
```

评测不需要这些内容，所以先去掉。

### 11.2 去掉 prompt 重复

有些模型会把用户输入的最后几行又重复一遍。脚本通过比较 `prompt` 末尾与输出开头，尽量把这部分裁掉。

### 11.3 截断到“一个补全单元”

这是最关键的一步。

目标不是保留模型的全部输出，而是尽量截取成和 ground truth 同粒度的补全片段，例如：

- 一个声明
- 一行赋值
- 一个方法头
- 一个块头

这一步尤其考虑了 C/Java 数据集中大量以：

- `;`
- `{`
- `)`
- 多行声明片段

结尾的样本。

其中有一个专门的修正点：

- 如果 `{` 出现在当前行末尾，例如 `ViewHolder holder) {`，则保留它
- 如果 `{` 单独出现在下一行，则只保留上一行签名

这样可以更好适配 C 和 Java 两种数据的真实 ground truth 形式。

## 12. 指标设计

评测仍然沿用两类指标：

### 12.1 Exact Match

预测和 `gt` 去掉首尾空白后完全一致，记为 1，否则为 0。

它衡量的是：

- 最严格意义上的补全正确率

### 12.2 Edit Similarity

使用 Levenshtein 编辑距离计算相似度：

- 越接近 1 越好

它能反映：

- 预测虽然不完全正确，但是否已经非常接近标准答案

这对于 agent 评测特别重要，因为 agent 有时会做对大方向，只是符号或格式略有差异。

## 13. 批量评测与断点续跑

`evaluate_samples` 支持：

- `ThreadPoolExecutor` 并发跑多个样本
- 每完成一个 future 就落盘一次
- 如果结果文件已存在，则跳过已完成样本

这说明脚本不是一个“只能单次完整跑完”的实验脚本，而是一个更接近长期实验使用的评测器。

这种设计对 agent 特别重要，因为 agent 的单样本耗时通常比纯 completion 更高。

## 14. 输出结果设计

每个样本保存的关键信息包括：

- `agent_res`
- `raw_agent_output`
- `gt`
- `exact_match`
- `edit_similarity`
- `tool_calls`
- `tool_trace`
- `error`

这比只保存最终预测更有价值，因为后续可以直接分析：

- 模型是否真正用到了工具
- 工具是否提高了结果
- 失败是生成失败还是工具失败

汇总文件则记录：

- 平均 Exact Match
- 平均 Edit Similarity
- 使用工具的样本数
- 平均工具调用次数
- 失败样本数

## 15. 这份设计为什么适合 rebuttal

如果 rebuttal 里要强调“当前 RAG 评测应该替换成 agent 评测”，这份脚本正好体现了这个观点：

### 15.1 它不再把检索结果固化进 prompt

这避免了把方法效果混在“prompt 工程质量”里。

### 15.2 它允许模型自主决定是否检索

这更符合 agent 的定义，也更接近真实编程场景。

### 15.3 它把检索行为显式记录下来

不仅能看最终分数，还能看 agent 是怎么获得答案的。

### 15.4 它保留了和旧方法可比的评测指标

虽然范式换了，但最终输出仍然能和 `evaluation_gpt5.py` 放在同一指标体系下比较。

## 16. 当前实现的边界

这版实现已经可以用于实验，但它仍然是一个“面向评测的轻量 agent”，不是完整开发 agent。

目前的边界主要有：

- 只支持只读工具，不支持编辑仓库
- 工具集合偏小，只覆盖补全最常见需求
- 对 Responses API 的字段格式有依赖
- 目标文件定位是 best effort，不是强绑定索引
- 批量并发时需要考虑 API 限速

这些限制并不妨碍它作为 rebuttal 或实验中的 agent baseline，反而使它更聚焦、更容易解释。

## 17. 一句话总结

`evaluation_agent.py` 的核心思想是：

> 不再把“检索”预先编译进 prompt，而是把模型放进一个带有限仓库工具的环境里，让它以 agent 的方式按需获取上下文，再输出代码补全，并用与原评测一致的指标进行比较。

这也是它和传统 RAG evaluation 最大的区别。
