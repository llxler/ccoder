可以，最好按**三层分层抽样**来做，而不是“直接随机抽 10%”。

因为你们这个 benchmark 本身有明显的**语言不均衡**、**难度梯度**和**repo 异质性**：全数据共 4044 个样本，其中 C 有 3336、Java 有 708；而且论文已经把难度定义为“文件内 unique external API calls 的数量”，并展示了难度越高性能越差，所以如果只做简单随机抽样，很容易被 reviewer 质疑样本分布偏了。  

---

## 一、推荐的抽样原则

你们要保住的是这句话：

> We additionally evaluate Claude Opus 4.6 on a fixed stratified 10% subset, preserving the original language composition and difficulty distribution.

这里的 “stratified” 最好具体指三层：

### 第 1 层：按语言分层

因为 C 和 Java 数量差很多，必须先保留原始比例。数据里 C:3336，Java:708，总计 4044。

10% 大约是：

* 总样本数：404
* C：3336 / 4044 ≈ 82.5%，所以取 **333**
* Java：708 / 4044 ≈ 17.5%，所以取 **71**

这样刚好 **333 + 71 = 404**。

这一步很重要，因为如果你不先按语言分层，随机抽出来的 Java 数可能会太少，而你们 benchmark 本来就是 C/Java 双子集。

---

### 第 2 层：在每种语言内部按难度分层

你们论文已经明确定义了难度：

* difficulty = 文件内 **number of unique external API calls**
* bucket 为 **[0,4), [4,8), [8,12), [12,+∞)**。

所以第二层必须保持这个 bucket 分布。

#### Java 可以直接按论文图里的数量来分

Figure 6 里 Java 的四个 bucket 样本数已经给了：

* [0,4): 141
* [4,8): 187
* [8,12): 158
* [12,+∞): 222
  总和正好 708。

那 Java 的 71 个样本就按 10% 比例取：

* [0,4): **14**
* [4,8): **19**
* [8,12): **16**
* [12,+∞): **22**

因为 14 + 19 + 16 + 22 = 71，刚好非常漂亮。

#### C 也要做同样的 bucket 分层

虽然你贴出来的正文里没有直接给出 C 各 bucket 的计数，但你们完全可以用同一个定义从元数据里算出来：

* 对每个 C 样本，计算 `external_api_calls_unique`
* 归到同样四个 bucket
* 在 C 的 333 个样本里按比例抽

这里**不要自己手填一个猜测比例**，而是要从原始 benchmark metadata 里现算。这样最稳。

---

### 第 3 层：在每个 (语言, 难度) 分层内按 repo 再做约束

这是很多人容易漏掉的一层，但对 rebuttal 非常有用。

因为 AutoCodeBench 来自 **58 个仓库**，而且 repo 分布本身是异质的，论文里也强调了不同 repo / package / functional topology 的多样性。 

如果你只做到“语言 + 难度”两层，依然可能出现：

* 某个大 repo 被抽很多
* 某些 repo 完全没抽到
* reviewer 说你这个 10% 其实主要测了某几个 repo

所以最好再加一个 repo 约束。

---

## 二、最实用的具体抽样算法

我建议你们直接用下面这个协议，足够严谨，也不复杂。

### Step 0：先为每个样本准备 metadata

每条样本至少准备这些字段：

* `sample_id`
* `language` ∈ {C, Java}
* `repo_name`
* `file_path`
* `difficulty_bucket` ∈ {[0,4), [4,8), [8,12), [12,+∞)}
* 可选：`package/category`

其中 `difficulty_bucket` 必须严格按论文定义来。

---

### Step 1：固定总样本数

设：

* `N_total = 404`

不要写 “about 10%” 然后抽 380 或 430。
最好就是固定成 404，因为全 benchmark 是 4044。

---

### Step 2：先分配语言 quota

* `N_C = 333`
* `N_Java = 71`

用原始 benchmark 的语言占比按比例取整。

---

### Step 3：在每种语言内分配 difficulty quota

#### Java

直接固定为：

* Java-[0,4): 14
* Java-[4,8): 19
* Java-[8,12): 16
* Java-[12,+∞): 22

这是最容易写进 rebuttal 的。

#### C

对 C 的四个难度 bucket 统计原始数量：

* `c1, c2, c3, c4`
* 总和 `c1+c2+c3+c4 = 3336`

然后分配：

* `q_i = 333 * c_i / 3336`

取整时用**largest remainder method**：

1. 先对每个 `q_i` 向下取整
2. 看还差几个名额
3. 按小数部分从大到小补齐

这样最后四个 bucket 的和一定等于 333。

---

### Step 4：在每个 (language, difficulty) 层内按 repo 分配 quota

这是关键的“防 cherry-pick”步骤。

假设你现在在某个 stratum，比如：

* Java-[4,8)，目标抽 19 个

这个 stratum 内可能有多个 repo，每个 repo 样本数不同。
设第 r 个 repo 在该 stratum 中有 `n_r` 个样本，总数是 `N_stratum`。

先算理论 quota：

* `q_r = 19 * n_r / N_stratum`

然后用下面规则：

#### 规则 A：按比例分配

默认按 `q_r` 比例分配。

#### 规则 B：小 repo 保护

如果某个 repo 在这个 stratum 里样本不少，但按比例算出来是 0，可以加一个最小覆盖规则：

* 若 `n_r >= 10` 且 `floor(q_r)=0`，则强制给它 **1 个**

这样可以避免全部落到大 repo 上。

#### 规则 C：最大占比上限

为了防止单一 repo 吞掉太多样本，可以设：

* 单 repo 在同一 stratum 中最多不超过 **20%** 的 quota

比如 Java-[4,8) 要抽 19 个，那同一 repo 最多拿 3 或 4 个。

#### 规则 D：最后仍用 largest remainder 调整总数

这样每个 stratum 的数量能严格对齐目标值。

---

## 三、如果时间不够，最低配也要做到什么程度

如果 rebuttal 时间非常紧，你们至少做到这个版本：

### 最低可接受版

1. 先按语言分层：333 C + 71 Java
2. 再按难度分层：

   * Java 用 Figure 6 的 14/19/16/22
   * C 用元数据按比例切四个 bucket
3. 每个 stratum 内**随机无放回抽样**
4. 固定随机种子
5. 公布抽样脚本和 sample_id 列表

这已经比“纯随机 10%”强很多了。

---

## 四、随机性怎么控制

这个一定要写清楚，不然 reviewer 会觉得你们是挑样本。

### 你们需要固定：

* random seed，例如 `seed = 20260325`
* sampling without replacement
* 所有 quota 在**运行前一次性确定**
* 抽样完成后**冻结样本集合**

不要出现：

* 先跑一版结果不好，再换 seed
* 先看哪个 subset 分数高，再决定用哪个
* 不公布 sample IDs

最稳的做法是 rebuttal 里写：

> We used a fixed random seed and will release the selected sample IDs in the supplementary material.

---

## 五、为什么一定不能只“随机抽 404 个”

因为你们论文已经证明了**难度显著影响性能**：随着 external dependencies 增加，EM 会单调下降，从 52.48% 掉到 33.78%。

这意味着如果只做纯随机抽样，哪怕总体还是 10%，也会有两个风险：

### 风险 1：高难样本被抽少了

那 Opus 4.6 的结果会被高估。

### 风险 2：Java 或某几个 repo 被抽得不均衡

那 reviewer 会说你的 subset 不代表原 benchmark。

所以 reviewer 真正在乎的不是“10% 行不行”，而是：
**这 10% 是否 faithful to the original distribution。**

---

## 六、我建议你们最后怎么写进 rebuttal

你们可以把抽样协议写得很具体，比如：

> Due to rebuttal-time budget constraints, we conducted an additional evaluation of Claude Opus 4.6 on a fixed 10% stratified subset of AutoCodeBench (404 tasks). We preserved the original C/Java composition of the benchmark (333 C and 71 Java tasks), and further stratified samples within each language by the same difficulty buckets used in our paper, defined by the number of unique external API calls per file: [0,4), [4,8), [8,12), and [12,+∞). Within each stratum, samples were drawn without replacement using a fixed random seed, with repository-level proportional allocation to avoid concentration on a few large repositories.

这个表述很稳。

---

## 七、我个人最推荐的最终方案

如果让我替你拍板，我会建议你们这样做：

### 正式方案

* 总数：404
* 语言：333 C + 71 Java
* 难度：四个 bucket 分层
* repo：同层内按比例 + 最大占比约束
* seed 固定
* 无放回抽样
* 先只跑 **Direct Generation**
* 若预算还够，再在**同一 404 个样本**上补跑 Graph-RAG

因为 reviewer 这条意见本质上是在问：
“你们没测到更前沿的模型，那你们的结论还能站住吗？”

所以第一优先级是给出 **Opus 4.6 on a fair stratified subset**，而不是先追求更复杂的设置。

---

## 八、一个很实用的小细节

你们最后最好在 rebuttal 里**顺便承认模型选择不是 exhaustive，而是 representative**。因为论文实验设置本来写的是 **8 widely used models**，并且结果分析里也用了 **representative models** 这个口径。 

这样 reviewer 就算继续追问，也更难抓你们“claim too strong”这个点。

---

如果你愿意，我下一条可以直接给你一版：
**“抽样 protocol + rebuttal 英文描述 + 可直接实现的伪代码”**。
