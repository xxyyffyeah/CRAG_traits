# Trait-Aware Movie Recommendation Evaluation

本文档说明如何使用 `evaluate_with_traits.py` 进行带有个人特征（Personal Trait）的电影推荐评估。

## 功能概述

该脚本实现了以下功能：

1. **随机采样**：从5231个测试样本中随机抽取100个样本
2. **Trait分配**：为每个样本独立随机分配一个personal trait（从traits.json的~150个trait中抽取）
3. **Trait-aware推荐**：在所有4类prompt中添加"Personal Trait: XXX"信息
4. **传统评估指标**：保留原有的Recall@k和NDCG@k评估
5. **Trait符合度评估**：使用GPT-4o评估推荐列表与trait的符合程度（1-5分）
6. **可视化**：生成对比图表展示结果

## 依赖要求

- Python 3.8+
- 所有evaluate.py的依赖
- OpenAI API key（需设置在.env文件或环境变量中）

## 文件说明

- `evaluate_with_traits.py` - 主评估脚本（100个样本）
- `test_traits_small.py` - 小规模测试脚本（10个样本，用于快速测试）
- `traits.json` - Personal trait定义文件
- `README_TRAITS.md` - 本文档

## 使用方法

### 1. 环境准备

确保已设置OpenAI API key：

```bash
# 方法1：使用.env文件
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 方法2：设置环境变量
export OPENAI_API_KEY=your_api_key_here
```

### 2. 小规模测试（推荐首次运行）

使用10个样本快速测试功能是否正常：

```bash
python test_traits_small.py
```

预计运行时间：约30-60分钟
API调用次数：约310次

### 3. 完整评估

使用100个样本进行完整评估：

```bash
python evaluate_with_traits.py
```

预计运行时间：约5-8小时
API调用次数：约3100次

### 4. 自定义参数

可以通过修改脚本开头的参数来自定义评估：

```python
n_samples = 100  # 采样数量
random_seed = 42  # 随机种子（保证可复现）
K_list = list(range(5, 40, 5))  # 检索数量列表
k_list = [5, 10, 15, 20]  # 评估的top-k值
```

## 输出结果

所有结果保存在 `results/trait_evaluation/` 目录：

### 数据文件（.pkl）

1. **sampled_data_seed42.pkl** - 采样的100个样本（含trait分配）
2. **test_with_traits_retrieval.pkl** - 检索后的结果
3. **test_with_traits_rerank.pkl** - 重排后的结果
4. **test_with_traits_final.pkl** - 最终评估结果（含trait评分）

### 指标文件（.json）

1. **trait_metrics.json** - Trait符合度指标
2. **comprehensive_metrics.json** - 所有指标的综合JSON

### 可视化图表（.jpg）

1. **trait_evaluation_results.jpg** - 主对比图
   - Trait alignment均值图（带误差棒）
   - 分数分布堆叠图
   - 质量率对比图

2. **trait_alignment_by_category.jpg** - 按类别分析图
   - 展示不同trait类别的平均符合度

### 日志文件（.txt）

**evaluation_log_<timestamp>.txt** - 详细日志
- 总体评估指标摘要
- 每个样本的详细信息：turn_id、assigned_trait、推荐列表、评分、解释

## 评估指标说明

### 传统推荐指标

- **Recall@k**：前k个推荐中的命中率
- **NDCG@k**：归一化折扣累计增益

### Trait符合度指标

对每个K值计算以下指标：

- **mean_alignment**：平均符合度分数（1-5）
- **std_alignment**：标准差
- **median_alignment**：中位数
- **excellent_rate**：完美符合率（分数=5的比例）
- **good_or_better_rate**：良好及以上率（分数>=4的比例）
- **poor_or_worse_rate**：差及以下率（分数<=2的比例）

### Trait评分标准

- **5 - Excellent**：所有推荐完美尊重trait
- **4 - Good**：推荐很好地符合trait
- **3 - Acceptable**：有些问题但大体适当
- **2 - Poor**：多个推荐不合适
- **1 - Very Poor**：多个推荐高度不适当

## 实现细节

### Prompt修改

所有4个prompt都添加了trait信息：

```
Here is the conversation: {context}
Personal Trait: {trait}  <-- 新增行
[其他信息...]
```

### 评估流程

1. 加载CF模型
2. 加载测试数据
3. 预处理
4. **采样100个+分配trait** ← 新增
5. 检索+反思（带trait）
6. 零样本推荐（带trait）
7. 带检索推荐（带trait）
8. 推荐重排（带trait）
9. **Trait符合度评估** ← 新增
10. 后处理
11. 综合评估
12. 可视化
13. 保存结果

### API调用量估算

以100个样本为例：

- 检索反思：100 × 7 = 700次
- 零样本推荐：100 × 1 = 100次
- 带检索推荐：100 × 7 = 700次
- 推荐重排：100 × 8 = 800次
- Trait符合度评估：100 × 8 = 800次

**总计**：约3100次GPT-4o API调用

## 可复现性

- 固定随机种子（seed=42）
- 采样数据保存为pkl文件
- 所有中间结果都可保存和重新加载

## 故障排查

### 问题1：API调用失败

**症状**：大量"API Failed"错误

**解决方案**：
- 检查API key是否正确设置
- 检查API rate limit
- 减少并发线程数（修改n_threads参数）

### 问题2：内存不足

**症状**：程序崩溃或OOM错误

**解决方案**：
- 减少采样数量（n_samples）
- 减少K值数量
- 使用分批处理

### 问题3：解析错误

**症状**：trait评分全部为默认值3

**解决方案**：
- 检查详细日志查看原始响应
- 可能需要调整解析逻辑

## 扩展建议

1. **增加样本量**：修改n_samples到200-500获得更稳健的结果
2. **分层采样**：按trait类别进行分层采样，确保每个类别都有足够样本
3. **对比实验**：运行有trait和无trait两个版本，对比差异
4. **细粒度评估**：对每部电影单独评估而非整体列表

## 联系与支持

如有问题，请查看：
- 主评估脚本：evaluate.py
- 工具函数：libs/utils.py, libs/model.py, libs/metrics.py
