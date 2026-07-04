# 度分布分析说明

本说明文档对应 `degree_analysis.py`，帮助你理解脚本的输入、运行方法以及 `degree_distribution/` 目录内各个产出（CSV 与图片）的含义。

## 1. 脚本职责与依赖
- **目的**：读取 `city_tables/<city>_stats.csv` 中的度计数，计算各城市的度分布统计量，并对六种分布（幂律、截断幂律、指数、对数正态、泊松、负二项）做极大似然拟合与诊断，同时绘制度频散点图和 log-log 图。
- **主要依赖**：Python 3.10+、NumPy、Matplotlib。运行脚本前需先用 `graph_stats.py` 生成 `city_tables/*.csv`。

## 2. 运行方式
```bash
python degree_analysis.py
```
可选参数目前写死在脚本内：自举 200 次（`BOOTSTRAP_REPS`），交叉验证 5 折（`CV_FOLDS`），默认将结果写入 `degree_distribution/`。

## 3. 结果文件结构
```
degree_distribution/
├── degree_distribution_summary.csv   # 度分布基准统计量
├── degree_distribution_fits.csv      # 各分布拟合及诊断指标
├── degree_frequency.png              # 频数散点矩阵
└── degree_loglog.png                 # log-log 图与线性拟合
```

### 3.1 degree_distribution_summary.csv
列名含义如下，所有度值均来自无向图节点的度：
- `city`：城市名称（与 *_Edgelist.csv 前缀一致）。
- `mean`：加权平均度，按 `degree_count_k` 频数加权。
- `variance` / `std_dev`：对应方差与标准差。
- `skewness`：三阶中心矩除以标准差立方，刻画分布左右偏斜程度；负值代表长左尾。
- `max_degree`：样本中的最大度。
- `mode`：出现频数最高的度值（多众数时取最小者）。

### 3.2 degree_distribution_fits.csv
每行表示某城市 × 某分布的拟合结果：
- `distribution`：`power_law`、`truncated_power_law`、`exponential`、`lognormal`、`poisson`、`negative_binomial` 之一。
- `parameters`：关键参数（示例：`alpha=2.04; xmin=1`）。若参数不可用则会是 `nan`。
- `log_likelihood`：对离散度概率质量函数取对数后乘以频数再求和，数值越大说明拟合越好。
- `num_params`：模型中的自由参数个数（计算 AICc 用）。
- `cvm_stat`：Cramér–von Mises 统计量，衡量经验分布与拟合 CDF 的偏离；越小越好。
- `cvm_pvalue`：基于 200 次自举的经验 p 值，表示“在拟合分布下观察到更大偏离的概率”；若 <0.05 则表明该分布可能不适配。
- `aicc`：修正后 Akaike 信息准则，越小越好；不同城市间不可直接比较。
- `cv_loglik`：5 折交叉验证的平均对数似然，取值为 “单位样本对数概率”；越高通常代表泛化更好。

### 3.3 图像文件
- `degree_frequency.png`：每个子图对应一个城市，**蓝色点 (`#4C78A8`)** 表示 `(度, 频数)` 散点；虚线网格帮助比对常见度值，标题即城市名。
- `degree_loglog.png`：坐标轴均为对数，**橙色点 (`#F58518`)** 是观测值，**深蓝折线 (`#003f5c`)** 是 `log10` 空间内的线性拟合，图例中的 `R²` 为拟合优度，可用来粗略判断幂律倾向。灰色网格展示幂律下的线性关系。

## 4. 如何解读输出
1. 先查看 `degree_distribution_summary.csv`，判断平均度、最大度、偏度等基本形态，便于辨别是否存在长尾或孤立节点集中。
2. 结合 `degree_distribution_fits.csv`：
   - 对比同一城市不同分布的 `aicc` 与 `cv_loglik` 以挑选相对最佳模型。
   - 使用 `cvm_pvalue` 检查拟合是否可接受（p 值很低表示模型与观测有显著差异）。
   - 关注参数本身（如幂律的 `alpha`、截断项 `lambda`），可与其他城市对照。
3. 查看两张图片：
   - 频数散点图能直观看到常见度值与异常点。
   - log-log 图可判断在高阶度下是否沿直线衰减；若颜色点明显偏离拟合线，则幂律假设较弱。

## 5. 常见问题
- 若 `city_tables` 内缺少某城市 CSV，脚本会报错 “No city tables found.”——请先运行 `python graph_stats.py`。
- 若某城市节点度非常低或样本为空，`mean`、`aicc` 等列会写入 `nan`，图像子图会显示 “数据不足”。

有了以上信息，即使不阅读源码，也能理解每个数值与颜色所指向的度分布特征。
