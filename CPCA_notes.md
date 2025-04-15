# 动态时空模式分析：重构时空图谱

## 背景与目的

在 fMRI 信号分析中，我们通常采用 **Hilbert 变换** 和 **奇异值分解（SVD）** 提取信号的时空模式，并通过相位重采样来重建大脑活动的时空图谱。此方法涉及将复数信号的相位信息和空间成分结合，从而生成大脑信号的时空模式。

## 数学原理

### 1. **Hilbert 变换**：

Hilbert 变换用于将实数信号转换为复数信号。通过这个变换，我们可以得到信号的 **相位信息** 和 **幅度信息**：

$$
\tilde{x}(t) = x(t) + i \cdot \mathcal{H}[x(t)]
$$

- $$ x(t) $$ 是原始信号。
- $$ \mathcal{H}[x(t)] $$ 是信号的 Hilbert 变换（虚部）。
- 复数信号的实部 $$ x(t) $$ 代表原始信号，虚部 $$\mathcal{H}[x(t)] $$ 代表相位信息。

### 2. **奇异值分解（SVD）** 和 **PCA**：

奇异值分解（SVD）用于将信号数据矩阵分解为以下三部分：

$$
X = U \Sigma V^H
$$

- **$$U$$**：时间向量矩阵，描述时间上的变化模式（时间成分）。
- **$$Σ$$**：奇异值矩阵，表示每个主成分的强度（数据的贡献度）。
- **$$V^H$$**：空间向量矩阵，描述空间上的变化模式（空间成分）。

**PCA** 是通过 **SVD** 实现的降维方法，目的是从高维信号中提取最能代表数据的成分。

### 3. **相位重采样**：

为了捕捉信号的传播模式，我们将 **相位信息** 重采样到 $$  n_{\text{bin}} $$ 个相位区间。相位信息由每个主成分的时间序列通过 **Euler 公式** 来得到。

具体来说，对于每个主成分 $$ u $$ 和 $$v $$，我们可以根据其相位进行分箱。对于每个相位区间，计算其平均相位，然后通过空间向量$$Vh $$和这些相位来重建时空图谱。

### 4. **重构时空图谱的公式**：

#### 计算相位：

$$
\text{angle_u} = \text{angle}(U) \quad (\text{每个时间成分的相位})
$$

#### 分箱操作：

将相位分为 $$ n_{\text{bin}} $$ 个区间：

$$
\text{bin_idx} = \text{digitize}(\text{angle_u}, \text{bins}) - 1
$$

其中，`bins` 是用来进行相位重采样的区间（从  0  到 $$2\pi $$）。

#### 重建时空图谱：

$$
\text{phase_map} = \text{mean_phase_u} \cdot Vh
$$

### 代码实现解析

```python
def reconstruct(u, vh, n_bin=32):
    n_tps, n_comp = u.shape  # n_tps: 时间点数，n_comp: 主成分数
    angle_u = np.mod(np.angle(u), 2 * np.pi)  # 计算时间成分的相位，限制在 [0, 2π]
    bins = np.linspace(0, 2 * np.pi, n_bin + 1)  # 创建相位区间

    # 获取每个时间成分的相位所在的区间
    bin_idx = np.digitize(angle_u, bins) - 1  # 计算每个元素属于哪个区间

    # 初始化存储相位总和和计数的数组
    sum_phase_u = np.zeros((n_bin, n_comp), dtype=np.complex128)  # 存储相位总和
    count_phase_u = np.zeros((n_bin, n_comp), dtype=int)  # 存储每个区间的计数

    # 对每个主成分进行加权求和（加权是通过相位区间的计数）
    for comp in range(n_comp):
        np.add.at(sum_phase_u[:, comp], bin_idx[:, comp], u[:, comp])  # 加权求和
        np.add.at(count_phase_u[:, comp], bin_idx[:, comp], 1)  # 计数

    # 计算每个区间的平均相位
    mean_phase_u = np.divide(sum_phase_u, count_phase_u, out=np.zeros_like(sum_phase_u), where=count_phase_u != 0)

    # 通过相位和空间成分重建时空图谱
    phase_map = mean_phase_u[:, :, np.newaxis] * vh[np.newaxis, :, :]  # 加权空间成分
    phase_map = np.transpose(phase_map, (1, 0, 2))  # 转置，使得时空图谱正确

    return phase_map
```