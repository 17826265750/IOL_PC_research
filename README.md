# IGBT功率循环寿命预测系统

<p align="center">
  <strong>Power Device Lifetime Prediction & Analysis System</strong><br>
  基于CIPS 2008模型的IGBT模块功率循环寿命预测与可靠性分析软件
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/React-18-blue.svg" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.115-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/MUI-5.x-purple.svg" alt="MUI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Tests-555+-brightgreen.svg" alt="Tests">
</p>

---

## 目录

- [项目简介](#项目简介)
- [理论背景](#理论背景)
  - [功率循环失效机理](#功率循环失效机理)
  - [寿命预测模型](#寿命预测模型)
  - [雨流计数法](#雨流计数法)
  - [Miner线性损伤累积](#miner线性损伤累积)
  - [Weibull可靠性分析](#weibull可靠性分析)
- [安装教程](#安装教程)
- [使用教程](#使用教程)
- [API参考](#api参考)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [测试](#测试)
- [开发指南](#开发指南)
- [参考文献](#参考文献)

---

## 项目简介

本软件是专为**IGBT模块等功率半导体器件**设计的寿命预测与可靠性分析系统。基于国际主流的**CIPS 2008 (Bayerer) 模型**，结合雨流计数法、Miner损伤累积法则和Weibull统计分析，为功率电子可靠性工程提供完整的解决方案。

### 核心特性

| 功能模块 | 描述 |
|---------|------|
| **寿命预测计算** | 5种主流寿命模型：Coffin-Manson、CM-Arrhenius、Norris-Landzberg、CIPS 2008、LESIT |
| **雨流计数分析** | ASTM E1049标准的循环提取算法，从任意温度历程中识别循环 |
| **损伤累积分析** | 基于Miner法则的线性损伤累积计算 |
| **Weibull分析** | B10/B50/B63.2特征寿命计算，可靠度曲线绘制 |
| **敏感性分析** | 龙卷风图、热力图、Sobol全局敏感性分析 |
| **参数拟合** | 非线性最小二乘拟合，支持CIPS 2008专用拟合 |
| **剩余寿命评估** | 退化趋势外推，健康指数计算 |
| **安全裕度计算** | 设计寿命vs预测寿命，统计不确定性分析 |
| **报告导出** | PDF/Excel专业报告生成 |

---

## 理论背景

### 功率循环失效机理

IGBT模块在功率循环过程中，由于不同材料的热膨胀系数（CTE）不匹配，会产生热机械应力。主要的失效模式包括：

1. **键合线脱落（Bond Wire Lift-off）**
   - 铝键合线与硅芯片之间的CTE差异
   - 导致键合点疲劳裂纹扩展
   - 表现为导通电阻逐渐增加

2. **焊层疲劳（Solder Joint Fatigue）**
   - 芯片与基板之间焊料层的CTE不匹配
   - 导致焊层裂纹和空洞
   - 表现为热阻增加

3. **铝金属化重构（Aluminum Reconstruction）**
   - 高温下铝原子迁移
   - 导致金属化层变薄、电阻增加

这些失效机理都与**结温波动（ΔTj）**和**最高结温（Tj_max）**密切相关，因此寿命预测模型主要围绕这些参数建立。

---

### 寿命预测模型

#### 1. Coffin-Manson模型

最基础的热疲劳寿命模型，基于塑性应变与疲劳寿命的幂律关系：

$$N_f = A \times (\Delta T_j)^{-\alpha}$$

| 参数 | 含义 | 典型值 |
|-----|------|-------|
| $N_f$ | 失效循环数 | - |
| $A$ | 材料系数 | $10^2 \sim 10^8$ |
| $\Delta T_j$ | 结温波动（K） | 10~200 K |
| $\alpha$ | 温度波动指数 | 1.0~6.0（典型值2~5） |

**适用场景**：基础热疲劳分析，参数较少，易于拟合

#### 2. Coffin-Manson-Arrhenius模型

在Coffin-Manson基础上增加Arrhenius温度加速因子：

$$N_f = A \times (\Delta T_j)^{-\alpha} \times \exp\left(\frac{E_a}{k_B \times T_j}\right)$$

| 参数 | 含义 | 典型值 |
|-----|------|-------|
| $E_a$ | 激活能（eV） | 0.2~0.5 eV |
| $k_B$ | 玻尔兹曼常数 | $8.617 \times 10^{-5}$ eV/K |
| $T_j$ | 平均结温（K） | - |

**适用场景**：考虑温度依赖性的寿命预测

#### 3. Norris-Landzberg模型

增加频率因子，考虑时间相关效应：

$$N_f = A \times (\Delta T_j)^{-\alpha} \times f^{\beta} \times \exp\left(\frac{E_a}{k_B \times T_{j,max}}\right)$$

| 参数 | 含义 | 典型值 |
|-----|------|-------|
| $f$ | 循环频率（Hz） | - |
| $\beta$ | 频率指数 | 通常为正值 |

**适用场景**：需要考虑循环频率影响的场合

#### 4. CIPS 2008模型（推荐）

由Bayerer等人在2008年CIPS会议上提出的最全面的IGBT寿命模型：

$$N_f = K \times (\Delta T_j)^{\beta_1} \times \exp\left(\frac{\beta_2}{T_{j,max}}\right) \times t_{on}^{\beta_3} \times I^{\beta_4} \times V^{\beta_5} \times D^{\beta_6}$$

| 参数 | 含义 | 典型值 | 有效范围 |
|-----|------|-------|---------|
| $K$ | 技术系数 | 需拟合 | - |
| $\Delta T_j$ | 结温波动（K） | - | 60~150 K |
| $T_{j,max}$ | 最高结温（K） | - | 398~523 K (125~250°C) |
| $t_{on}$ | 加热时间（s） | - | 1~60 s |
| $I$ | 负载电流（A） | - | 器件相关 |
| $V$ | 阻断电压（V） | - | 600~1700 V |
| $D$ | 键合线直径（μm） | - | 100~400 μm |
| $\beta_1$ | 温度波动指数 | -4.423 | - |
| $\beta_2$ | 最高温度系数 | 1285 | - |
| $\beta_3$ | 加热时间指数 | -0.462 | - |
| $\beta_4$ | 电流指数 | -0.716 | - |
| $\beta_5$ | 电压指数 | -0.761 | - |
| $\beta_6$ | 键合线直径指数 | -0.5 | - |

**适用场景**：IGBT模块功率循环寿命预测的行业标准

#### 5. LESIT模型

考虑最低温度的影响：

$$N_f = A \times (\Delta T_j)^{\alpha} \times \exp\left(\frac{Q}{R \times T_{j,min}}\right)$$

| 参数 | 含义 | 典型值 |
|-----|------|-------|
| $Q$ | 激活能（J/mol） | - |
| $R$ | 气体常数 | 8.314 J/(mol·K) |
| $T_{j,min}$ | 最低结温（K） | - |

**适用场景**：最低温度对失效有显著影响的情况

---

### 雨流计数法

#### 原理介绍

雨流计数法（Rainflow Counting）是一种从不规则载荷/温度历程中提取循环的算法。该方法由Matsuishi和Endo于1968年提出，现已成为**ASTM E1049**标准。

#### 核心思想

想象雨水流过一系列屋顶（载荷历程的峰谷），当雨水滴落时：
1. 从每个峰/谷开始流动
2. 当遇到更大的峰/谷时停止
3. 当遇到之前流过的路径时停止

这样可以将不规则的载荷历程分解为一系列完整的循环和半循环。

#### 三点法算法（ASTM E1049 §5.4.4）

```
设三个连续的转折点为 A, B, C
Y = |B - A|  (前一范围)
X = |C - B|  (当前范围)

如果 X ≥ Y：
    - 计数 Y 为一个循环
    - 如果 Y 从第一个转折点开始，计为半循环(0.5)
    - 否则计为全循环(1.0)
    - 移除 A 和 B，重新检查
否则：
    - 移动到下一个转折点
```

#### 输出结果

| 输出 | 含义 |
|-----|------|
| `range` | 循环幅值（峰-峰值） |
| `mean` | 循环均值 |
| `count` | 循环计数（0.5为半循环，1.0为全循环） |
| `min_val` | 循环最小值 |
| `max_val` | 循环最大值 |

---

### 功率损耗到结温的转换

在实际应用中，我们通常获得的是**功率损耗时间序列**（P(t)），而不是直接测量的结温。因此需要通过**热模型**将功率转换为结温，然后再进行雨流计数分析。

#### 热阻抗的基本概念

热阻抗（Thermal Impedance）$Z_{th}(t)$ 描述了热流从热源（芯片）传导到环境的热阻随时间的变化特性：

$$Z_{th}(t) = \frac{T_j(t) - T_{amb}}{P}$$

| 参数 | 含义 | 单位 |
|-----|------|------|
| $Z_{th}(t)$ | 瞬态热阻抗 | K/W |
| $T_j(t)$ | 结温 | °C 或 K |
| $T_{amb}$ | 环境温度 | °C 或 K |
| $P$ | 功率损耗 | W |

#### Foster热网络模型

IGBT模块的热特性通常用**Foster RC网络**来建模，这是一个由多个RC并联支路串联组成的电路类比模型：

$$Z_{th}(t) = \sum_{i=1}^{n} R_i \cdot \left(1 - e^{-t/\tau_i}\right)$$

| 参数 | 含义 | 说明 |
|-----|------|------|
| $R_i$ | 第i级热阻 | K/W，表示各级的热阻 |
| $\tau_i$ | 第i级时间常数 | s，$\tau_i = R_i \times C_i$ |
| $C_i$ | 第i级热容 | J/K，表示各级的热容 |
| $n$ | RC级数 | 通常4~6级 |

**Foster模型的特点**：
- 参数可以从热阻抗测量曲线拟合获得
- 计算简单，适合数值仿真
- 各级参数**没有直接物理意义**（不能对应到具体的物理层）
- 是**数学等效模型**，非物理模型

**典型IGBT模块的Foster参数示例**：

| 级数 | $R_i$ (K/W) | $\tau_i$ (s) | 对应物理层 |
|-----|-------------|--------------|-----------|
| 1 | 0.05 | 0.001 | 芯片 |
| 2 | 0.08 | 0.01 | 芯片-基板界面 |
| 3 | 0.12 | 0.1 | 基板 |
| 4 | 0.15 | 1.0 | 散热器界面 |
| 5 | 0.20 | 10.0 | 散热器 |

#### Cauer热网络模型（物理模型）

与Foster模型不同，**Cauer网络**是基于物理结构的热模型：

$$Z_{th}(s) = \frac{1}{sC_1 + \frac{1}{R_1 + \frac{1}{sC_2 + \frac{1}{R_2 + ...}}}}$$

**Cauer模型的特点**：
- 每一级对应一个实际的物理层（芯片、焊料、基板、散热器等）
- 参数有明确的物理意义
- 适合用于分析各层温度分布
- 计算相对复杂

#### 功率-温度卷积计算

利用叠加原理，结温响应可以通过功率与热响应的**卷积**计算：

##### 连续时间形式

$$T_j(t) = T_{amb} + \int_0^t P(\tau) \cdot h(t - \tau) \, d\tau$$

其中 $h(t)$ 是**脉冲热响应**（Impulse Response），与阶跃响应的关系为：
$$h(t) = \frac{dZ_{th}(t)}{dt}$$

##### 离散时间形式

对于离散采样数据，使用离散卷积：

$$T_j[n] = T_{amb} + \Delta t \cdot \sum_{k=0}^{n} P[k] \cdot h[n-k]$$

或者使用阶跃响应的差分形式：

$$T_j[n] = T_{amb} + \sum_{k=0}^{n} P[k] \cdot \left(Z_{th}[n-k] - Z_{th}[n-k-1]\right)$$

#### Foster模型的递推计算

对于实时应用，使用**状态空间递推**更高效，避免存储完整历史数据：

$$T_i[n] = T_i[n-1] \cdot e^{-\Delta t/\tau_i} + R_i \cdot \left(1 - e^{-\Delta t/\tau_i}\right) \cdot P[n]$$

$$T_j[n] = T_{amb} + \sum_{i=1}^{n} T_i[n]$$

**递推计算的优势**：
- 只需存储上一时刻的状态
- 计算复杂度 O(n) 而非 O(n²)
- 适合嵌入式系统和实时监测

#### 多热源耦合模型

实际IGBT模块中，IGBT芯片和二极管芯片通常是**热耦合**的：

$$\begin{bmatrix} T_{j,IGBT}(t) \\ T_{j,Diode}(t) \end{bmatrix} = \begin{bmatrix} Z_{th,11}(t) & Z_{th,12}(t) \\ Z_{th,21}(t) & Z_{th,22}(t) \end{bmatrix} * \begin{bmatrix} P_{IGBT}(t) \\ P_{Diode}(t) \end{bmatrix} + \begin{bmatrix} T_{amb} \\ T_{amb} \end{bmatrix}$$

| 热阻抗 | 含义 |
|--------|------|
| $Z_{th,11}$ | IGBT自热（IGBT功率→IGBT温升） |
| $Z_{th,22}$ | 二极管自热 |
| $Z_{th,12}$ | 耦合热阻抗（二极管功率→IGBT温升） |
| $Z_{th,21}$ | 耦合热阻抗（IGBT功率→二极管温升） |

#### 完整的功率循环分析流程

```
┌─────────────────┐
│  负载电流 i(t)  │
│  母线电压 Vdc   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   功率损耗计算   │  P(t) = Vce(sat)×Ic + Esw×fsw
│   导通+开关损耗  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   热模型计算     │  Tj(t) = Tamb + Zth(t) * P(t)
│   Foster/Cauer  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   结温历程 Tj(t) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   雨流计数分析   │  提取 ΔTj, Tj_mean, n
│   ASTM E1049    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   寿命模型计算   │  计算 Nf(ΔTj, Tj_max, ...)
│   CIPS 2008     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Miner损伤累积  │  D = Σ(ni/Nfi)
│   失效评估      │
└─────────────────┘
```

#### 功率损耗的计算方法

##### 导通损耗

$$P_{cond} = V_{CE(sat)} \times I_C \times D$$

或更精确地使用输出特性曲线：

$$P_{cond} = V_{CE0} \times I_C + r_{CE} \times I_C^2$$

| 参数 | 含义 |
|-----|------|
| $V_{CE0}$ | 阈值电压（约0.5~1.5V） |
| $r_{CE}$ | 斜率电阻 |
| $D$ | 占空比 |

##### 开关损耗

$$P_{sw} = (E_{on} + E_{off}) \times f_{sw}$$

| 参数 | 含义 |
|-----|------|
| $E_{on}$ | 开通能量（J） |
| $E_{off}$ | 关断能量（J） |
| $f_{sw}$ | 开关频率（Hz） |

开关能量与电流、电压、结温相关：

$$E_{sw} = E_{sw,nom} \times \frac{I_C}{I_{nom}} \times \frac{V_{CE}}{V_{nom}} \times f(T_j)$$

#### API使用示例

```python
import requests

# 1. 定义功率损耗历程
power_curve = [100, 150, 200, 150, 100, 50, 100, 150, 200]  # W

# 2. 定义Foster热网络参数（来自数据手册或拟合）
foster_params = [
    {"R": 0.05, "tau": 0.001},
    {"R": 0.08, "tau": 0.01},
    {"R": 0.12, "tau": 0.1},
    {"R": 0.15, "tau": 1.0},
    {"R": 0.20, "tau": 10.0}
]

# 3. 计算结温历程
response = requests.post(
    "http://localhost:8000/api/rainflow/junction-temperature",
    json={
        "power_curve": power_curve,
        "foster_params": foster_params,
        "ambient_temperature": 25.0,
        "dt": 0.1  # 采样间隔 0.1s
    }
)

tj_curve = response.json()["tj_curve"]
print(f"最高结温: {max(tj_curve):.1f}°C")
print(f"结温波动: {max(tj_curve) - min(tj_curve):.1f}K")

# 4. 对结温进行雨流计数
rainflow_response = requests.post(
    "http://localhost:8000/api/rainflow/count",
    json={
        "data_points": [{"time": i, "value": tj} for i, tj in enumerate(tj_curve)],
        "bin_count": 64
    }
)

cycles = rainflow_response.json()["cycles"]
```

---

### Miner线性损伤累积

#### Palmgren-Miner法则

假设损伤线性累积，当总损伤达到1时发生失效：

$$D = \sum_{i=1}^{n} \frac{n_i}{N_{f,i}}$$

| 变量 | 含义 |
|-----|------|
| $D$ | 累积损伤指数 |
| $n_i$ | 第i级应力/温度下的实际循环数 |
| $N_{f,i}$ | 第i级应力/温度下的失效循环数 |

#### 失效判据

- **D < 1.0**：尚未达到失效
- **D ≥ 1.0**：预测失效
- **剩余寿命分数** = 1 - D

#### 应用流程

```
1. 从温度历程提取循环（雨流计数）
2. 对每个循环计算Nf（使用寿命模型）
3. 计算每个循环的损伤贡献：n_i / Nf_i
4. 累加所有损伤贡献
5. 评估是否达到临界损伤
```

---

### Weibull可靠性分析

#### Weibull分布

用于描述失效时间分布的概率分布：

**概率密度函数（PDF）**：

$$f(t) = \frac{\beta}{\eta} \left(\frac{t - \gamma}{\eta}\right)^{\beta - 1} \exp\left[-\left(\frac{t - \gamma}{\eta}\right)^{\beta}\right]$$

**累积分布函数（CDF）**：

$$F(t) = 1 - \exp\left[-\left(\frac{t - \gamma}{\eta}\right)^{\beta}\right]$$

**可靠度函数**：

$$R(t) = \exp\left[-\left(\frac{t - \gamma}{\eta}\right)^{\beta}\right]$$

| 参数 | 含义 | 说明 |
|-----|------|------|
| $\beta$ | 形状参数 | β < 1：早期失效；β = 1：随机失效；β > 1：磨损失效 |
| $\eta$ | 尺度参数 | 特征寿命，63.2%失效时的寿命 |
| $\gamma$ | 位置参数 | 最小寿命，通常为0 |

#### B寿命计算

$$B(P) = \eta \times \left[-\ln(1 - P)\right]^{1/\beta}$$

| 指标 | 含义 |
|-----|------|
| **B10** | 10%失效时的寿命（90%可靠度） |
| **B50** | 50%失效时的寿命（中位寿命） |
| **B63.2** | 63.2%失效时的寿命（特征寿命，等于η） |

#### 形状参数β的物理意义

| β值范围 | 失效类型 | 特征 |
|--------|---------|------|
| β < 1 | 早期失效（婴儿期） | 失效率随时间降低 |
| β ≈ 1 | 随机失效 | 失效率恒定（指数分布） |
| 1 < β < 2 | 早期磨损 | 失效率缓慢增加 |
| β ≈ 2 | 线性磨损 | 失效率线性增加 |
| β > 2 | 加速磨损 | 失效率快速增加 |
| β ≈ 3.5 | 近似正态分布 | - |

---

### 敏感性分析

#### 单参数敏感性

计算弹性系数（Elasticity）：

$$E = \frac{\%\Delta Output}{\%\Delta Input} = \frac{\Delta Y / Y}{\Delta X / X}$$

- |E| > 1：弹性（敏感）
- |E| < 1：非弹性（不敏感）

#### 龙卷风图（Tornado Diagram）

将各参数按影响程度排序，直观显示：
- 每个参数在最小值时的输出
- 每个参数在最大值时的输出
- 以基准值为中心的双向条形图

#### Sobol全局敏感性分析

将输出方差分解为各参数的贡献：

- **一阶指数 $S_i$**：参数i的独立贡献
- **全阶指数 $S_{Ti}$**：参数i及其交互作用的总贡献

$$S_i = \frac{V_i}{V(Y)}, \quad S_{Ti} = 1 - \frac{V_{\sim i}}{V(Y)}$$

---

### 参数拟合

#### 非线性最小二乘法

最小化残差平方和：

$$\min_{\theta} \sum_{i=1}^{n} (y_i - f(x_i, \theta))^2$$

#### 拟合优度指标

| 指标 | 公式 | 说明 |
|-----|------|------|
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 决定系数，越接近1越好 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 均方根误差，越小越好 |
| **95% CI** | $\hat{\theta} \pm 1.96 \times SE(\hat{\theta})$ | 置信区间 |

#### CIPS 2008专用拟合

在**对数空间**进行拟合以保证数值稳定性：

$$\ln(N_f) = \ln(K) + \beta_1 \ln(\Delta T_j) + \frac{\beta_2}{T_{j,max}} + \beta_3 \ln(t_{on}) + \beta_4 \ln(I) + \beta_5 \ln(V) + \beta_6 \ln(D)$$

---

## 安装教程

### 环境要求

| 组件 | 版本要求 |
|-----|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| npm/yarn | 最新稳定版 |
| 操作系统 | Windows 10/11, Linux, macOS |

### 详细安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/your-repo/IOL_PC_research.git
cd IOL_PC_research
```

#### 2. 后端安装

##### Windows

```powershell
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "from app.core.models.cips_2008 import CIPS2008Model; print('OK')"
```

##### Linux/macOS

```bash
cd backend

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "from app.core.models.cips_2008 import CIPS2008Model; print('OK')"
```

#### 3. 前端安装

```bash
cd frontend

# 安装依赖
npm install

# 验证安装
npm run build
```

#### 4. 启动服务

##### 启动后端

```bash
cd backend
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows

python run.py
```

后端服务将在 `http://localhost:8000` 启动

API文档地址：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

##### 启动前端

```bash
cd frontend
npm run dev
```

前端界面将在 `http://localhost:5173` 打开

#### 5. 验证安装

访问 `http://localhost:5173`，应该能看到主界面。

---

## 使用教程

### 示例1：基础寿命预测

使用CIPS 2008模型预测IGBT模块寿命：

```python
import requests

# API请求
response = requests.post(
    "http://localhost:8000/api/prediction/calculate",
    json={
        "model_type": "cips-2008",
        "parameters": {
            "delta_Tj": 80,       # 结温波动 80K
            "Tj_max": 398,        # 最高结温 398K (125°C)
            "t_on": 1.0,          # 加热时间 1s
            "I": 100,             # 负载电流 100A
            "V": 1200,            # 阻断电压 1200V
            "D": 300              # 键合线直径 300μm
        },
        "safety_factor": 1.0
    }
)

result = response.json()
print(f"预测寿命: {result['cycles_to_failure']:.2e} 循环")
```

### 示例2：雨流计数分析

从温度历程提取循环：

```python
import requests

# 温度历程数据
temperature_data = [
    {"time": 0, "value": 25},
    {"time": 1, "value": 85},
    {"time": 2, "value": 35},
    {"time": 3, "value": 95},
    {"time": 4, "value": 45},
    {"time": 5, "value": 105},
    {"time": 6, "value": 55},
    {"time": 7, "value": 25}
]

response = requests.post(
    "http://localhost:8000/api/rainflow/count",
    json={
        "data_points": temperature_data,
        "bin_count": 64
    }
)

cycles = response.json()["cycles"]
for cycle in cycles:
    print(f"幅值: {cycle['range']:.1f}K, 均值: {cycle['mean']:.1f}K, 计数: {cycle['count']}")
```

### 示例3：损伤累积计算

```python
import requests

# 定义任务剖面
mission_profile = [
    {"range": 80, "mean": 65, "count": 1000},   # 大循环
    {"range": 40, "mean": 45, "count": 10000},  # 中循环
    {"range": 20, "mean": 35, "count": 50000}   # 小循环
]

response = requests.post(
    "http://localhost:8000/api/damage/calculate",
    json={
        "cycles": mission_profile,
        "model_type": "coffin-manson",
        "model_params": {
            "A": 1e6,
            "alpha": 2.5
        }
    }
)

damage = response.json()
print(f"累积损伤: {damage['total_damage']:.4f}")
print(f"剩余寿命分数: {damage['remaining_life_fraction']:.2%}")
print(f"状态: {'失效' if damage['is_critical'] else '正常'}")
```

### 示例4：Weibull分析

```python
import requests

# 失效数据（循环数）
failure_data = [100000, 150000, 200000, 250000, 300000,
                350000, 400000, 450000, 500000, 600000]

response = requests.post(
    "http://localhost:8000/api/analysis/weibull/fit",
    json={
        "failure_data": failure_data,
        "confidence_level": 0.9
    }
)

result = response.json()
print(f"形状参数 β: {result['shape']:.3f}")
print(f"尺度参数 η: {result['scale']:.0f}")
print(f"B10寿命: {result['b10_life']:.0f} 循环")
print(f"B50寿命: {result['b50_life']:.0f} 循环")
print(f"拟合优度 R²: {result['r_squared']:.4f}")
```

### 示例5：参数拟合

使用实验数据拟合CIPS 2008模型参数：

```python
import requests

# 实验数据
experiment_data = [
    {"dTj": 80, "Tj_max": 125, "t_on": 1, "I": 100, "V": 1200, "D": 300, "Nf": 500000},
    {"dTj": 100, "Tj_max": 150, "t_on": 1, "I": 100, "V": 1200, "D": 300, "Nf": 150000},
    {"dTj": 60, "Tj_max": 100, "t_on": 1, "I": 100, "V": 1200, "D": 300, "Nf": 2000000},
    {"dTj": 80, "Tj_max": 175, "t_on": 1, "I": 100, "V": 1200, "D": 300, "Nf": 300000},
]

response = requests.post(
    "http://localhost:8000/api/analysis/fitting/cips2008",
    json={
        "experiment_data": experiment_data,
        "fixed_params": {"β3": -0.462, "β4": -0.716, "β5": -0.761, "β6": -0.5}
    }
)

result = response.json()
print(f"拟合参数:")
print(f"  K = {result['parameters']['K']:.4e}")
print(f"  β1 = {result['parameters']['β1']:.4f}")
print(f"  β2 = {result['parameters']['β2']:.2f}")
print(f"R² = {result['r_squared']:.4f}")
```

### 示例6：敏感性分析

```python
import requests

response = requests.post(
    "http://localhost:8000/api/prediction/sensitivity",
    json={
        "model_type": "cips-2008",
        "base_params": {
            "delta_Tj": 80, "Tj_max": 398, "t_on": 1.0, "I": 100, "V": 1200, "D": 300
        },
        "param_ranges": {
            "delta_Tj": [40, 120],
            "Tj_max": [350, 450],
            "t_on": [0.1, 10],
            "I": [50, 200]
        }
    }
)

tornado = response.json()["tornado_data"]
for item in tornado:
    print(f"{item['parameter']}: 影响 {item['range_width']:.2e} ({item['percent_change']:+.1f}%)")
```

---

## API参考

### 基础URL

```
http://localhost:8000/api
```

### 寿命预测API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/prediction/calculate` | 单模型寿命预测 |
| POST | `/prediction/compare` | 多模型对比 |
| POST | `/prediction/sensitivity` | 敏感性分析 |
| GET | `/prediction/models/available` | 可用模型列表 |
| GET | `/prediction/models/{name}` | 模型详情 |

#### 请求示例

```json
// POST /prediction/calculate
{
    "model_type": "cips-2008",
    "parameters": {
        "delta_Tj": 80,
        "Tj_max": 398,
        "t_on": 1.0,
        "I": 100,
        "V": 1200,
        "D": 300
    },
    "safety_factor": 1.0
}
```

#### 响应示例

```json
{
    "cycles_to_failure": 1234567.89,
    "model_used": "CIPS-2008",
    "parameters": { ... },
    "confidence_interval": [1000000, 1500000]
}
```

### 雨流计数API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/rainflow/count` | 雨流循环提取 |
| POST | `/rainflow/histogram` | 直方图数据 |
| POST | `/rainflow/matrix` | 循环矩阵生成 |
| POST | `/rainflow/equivalent` | 等效恒幅计算 |
| POST | `/rainflow/pipeline` | 完整处理流水线 |

### 损伤分析API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/damage/calculate` | Miner损伤计算 |
| POST | `/damage/remaining-life` | 剩余寿命评估 |
| POST | `/damage/safety-margin/calculate` | 安全裕度计算 |
| POST | `/damage/lifetime-curve/generate` | 寿命曲线生成 |

### Weibull分析与拟合API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/analysis/weibull/fit` | Weibull拟合 |
| POST | `/analysis/weibull/b-life` | B寿命计算 |
| POST | `/analysis/weibull/reliability` | 可靠度计算 |
| POST | `/analysis/fitting/fit-model` | 通用模型拟合 |
| POST | `/analysis/fitting/cips2008` | CIPS 2008专用拟合 |

### 数据导出API

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/export/report/pdf` | PDF报告生成 |
| POST | `/export/report/excel` | Excel报告 |
| GET | `/export/prediction/{id}/pdf` | 预测PDF报告 |

---

## 项目结构

```
IOL_PC_research/
├── backend/                         # FastAPI后端
│   ├── app/
│   │   ├── core/                    # 核心算法
│   │   │   ├── models/              # 寿命模型
│   │   │   │   ├── cips_2008.py     # CIPS 2008模型
│   │   │   │   ├── coffin_manson.py # Coffin-Manson模型
│   │   │   │   ├── coffin_manson_arrhenius.py
│   │   │   │   ├── norris_landzberg.py
│   │   │   │   ├── lesit.py
│   │   │   │   ├── model_base.py    # 模型基类
│   │   │   │   └── model_factory.py # 工厂模式
│   │   │   ├── rainflow.py          # 雨流计数
│   │   │   ├── damage_accumulation.py # Miner损伤
│   │   │   ├── remaining_life.py    # 剩余寿命
│   │   │   ├── safety_margin.py     # 安全裕度
│   │   │   ├── weibull.py           # Weibull分析
│   │   │   ├── sensitivity.py       # 敏感性分析
│   │   │   ├── fitting.py           # 参数拟合
│   │   │   └── export/              # 导出模块
│   │   │       ├── pdf_generator.py
│   │   │       └── excel_generator.py
│   │   ├── api/                     # API端点
│   │   │   ├── prediction.py
│   │   │   ├── rainflow.py
│   │   │   ├── damage.py
│   │   │   ├── analysis.py
│   │   │   ├── experiments.py
│   │   │   └── export.py
│   │   ├── models/                  # SQLAlchemy数据模型
│   │   ├── schemas/                 # Pydantic验证模式
│   │   └── db/                      # 数据库配置
│   ├── tests/                       # 单元测试 (555用例)
│   │   ├── test_models/             # 寿命模型测试
│   │   ├── test_core/               # 核心算法测试
│   │   └── test_api/                # API测试
│   ├── requirements.txt
│   └── run.py
│
├── frontend/                        # React前端
│   ├── src/
│   │   ├── components/              # React组件
│   │   │   ├── Prediction/          # 寿命预测
│   │   │   ├── Rainflow/            # 雨流计数
│   │   │   ├── DamageAccumulation/  # 损伤累积
│   │   │   ├── RemainingLife/       # 剩余寿命
│   │   │   └── Visualization/       # 可视化
│   │   ├── pages/                   # 页面
│   │   │   ├── Home.tsx
│   │   │   ├── Prediction.tsx
│   │   │   ├── RainflowCounting.tsx
│   │   │   ├── DamageAccumulation.tsx
│   │   │   ├── Analysis.tsx
│   │   │   ├── ParameterFitting.tsx
│   │   │   └── RemainingLife.tsx
│   │   ├── services/                # API服务
│   │   ├── stores/                  # Zustand状态
│   │   └── types/                   # TypeScript类型
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

---

## 技术栈

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.10+ | 核心语言 |
| **FastAPI** | 0.115 | Web框架，自动API文档 |
| **SQLAlchemy** | 2.0 | ORM数据库访问 |
| **NumPy** | 2.1 | 数值计算 |
| **SciPy** | 1.14 | 科学计算、优化拟合 |
| **scikit-learn** | 1.5 | 机器学习工具 |
| **reportlab** | 4.2 | PDF报告生成 |
| **openpyxl** | 3.1 | Excel文件处理 |
| **Pydantic** | 2.x | 数据验证 |
| **pytest** | 8.x | 单元测试框架 |

### 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **React** | 18.2 | UI框架 |
| **TypeScript** | 5.2 | 类型安全 |
| **MUI** | 5.15 | Material Design组件库 |
| **ECharts** | 5.4 | 数据可视化图表 |
| **Zustand** | 4.4 | 轻量状态管理 |
| **Axios** | 1.6 | HTTP客户端 |
| **Vite** | 5.0 | 构建工具 |

---

## 测试

### 运行测试

```bash
cd backend

# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows

# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_models/test_cips_2008.py -v
pytest tests/test_core/test_rainflow.py -v

# 生成覆盖率报告
pytest tests/ --cov=app --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html  # macOS
# 或
start htmlcov/index.html  # Windows
```

### 测试覆盖

| 模块 | 测试文件 | 用例数 |
|------|----------|--------|
| 雨流计数 | `test_rainflow.py` + `test_core/test_rainflow.py` | 120+ |
| 损伤累积 | `test_damage_accumulation.py` + `test_core/test_damage_accumulation.py` | 90+ |
| 寿命模型 | `test_models/` (5个模型) | 80+ |
| Weibull分析 | `test_core/test_weibull.py` | 52 |
| 敏感性分析 | `test_core/test_sensitivity.py` | 65 |
| 参数拟合 | `test_core/test_fitting.py` | 48 |
| 剩余寿命 | `test_remaining_life.py` | 50+ |
| 安全裕度 | `test_safety_margin.py` | 50+ |
| **总计** | - | **555+** |

---

## 开发指南

### 添加新的寿命模型

1. 在 `backend/app/core/models/` 创建新模型文件：

```python
# my_model.py
from app.core.models.model_base import LifetimeModelBase

class MyModel(LifetimeModelBase):
    """自定义寿命模型"""

    def __init__(self, param1: float = 1.0, param2: float = 2.0):
        self.param1 = param1
        self.param2 = param2

    def get_model_name(self) -> str:
        return "My-Model"

    def calculate_cycles_to_failure(self, **params) -> float:
        # 实现计算逻辑
        delta_Tj = params.get("delta_Tj")
        # ... 计算公式 ...
        return Nf

    def get_equation(self) -> str:
        return "Nf = f(delta_Tj, ...)"

    def get_parameters_info(self) -> dict:
        return {
            "param1": {"description": "...", "typical_range": "..."},
            # ...
        }
```

2. 在 `model_factory.py` 注册模型：

```python
ModelFactory.register_model("my-model", MyModel)
```

3. 添加对应的测试文件：

```python
# tests/test_models/test_my_model.py
import pytest
from app.core.models.my_model import MyModel

class TestMyModel:
    def test_calculate_cycles_basic(self):
        model = MyModel()
        result = model.calculate_cycles_to_failure(delta_Tj=80)
        assert result > 0

    # 更多测试...
```

### 添加新的API端点

1. 在 `backend/app/api/` 创建或修改路由文件：

```python
# my_api.py
from fastapi import APIRouter, Depends
from app.schemas.my_schema import MyRequest, MyResponse

router = APIRouter(prefix="/my-endpoint", tags=["My Feature"])

@router.post("/action", response_model=MyResponse)
async def my_action(request: MyRequest):
    """执行某个操作"""
    # 实现逻辑
    return MyResponse(result=...)
```

2. 在 `main.py` 中注册路由：

```python
from app.api.my_api import router as my_router
app.include_router(my_router, prefix="/api")
```

3. 添加对应的Pydantic schema：

```python
# schemas/my_schema.py
from pydantic import BaseModel, Field

class MyRequest(BaseModel):
    param1: float = Field(..., description="参数1")
    param2: float = Field(..., description="参数2")

class MyResponse(BaseModel):
    result: float = Field(..., description="计算结果")
```

### 代码规范

- **Python**: 遵循 PEP 8，使用 Black 格式化
- **TypeScript**: ESLint + Prettier
- **提交信息**: Conventional Commits 格式
- **文档字符串**: Google 风格

---

## 参考文献

### 核心论文

1. **Bayerer, R., et al.** (2008). "Model for Power Cycling Lifetime of IGBT Modules - various factors influencing lifetime", *CIPS 2008, 6th International Conference on Integrated Power Electronic Systems*, pp. 1-6. [PDF](./2008_CIPS_ModelforPowerCyclinglifetimeofIGBTModulesvariousfactorsinfluencinglifetime.pdf)

2. **ASTM E1049-85(2017)**. "Standard Practices for Cycle Counting in Fatigue Analysis", ASTM International. DOI: 10.1520/E1049-85R17

3. **Miner, M.A.** (1945). "Cumulative Damage in Fatigue", *Journal of Applied Mechanics*, 12(3), A159-A164.

4. **Coffin, L.F.** (1954). "A Study of the Effects of Cyclic Thermal Stresses on a Ductile Metal", *Transactions of ASME*, 76, 931-950.

5. **Manson, S.S.** (1965). "Fatigue: A Complex Subject—Some Simple Approximations", *Experimental Mechanics*, 5(7), 193-226.

6. **Norris, K.C. & Landzberg, A.H.** (1969). "Reliability of Controlled Collapse Interconnections", *IBM Journal of Research and Development*, 13(3), 266-271.

### 推荐阅读

- 功率半导体器件封装、测试和可靠性（参考书籍）
- IEC 61714: Reliability testing - Failure rates
- Weibull分析在可靠性工程中的应用

---

## License

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request。请遵循以下准则：

1. Fork 项目并创建功能分支
2. 确保所有测试通过
3. 添加必要的测试用例
4. 更新相关文档
5. 提交 Pull Request

---

<p align="center">
  Made with ❤️ for Power Electronics Reliability Engineering
</p>
