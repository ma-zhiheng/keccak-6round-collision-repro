# Keccak 5/6 轮碰撞攻击复现

这个仓库用于复现和推进 Song、Liao、Guo 等人关于轮数削减版 Keccak/SHA-3 碰撞攻击的工作。

对应论文：

- Ling Song, Guohong Liao, Jian Guo,
  **"Non-full Sbox Linearization: Applications to Collision Attacks on
  Round-Reduced Keccak"**,
  CRYPTO 2017.
- Jian Guo, Guohong Liao, Guozhen Liu, Meicheng Liu, Kexin Qiao, Ling Song,
  **"Practical Collision Attacks against Round-Reduced SHA-3"**,
  Journal of Cryptology 2020.

这个仓库从已完成的 5 轮 `Keccak[1440,160,5,160]` 复现代码发展而来，进一步加入了面向 6 轮 Keccak challenge 实例的 non-full S-box linearization 和三轮 connector 复现路径。

## 当前状态

已经实现并本地检查的内容：

- Keccak-f[1600] 轮函数和 digest 辅助函数。
- Keccak 5-bit `chi` S-box、DDT、仿射子空间枚举，以及 connector 论文使用的线性化性质。
- 已完成的 5 轮 baseline：`Keccak[1440,160,5,160]`
  - Table 7 trail core No. 2 验证；
  - 二轮 connector 复现；
  - C++/CUDA 后续搜索工具；
  - 保存的碰撞候选：`results/core2_cuda_candidates.txt`；
  - Python 独立验证这些候选。
- 6 轮 `Keccak[1440,160,6,160]` 的 staged reproduction：
  - Journal 版本 Table 18 打印的 6 轮碰撞可被本地 Keccak 实现验证；
  - Table 11 trail core No. 5 已转录并部分验证；
  - 使用论文打印碰撞路径导出一个成功的 first two-round connector；
  - 构造了 conservative third-round connector 和 bitwise non-full third-round connector；
  - Python 直接抽样验证 `R^3(M1) xor R^3(M2) = alpha3`；
  - 可以导出三轮 connector 的 C++/CUDA 搜索数据。

尚未完成的内容：

- 还没有从这个 checkout 上运行长时间 CUDA 搜索来独立找到新的 6 轮碰撞。
- `trail_data_6round.py` 中的 `beta5` 转录被标记为 provisional，当前 connector 代码不会使用它。
- 当前 6 轮路径使用论文打印碰撞来导出成功 connector 路径，还不是从零独立搜索所有 connector 选择。

## 快速检查

运行 5 轮 baseline 的 Python 自检：

```bash
python run_all.py
```

预期关键输出：

```text
2-dimensional linearizable affine subspaces: 80
3-dimensional linearizable affine subspaces: 0
Table 7 core No. 2 connector reproduced
added G rows: 195/195
dimension: 180
verifies R^2(M1)+R^2(M2)=alpha2: True
```

验证保存的 5 轮候选：

```bash
python verify_core2_candidates.py results/core2_cuda_candidates.txt
```

预期结果：所有 candidate 都输出 `ok=True`。

验证论文打印的 6 轮碰撞：

```bash
python verify_paper_6round_collision.py
```

预期关键输出：

```text
collision: True
matches printed digest: True
```

检查 6 轮 trail 转录：

```bash
python trail_data_6round.py
```

预期关键输出：

```text
table11_core5_keccak_1440_160_6_160
beta2->alpha3 compatible: True
beta3->alpha4 compatible: True
beta2->alpha3 weight=25: True
beta3->alpha4 weight=18: True
beta5 transcription is provisional and is not used yet
```

## 复现 6 轮 Connector 路径

构造 paper-derived first two-round connector 和第三轮 non-full connector：

```bash
python derive_paper_first_two_connector.py --build-third
```

预期关键输出：

```text
paper-derived first 2-round connector
  added G rows: 292/292
  rank/dimension: 1420/180
  sample verifies R^2 target: True

paper-derived bitwise third-round connector
  third-round rows: 25
  rank/dimension: 1544/56
  sample verifies R^3 target: True
```

导出原生搜索数据：

```bash
python export_core3_cpp_data.py
```

预期关键输出：

```text
basis size: 56
connector rank/dimension: 1544/56
```

`reproduce_core3_connector.py` 是 staged overview 入口，会打印预估搜索规模并验证当前可用 trail 数据：

```bash
python reproduce_core3_connector.py
```

## 原生搜索

在 Linux 服务器上构建 CPU 和 CUDA 搜索程序：

```bash
make core3-search
make core3-search-cuda
```

CPU 小规模校准：

```bash
./core3_trail_search \
  --samples 10000000 \
  --threads 32 \
  --report 2000000
```

多 GPU CUDA 校准：

```bash
./core3_trail_search_cuda \
  --samples 1000000000 \
  --devices all \
  --max-alpha3-print 0 \
  --threads-per-block 256 \
  --blocks-per-sm 4
```

长搜索建议分块运行，并把候选追加到结果文件：

```bash
./core3_trail_search_cuda \
  --samples 100000000000 \
  --devices all \
  --seed 917306210421 \
  --max-alpha3-print 0 \
  --threads-per-block 256 \
  --blocks-per-sm 4 \
  --candidate-file results/core3_cuda_candidates.txt
```

当前计划数字：

```text
expected 2^47.81 at 1e9/s: about 68.5 hours
paper actual 2^49.07 at 1e9/s: about 164.2 hours
```

这些是概率估计，不是确定倒计时。

## 项目结构

核心模块：

- `keccak_state.py`：Keccak-f[1600] 执行与 digest 辅助函数。
- `sbox_linearization.py`：`chi` S-box、DDT 和仿射线性化检查。
- `sbox_constraints.py`：局部 S-box 转移方程和线性模型。
- `gf2.py`：GF(2) 线性方程系统。
- `linear_layer.py`：Keccak 线性层和逆矩阵辅助函数。
- `state_lift.py`：将局部 S-box 方程提升到 1600-bit 状态坐标。
- `connector_equations.py`：full-linearized 二轮 connector 方程。
- `incremental_connector.py`：增量式 connector 构造工具。
- `core2_connector.py`：已完成的 5 轮 baseline connector。
- `core3_connector.py`：paper-derived 与 bitwise non-full 6 轮 connector 辅助函数。

Trail 与验证数据：

- `trail_data.py`：5 轮 baseline 的 Table 7 core No. 2。
- `trail_data_6round.py`：6 轮目标的 Table 11 core No. 5。
- `paper_collisions.py`：论文打印碰撞数据辅助函数。
- `verify_paper_6round_collision.py`：验证论文打印的 6 轮碰撞。
- `verify_core2_candidates.py`：验证保存的 5 轮候选。

主要复现脚本：

- `run_all.py`：baseline Python 自检。
- `reproduce_core2_connector.py`：确定性复现 5 轮 baseline connector。
- `derive_paper_first_two_connector.py`：导出 6 轮 first two-round connector 并构造第三轮 non-full connector。
- `reproduce_core3_connector.py`：6 轮 staged overview 和 trail 检查。
- `export_core3_cpp_data.py`：导出三轮 connector 供原生搜索使用。

原生搜索：

- `core2_trail_search.cpp`、`core2_trail_search_cuda.cu`：已完成 5 轮后续搜索。
- `core3_trail_search.cpp`、`core3_trail_search_cuda.cu`：6 轮后续搜索。
- `Makefile`：构建入口。

探索脚本：

- `search_core3_beta_pairs.py`
- `search_core3_first_two_parallel.py`
- `search_core3_first_two_repair.py`
- `search_core3_forced_models.py`
- `search_core3_row_reorder.py`
- `search_core3_with_beta_pair.py`
- `repair_core3_with_beta_pair.py`

这些脚本保留用于记录切换到 paper-derived connector 路径之前的探索过程。

## 说明

- 本仓库用于 reduced-round cryptanalysis 学习与复现。
- 完整 24 轮 SHA-3 不受影响。
- 6 轮部分目前是 staged reproduction 和长搜索入口，长时间 CUDA 搜索可以后续在服务器上继续运行。
- `README_6ROUND_PLAN.md` 记录了更详细的 6 轮开发过程。
