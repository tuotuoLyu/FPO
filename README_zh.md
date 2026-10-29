<p align="center">
  <img src="assets/banner.png" alt="FPO banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ICRA-2026%20accepted-1f6feb" alt="ICRA 2026">
  <a href="https://arxiv.org/abs/2510.09976"><img src="https://img.shields.io/badge/arXiv-2510.09976-b31b1b" alt="arXiv 2510.09976"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.11-3776ab" alt="Python 3.11">
</p>

<h1 align="center">Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models</h1>

<p align="center">
Mingyang Lyu<sup>a,b,e,1</sup>,
Yinqian Sun<sup>a,b,d,f,1</sup>,
Erliang Lin<sup>a</sup>,
Huangrui Li<sup>d</sup>,
Ruolin Chen<sup>a,b,e</sup>,
Feifei Zhao<sup>a,b,d,f,2</sup>,
Yi Zeng<sup>a,b,c,d,e,f,2</sup>
</p>

<p align="center"><sub>
<sup>a</sup> Brain-inspired Cognitive AI Lab, Institute of Automation, Chinese Academy of Sciences, Beijing, China &nbsp;
<sup>b</sup> Beijing Institute of AI Safety and Governance, China &nbsp;
<sup>c</sup> State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology &nbsp;
<sup>d</sup> Beijing Key Laboratory of Safe AI and Superalignment, China &nbsp;
<sup>e</sup> University of Chinese Academy of Sciences (UCAS), Beijing, China &nbsp;
<sup>f</sup> Long-term AI, Beijing, China
<br>
<sup>1</sup> 共同第一作者 &nbsp; <sup>2</sup> 共同通讯作者
</sub></p>

<p align="center"><b>已被 2026 IEEE International Conference on Robotics and Automation（ICRA 2026，奥地利维也纳）接收。</b><br>
<a href="https://arxiv.org/abs/2510.09976">arXiv:2510.09976</a></p>

<p align="center"><a href="README.md">English</a></p>

**一句话概括**：FPO（Flow Policy Optimization）用"同一样本上条件流匹配（CFM）损失的变化"构造免似然的策略比率，从而对流匹配 VLA 策略 pi0 做 PPO 式的在线强化微调。

<p align="center">
  <img src="assets/fpo_overview.jpg" alt="FPO 方法总览（论文图 1）" width="100%">
  <br><sub>论文图 1：(b) 采样阶段缓存 theta_old 下的逐样本 CFM 损失；(c) 更新阶段在当前 actor 下重算损失，差值经批内标准化映射为比率，Q 集成给出优势，actor 以裁剪目标更新。</sub>
</p>

## 要点

- **免似然比率**：对每个存储的 (s_t, x_t)，Delta l = l_cfm(theta_old) - l_cfm(theta)，批内标准化后 rho = exp(beta z)（式 2-3）。
- **裁剪代理目标**（式 4）+ **Q 集成**（取最小值的保守目标、GAE 优势；式 5-6）。
- **多步潜空间探索**：actor 的潜变量由其流场的 K 步 Euler 积分得到（式 7），再由冻结的 pi0 解码成动作。
- **论文设置与消融均为开关**：LIBERO（四个子集与 LIBERO-90 消融任务）和 ALOHA Transfer Cube；表 II 的每个变体对应一个参数。

论文中的每个公式、实验设置与默认值和代码的对应关系，以及论文未给出的实现细节的取舍，见
[`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md)。

## 演示

<p align="center">
  <img src="assets/fig6_rollouts.gif" alt="论文图 6 的轨迹帧" width="720">
  <br><sub>按顺序播放论文图 6 的四帧：微调前的 pi0（上）侧向抓取失败；FPO 微调后（下）在同一初始状态下改为自上而下抓取并成功。圆圈为论文原图标注。</sub>
</p>

## 结果（来自论文）

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| GRAPE (DPO) | 87.6 | 91.2 | 82.2 | 55.8 | 79.2 |
| Octo (SFT) | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA (SFT) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| VLA-RL | 90.2 | 91.8 | 82.2 | 59.8 | 81.0 |
| pi0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| **pi0-FPO（本文）** | **97.2** | **97.3** | **89.4** | **65.3** | **87.2** |

消融（LIBERO-90 "pick up the butter and put it in the tray"，表 II）：完整 FPO 78.5；去掉 CFM 比率（改用 SAC 式潜空间更新）32.4；去掉 PPO 裁剪 45.1；单步积分（K = 1）61.7；单个 critic 71.2。ALOHA Transfer Cube（图 4）：从约 40% 提升到 65% 以上。

## 安装与运行

安装步骤见 [英文 README](README.md#installation)。

```bash
# CPU 自检（仅需 NumPy，数秒）
python -m unittest discover -s tests -v
python scripts/smoke_cpu.py
python scripts/check_loop_cpu.py

# GPU 训练
bash examples/scripts/run_fpo_libero.sh                  # LIBERO-90 消融任务上的完整 FPO
bash examples/scripts/run_fpo_libero_suite.sh libero_10  # 表 I：逐任务运行一个子集
bash examples/scripts/run_fpo_aloha.sh                   # ALOHA Transfer Cube
bash examples/scripts/run_fpo_ablations.sh all           # 表 II 全部消融
```

批大小（LIBERO 210、ALOHA 192）需能被可见 GPU 数整除；输出目录由 `EXP` 指定，pi0 权重缓存目录由 `OPENPI_DATA_HOME` 指定。

## 引用

```bibtex
@inproceedings{lyu2026fpo,
  title         = {Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models},
  author        = {Lyu, Mingyang and Sun, Yinqian and Lin, Erliang and Li, Huangrui and Chen, Ruolin and Zhao, Feifei and Zeng, Yi},
  booktitle     = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year          = {2026},
  eprint        = {2510.09976},
  archivePrefix = {arXiv},
}
```

## 许可

本仓库以 MIT 许可发布，详见 [`LICENSE`](LICENSE)；第三方组件及其许可见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。
