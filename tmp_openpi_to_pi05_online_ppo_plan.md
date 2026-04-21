# tmp-openpi pi0fast 在线 PPO 迁移计划（到 RoboTwin/policy/pi05）

## 0. 本次迁移硬约束（已确认）

1. 只要在线 PPO 逻辑，不迁移其他改造能力。
2. 配置体系完全以 `RoboTwin/policy/pi05` 现状为准（不引入 tmp-openpi 的 config 语义）。
3. websocket 协议完全以 `RoboTwin/policy/pi05` 当前链路为准（不引入新协议）。
4. 首阶段仅单环境（`num_envs=1`）。
5. 需要支持 `repo_root` 本地数据根目录。
6. reward 先用 `fixed_reward`。
7. keyframe exploration 先固定为“始终开启”（等价于 always-on gate）。
8. 现有文件尽量不动；在线 PPO 相关能力优先全部新建文件承载。

## 1. 当前链路思路梳理（tmp-openpi）

主入口是：
- `tmp-openpi/scripts/train_pi0_fast_online.py`

主数据流：
1. 加载训练配置与 checkpoint，强制开启 `Pi0FASTConfig.use_value_head=True`（可选 `use_keyframe_head`）。
2. 构建 RL policy 包装器 `Pi0FastRLPolicy`，它在标准 `policy.infer` 外新增了 PPO 所需 trace 能力。
3. 通过 `openpi.envs.make_env(...)` 连接 websocket 向量环境。
4. `Pi0FastChunkCollector` 按 chunk 收集样本，拿到：
   - `action_tokens`、`token_mask`、`old_token_logprobs`
   - `reward`、`value`、`next_value`
   - 以及可选 keyframe/exploration 相关字段
5. `Pi0FastOnlineTrainer` 做在线 PPO：
   - 先算 chunk 级 TD target/advantage
   - actor 与 value 分开更新（value head 参数与 actor 参数分离）
   - 用 `policy_version` 控制近策略样本，过旧样本丢弃
   - 更新后同步参数回 rollout policy

核心模块：
- RL 框架：`tmp-openpi/src/openpi/rl/*`
- 环境适配：`tmp-openpi/src/openpi/envs/websocket_env.py`
- RobotWin 数据变换：`tmp-openpi/src/openpi/policies/robotwin_policy.py`
- 模型增强：`tmp-openpi/src/openpi/models/pi0_fast.py`
- 配置增强：`tmp-openpi/src/openpi/training/config.py`、`data_loader.py`

## 2. 与 RoboTwin/policy/pi05 当前差异

已存在：
- `RoboTwin/policy/pi05/src/openpi` 基础训练/推理框架完整。
- websocket policy server 已有（推理端）。

缺失/不一致（迁移阻塞点）：
1. `src/openpi/rl` 整个目录不存在。
2. `src/openpi/envs` 目录不存在。
3. `src/openpi/policies/robotwin_policy.py` 不存在。
4. `src/openpi/models/pi0_fast.py` 缺少 RL 所需能力：
   - `use_value_head/use_keyframe_head`
   - `sample_actions_with_trace`
   - `recompute_action_logprobs`
   - `predict_value*`、`project_values_to_bins`
5. `src/openpi/training/config.py` 无 `LeRobotRobotWinDataConfig`，也无 `pi0_fast_robotwin/pi05_robotwin` 配置名。
6. `src/openpi/training/data_loader.py` 无 `repo_root` 支持（tmp-openpi 用它读本地 LeRobot 数据根目录）。
7. `scripts/` 下没有在线 PPO 入口脚本。

## 3. 迁移目标（按约束收敛）

阶段 A（最小闭环，全部新建在线 PPO 文件）：
1. 新建在线 PPO 专用命名空间目录（例如 `src/openpi_online_ppo/`），不改现有 `src/openpi/*` 逻辑。
2. 新建在线 PPO 训练入口（例如 `scripts/train_pi0_fast_online_v2.py`）。
3. 在线 PPO 中直接调用 `RoboTwin/policy/pi05/src/openpi` 已有配置加载与模型构建逻辑。
4. 新建 `repo_root` 版数据加载桥接层（不改现有 `training/data_loader.py`）。
5. 奖励先用固定值 reward provider；单环境采样跑通 PPO 更新闭环。
6. keyframe exploration 在新建逻辑里默认 always-on。

阶段 B（后续增强）：
1. 再考虑 reward 接环境真实反馈。
2. 再考虑多环境并行。
3. 再考虑把新建链路回收/并入主 `openpi` 命名空间。

## 4. 文件级迁移清单（全部“新建优先”）

建议新建目录：
1. `RoboTwin/policy/pi05/src/openpi_online_ppo/`
2. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/`
3. `RoboTwin/policy/pi05/src/openpi_online_ppo/env/`
4. `RoboTwin/policy/pi05/src/openpi_online_ppo/data/`

建议新建文件（首批）：
1. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/chunk_types.py`
2. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/chunk_buffer.py`
3. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/ppo_losses.py`
4. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/reward_value.py`（含 fixed_reward provider）
5. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/exploration.py`（默认 always-on 配置）
6. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/pi0_fast_policy.py`
7. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/pi0_fast_rollout.py`
8. `RoboTwin/policy/pi05/src/openpi_online_ppo/rl/pi0_fast_online_trainer.py`
9. `RoboTwin/policy/pi05/src/openpi_online_ppo/env/single_env_ws.py`（协议对齐当前 pi05 websocket 链路）
10. `RoboTwin/policy/pi05/src/openpi_online_ppo/data/local_lerobot_loader.py`（支持 `repo_root`）
11. `RoboTwin/policy/pi05/scripts/train_pi0_fast_online_v2.py`
12. `RoboTwin/policy/pi05/scripts/run_pi0_fast_online_v2.sh`

备注：
- 现有 `eval/train/serve` 文件保持不变。
- 在线 PPO 入口通过新脚本独立运行。

## 5. config 策略（按你要求）

原则：
- 不改现有 `RoboTwin/policy/pi05/src/openpi/training/config.py` 字段语义与主流程配置行为。
- 在线 PPO 脚本直接读取现有配置（例如 `pi0_fast_aloha_robotwin_full`），并在运行时做 PPO 所需增量参数（仅进程内覆盖，不回写配置文件）。

说明：
- 因为“现有文件尽量不动”，这里不新增/修改主 config 文件中的配置项。
- 若后续你同意，可在第二阶段再补正式 `*_online_ppo` 配置名。

## 6. 技术风险与注意事项（基于你的约束）

1. websocket 协议一致性风险：
   - 新链路必须严格复用 `RoboTwin/policy/pi05` 当前协议，不复用 tmp-openpi 的自定义协议。
   - 新建 env client 需要先做协议回归测试。
2. 观测键映射风险：
   - 在线 PPO 使用现有 pi05 配置时，obs/action 键必须与其 transform 链严格一致。
3. 动作维度风险：
   - 环境 action_dim 与 policy 输出 action_dim 必须严格一致（例如 14）。
4. value/keyframe 头初始化风险：
   - 若底座模型未带相应 head，新建链路需在运行时安全初始化并记录日志。
5. 近策略样本管理：
   - `max_policy_lag` 太大容易 off-policy，太小样本利用率低。
6. 训练恢复：
   - 当前链路更偏研究原型，断点恢复和容错还需额外工程化。

## 7. 建议实施顺序（执行版）

1. 新建 `openpi_online_ppo` 目录与脚本，不改旧文件。
2. 先做单环境 + fixed reward + always-on keyframe exploration。
3. 跑 smoke test：
   - 1 env、fixed reward、几十个 update，确认 loss/kl/entropy 正常。
4. 再接真实 reward 与多 env。

## 8. 已决议项（替代不确定项）

1. 仅迁移在线 PPO 逻辑。
2. 配置与 websocket 协议都以 `RoboTwin/policy/pi05` 为准。
3. 需要 `repo_root`。
4. 首阶段 `fixed_reward`。
5. 首阶段 `num_envs=1`。
6. keyframe exploration 首阶段 always-on。

## 9. 仍需最小确认（仅 2 项）

1. “always-on keyframe exploration”是否等价为忽略阈值、每个 chunk 都注入探索噪声？
2. `fixed_reward` 默认值是否先用 `0.0`（纯打通），还是你希望固定为某个正值（如 `1.0`）？
