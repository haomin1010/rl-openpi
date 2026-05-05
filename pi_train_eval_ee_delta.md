# RoboTwin `beat_block_hammer` EE-Delta 14D 数据处理 + `pi0_fast` 训练评估指南

这份文档是 `/mnt/data/lhm/vla-rl/pi_train_eval.md` 的 EE 模式版本，统一使用：

- 仓库路径：`/mnt/data/lhm/vla-rl`
- 动作/状态表示：EE-Delta 14D
- 训练配置：`full`（不使用 LoRA）
- 主要评估入口：`RoboTwin/policy/pi05/eval.sh`

对应配置定义在：

- `RoboTwin/policy/pi05/src/openpi/training/config.py`
- `pi0_fast_aloha_robotwin_full_ee_delta`
- `pi0_fast_aloha_robotwin_ppo_ee_delta`

当前约定：

- SFT checkpoint 与 PPO 使用同一套标准 FAST tokenizer 路径
- PPO exploration 不再依赖 rowwise / transpose 的特殊 decode 链路
- 在线 PPO / sigma exploration 的命令以 [ONLINE_PPO_WORKFLOW.md](/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/ONLINE_PPO_WORKFLOW.md) 为准

EE-Delta 14D 定义见：

- `RoboTwin/policy/pi05/EE_DELTA_14D_DESIGN.md`

## 1. 准备数据软链接

如果你的原始数据在 `/mnt/data/lhm/test/RoboTwin/data/beat_block_hammer`，先软链接到仓库内：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin
ln -sfn /mnt/data/lhm/test/RoboTwin/data/beat_block_hammer data/beat_block_hammer
```

注意：原始采集阶段不需要额外选择 EE 模式，但 task config 里需要保留：

```yaml
data_type:
  endpose: true
  qpos: true
```

`demo_clean.yml` 当前已经满足这个条件。

## 2. 数据预处理为 EE-Delta 14D

将 RoboTwin 原始数据转换为 EE-Delta 14D 版 processed data：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
python scripts/process_data.py beat_block_hammer demo_clean 100 --representation ee_delta
```

处理完成后目录应为：

```bash
/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/processed_data/beat_block_hammer-demo_clean-100-ee_delta
```

说明：

- 不加 `--representation ee_delta` 时仍然会生成旧的 joint/qpos 版本。
- EE 版 `observations/qpos` 字段名目前沿用旧名，但实际内容是 14D EE observation。
- EE 版 `action` 是 14D delta-EE action。

## 3. 转成训练需要的 LeRobot 数据集

下面示例沿用配置里的默认 `repo_id=lerobot-hammer-100`。如果你想用别的 `repo_id`，需要同步修改 `config.py` 里的 EE config。

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/beat_block_hammer-demo_clean-100-ee_delta \
  --repo-id lerobot-hammer-clean-100-ee_delta \
  --task beat_block_hammer \
  --mode image
```

## 4. 计算归一化统计量

训练前必须先计算，且配置名要和训练时完全一致。

### 4.1 `pi0_fast` EE full

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_full_ee_delta
```

### 4.2 `pi0_fast` EE PPO

如果后续 formal PPO 使用 `pi0_fast_aloha_robotwin_ppo_ee_delta`，也建议计算对应统计量：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_ppo_ee_delta
```

## 5. 训练 `pi0_fast`（EE full）

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash finetune.sh pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 4,5,6,7
```

训练输出目录：

```bash
/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta
```

默认最终 checkpoint 步数：

```bash
30000
```

## 6. 评估 `pi0_fast`（EE full）

### 6.1 在 `demo_clean` 上评估

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 0 0
```

### 6.2 在 `demo_randomized` 上评估

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_randomized pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 0 0
```

## 7. 结果位置

### 7.1 训练 checkpoint

```bash
RoboTwin/policy/pi05/checkpoints/<train_config_name>/<model_name>/
```

### 7.2 评估结果

```bash
RoboTwin/eval_result/beat_block_hammer/pi05/<task_config>/<model_name>/<timestamp>/
```

评估文本通常在：

```bash
RoboTwin/eval_result/.../_result.txt
```

## 8. 一套最常用的完整命令顺序

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin
ln -sfn /mnt/data/lhm/test/RoboTwin/data/beat_block_hammer data/beat_block_hammer

cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
python scripts/process_data.py beat_block_hammer demo_clean 100 --representation ee_delta

uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/beat_block_hammer-demo_clean-100-ee_delta \
  --repo-id lerobot-hammer-clean-100-ee_delta \
  --task beat_block_hammer \
  --mode image

uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_full_ee_delta
bash finetune.sh pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 4,5,6,7
bash eval.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 0 0
```

## 9. WebSocket Eval

启动 policy server：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config=pi0_fast_aloha_robotwin_full_ee_delta \
  --policy.dir=checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/
```

WebSocket eval：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval_ws.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full_ee_delta beat_block_hammer_pi0fast_full_ee_delta 0 4 127.0.0.1 8000
```

## 10. 补充说明

- `compute_norm_stats` 的配置名必须与 `finetune.sh` 训练配置名一致。
- `eval.sh` 里的 `model_name` 必须与训练时 `finetune.sh` 的第二个参数一致。
- 如果你改了 `repo_id`，要同步修改 `config.py` 中 EE 训练配置的 `repo_id`。
- `process_data.py --representation ee_delta` 是离线 EE 数据的关键开关。
- 在线 PPO / 在线 rollout / exploration net 统一以 [ONLINE_PPO_WORKFLOW.md](/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/ONLINE_PPO_WORKFLOW.md) 为准。

## 11. 当前在线 PPO / Sigma Net 常用命令

如果你已经完成了 SFT checkpoint、在线 corpus 合并、keyphase 标注和值头训练，当前推荐的 sigma net 训练命令是：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run python scripts/fit_dimwise_sigma_net_lerobot.py \
  --dataset_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus \
  --repo_id lerobot_ppo_corpus \
  --annotations_json /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus/keyphases.json \
  --out checkpoints/explored_net/dimwise_sigma_dct.pkl \
  --chunk_size 32 \
  --action_noise_dims "0,1,2,3,4,5,6" \
  --action_delta_limit 0.1 \
  --noise_dct_keep_k 4 \
  --epochs 100 \
  --batch_size 128
```

语义：

- 输入是 `C[:4, selected_action_dims]` 的 flatten 结果
- 输出是对应位置的 `sigma`
- 不是方差

当前推荐的 sigma-net PPO 命令是：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run python scripts/train_pi0_fast_online_v2.py \
  --policy.path checkpoints/pi0_fast_aloha_robotwin_full_ee_delta/beat_block_hammer_pi0fast_full_ee_delta/30000/ \
  --policy.config pi0_fast_aloha_robotwin_ppo_ee_delta \
  --env.ws_url ws://127.0.0.1:8765 \
  --total_updates 10 \
  --rollout_batch_size 64 \
  --mini_batch_size 32 \
  --ppo_epochs 1 \
  --value.ws_url ws://127.0.0.1:8877 \
  --explore_mode conditional_keyframe \
  --explore_keyframe_threshold 0.5 \
  --explore_perturb_backend dct_sigma_network \
  --explore_network_ckpt checkpoints/explored_net/dimwise_sigma_dct.pkl \
  --explore_action_noise_dims "0,1,2,3,4,5,6"
```

这条链路现在是：

- 先标准 full-chunk decode
- 再做 DCT 子块 sigma 扰动
- 最后对执行 token 重新算 logprob
- phase / value 都统一走 ws value server
- keyframe exploration gate 也统一走 ws value server
- 整个 chunk 的执行后 value 预测固定使用 chunk 起点 phase
