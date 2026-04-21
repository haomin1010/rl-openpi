# RoboTwin `beat_block_hammer` 数据处理 + `pi05` / `pi0_fast`（full）训练评估指南

这份文档统一使用：

- 仓库路径：`/mnt/data/lhm/vla-rl`
- 训练配置：`full`（不使用 LoRA）
- 评估入口：`RoboTwin/policy/pi05/eval.sh`

对应配置定义在：

- `RoboTwin/policy/pi05/src/openpi/training/config.py`
- `pi05_aloha_full_base`
- `pi0_fast_aloha_robotwin_full`

## 1. 准备数据软链接

如果你的原始数据在 `/mnt/data/lhm/test/RoboTwin/data/beat_block_hammer`，先软链接到仓库内：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin
ln -sfn /mnt/data/lhm/test/RoboTwin/data/beat_block_hammer data/beat_block_hammer
```

## 2. 数据预处理

将 RoboTwin 原始数据转换为 `processed_data/beat_block_hammer-demo_clean-100`：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
python scripts/process_data.py beat_block_hammer demo_clean 100
```

处理完成后目录应为：

```bash
/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/processed_data/beat_block_hammer-demo_clean-100
```

## 3. 转成训练需要的 LeRobot 数据集

下面示例沿用配置里的默认 `repo_id=your_repo_id`：

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/beat_block_hammer-demo_clean-100 \
  --repo-id your_repo_id \
  --task beat_block_hammer \
  --mode image
```

## 4. 计算归一化统计量（full 配置）

训练前必须先计算，且配置名要和训练时完全一致。

### 4.1 `pi05` full

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base
```

### 4.2 `pi0_fast` full

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_full
```

## 5. 训练 `pi05`（full）

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash finetune.sh pi05_aloha_full_base beat_block_hammer_pi05_full 0,1,2,3
```

训练输出目录：

```bash
/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/beat_block_hammer_pi05_full
```

默认最终 checkpoint 步数：

```bash
20000
```

## 6. 训练 `pi0_fast`（full）

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash finetune.sh pi0_fast_aloha_robotwin_full beat_block_hammer_pi0fast_full 4,5,6,7
```

训练输出目录：

```bash
/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/checkpoints/pi0_fast_aloha_robotwin_full/beat_block_hammer_pi0fast_full
```

默认最终 checkpoint 步数：

```bash
30000
```

## 7. 评估 `pi05`（full）

### 7.1 在 `demo_clean` 上评估

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_clean pi05_aloha_full_base beat_block_hammer_pi05_full 0 0
```

### 7.2 在 `demo_randomized` 上评估

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_randomized pi05_aloha_full_base beat_block_hammer_pi05_full 0 0
```

## 8. 评估 `pi0_fast`（full）

### 8.1 在 `demo_clean` 上评估

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full beat_block_hammer_pi0fast_full 0 0
```

### 8.2 在 `demo_randomized` 上评估

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
bash eval.sh beat_block_hammer demo_randomized pi0_fast_aloha_robotwin_full beat_block_hammer_pi0fast_full 0 0
```

## 9. 结果位置

### 9.1 训练 checkpoint

```bash
RoboTwin/policy/pi05/checkpoints/<train_config_name>/<model_name>/
```

### 9.2 评估结果

```bash
RoboTwin/eval_result/beat_block_hammer/pi05/<task_config>/<model_name>/<timestamp>/
```

评估文本通常在：

```bash
RoboTwin/eval_result/.../_result.txt
```

## 10. 一套最常用的完整命令顺序

### 10.1 `pi05` full

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin
ln -sfn /mnt/data/lhm/test/RoboTwin/data/beat_block_hammer data/beat_block_hammer

cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
python scripts/process_data.py beat_block_hammer demo_clean 100

uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/beat_block_hammer-demo_clean-100 \
  --repo-id your_repo_id \
  --task beat_block_hammer \
  --mode image

uv run scripts/compute_norm_stats.py pi05_aloha_full_base
bash finetune.sh pi05_aloha_full_base beat_block_hammer_pi05_full 0
bash eval.sh beat_block_hammer demo_clean pi05_aloha_full_base beat_block_hammer_pi05_full 0 0
```

### 10.2 `pi0_fast` full

```bash
cd /mnt/data/lhm/vla-rl/RoboTwin
ln -sfn /mnt/data/lhm/test/RoboTwin/data/beat_block_hammer data/beat_block_hammer

cd /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05
python scripts/process_data.py beat_block_hammer demo_clean 100

uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir processed_data/beat_block_hammer-demo_clean-100 \
  --repo-id your_repo_id \
  --task beat_block_hammer \
  --mode image

uv run scripts/compute_norm_stats.py pi0_fast_aloha_robotwin_full
bash finetune.sh pi0_fast_aloha_robotwin_full beat_block_hammer_pi0fast_full 0
bash eval.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full beat_block_hammer_pi0fast_full 0 0
```

## 11. 补充说明

- `compute_norm_stats` 的配置名必须与 `finetune.sh` 训练配置名一致。
- `eval.sh` 里的 `model_name` 必须与训练时 `finetune.sh` 的第二个参数一致。
- 如果你改了 `repo_id`，要同步修改 `config.py` 中对应训练配置的 `repo_id`，或在转换数据时继续使用 `your_repo_id`。

## 12. websokcet eval

- CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py --port 8000 policy:checkpoint   --policy.config=pi0_fast_aloha_robotwin_full   --policy.dir=checkpoints/pi0_fast_aloha_robotwin_full/beat_block_hammer_pi0fast_full/30000/

- bash eval_ws.sh beat_block_hammer demo_clean pi0_fast_aloha_robotwin_full beat_block_hammer_pi0fast_full 0 4 127.0.0.1 8000

## 13. ppo训练
### 1. 收集数据
-- uv run python scripts/serve_robotwin_env_ws.py   --task_name beat_block_hammer   --task_config demo_clean   --instruction_type unseen   --port 8765   --save_lerobot   --lerobot_root /mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_online_run1   --lerobot_repo_id lerobot-hammer-online   --lerobot_fps 50   --lerobot_overwrite

-- uv run python scripts/train_pi0_fast_online_v2.py   --policy.path checkpoints/pi0_fast_aloha_robotwin_full/beat_block_hammer_pi0fast_full/10000/   --policy.config pi0_fast_aloha_robotwin_ppo   --env.ws_url ws://127.0.0.1:8765   --total_updates 1   --rollout_batch_size 1024   --mini_batch_size 32   --ppo_epochs 0   --value_epochs 0
