# EE Delta 14D Design For RoboTwin pi05

## Goal

This document summarizes the agreed design direction for migrating the current RoboTwin `joint/qpos` pipeline to an EE-centered pipeline while preserving the main advantages of the current pi05 stack:

- keep the action dimension at 14
- keep PPO / Pi0-FAST / exploration framework structure mostly unchanged
- make exploration semantics happen in end-effector space rather than raw joint space
- avoid directly training on absolute quaternions as actions

This is a design note only. It is intentionally written before implementation.


## Current Situation

Today the relevant pipeline pieces are centered around joint-space data:

- `/mnt/data/lhm/vla-rl/RoboTwin/collect_data.sh`
- `RoboTwin/script/collect_data.py`
- `RoboTwin/policy/pi05/scripts/process_data.py`
- `RoboTwin/policy/pi05/finetune.sh`
- `RoboTwin/policy/pi05/eval.sh`
- `RoboTwin/policy/pi05/scripts/serve_robotwin_env_ws.py`
- `RoboTwin/policy/pi05/scripts/train_pi0_fast_online_v2.py`

The current online and offline flow mostly assumes:

- `observation.state` is joint-based state
- `action` is joint-based target action
- some training transforms assume a joint-style delta/absolute relationship

RoboTwin itself already exposes EE pose information and also supports EE execution through planning:

- current EE pose can be read from `envs/robot/robot.py`
- env execution already supports `take_action(..., action_type='ee')`

Important pose detail:

- single-arm EE pose is currently represented as `xyz + quaternion`
- that means pose is 7D per arm
- if gripper is appended, single-arm EE execution is 8D
- dual-arm EE execution would be 16D if we used absolute EE targets directly

We do **not** want that as the training action representation for the first version.


## Chosen Direction

We will use an **EE-centered, delta-style 14D representation**.

The high-level idea is:

- observation uses EE state, but in a compact 14D form
- action uses EE delta action in 14D form
- env execution still reconstructs an absolute EE target internally and executes in EE mode
- the model still outputs 14D actions, so most of the existing policy code can remain structurally similar


## Final Representation

### Observation: 14D

Per arm:

1. `x`
2. `y`
3. `z`
4. `r_ref_x`
5. `r_ref_y`
6. `r_ref_z`
7. `grip`

Dual arm:

- left arm 7D
- right arm 7D
- total 14D

Interpretation:

- `xyz` is the current absolute EE position
- `r_ref` is a 3D rotation vector describing the current EE orientation relative to a fixed reference orientation
- `grip` is current gripper state

Recommended reference orientation:

- use each arm's reset / home orientation as the fixed reference
- keep that convention global and stable across episodes

Why this observation design:

- keeps obs at 14D
- avoids directly feeding quaternion into the model
- preserves absolute position information
- keeps orientation in a compact continuous 3D form


### Action: 14D

Per arm:

1. `dx`
2. `dy`
3. `dz`
4. `dr_x`
5. `dr_y`
6. `dr_z`
7. `grip_target`

Dual arm:

- left arm 7D
- right arm 7D
- total 14D

Interpretation:

- `dxyz` is EE translational delta
- `dr` is a 3D rotation vector representing the relative rotation from current orientation to target orientation
- `grip_target` is the target gripper value

Why this action design:

- preserves EE semantics
- keeps action dimension at 14
- avoids training directly on absolute quaternion actions
- makes exploration in EE translation/rotation dimensions straightforward


## Why Not Absolute Quaternion Actions

Absolute quaternion EE actions would make dual-arm action naturally become 16D:

- left pose: `xyz + quat` = 7
- left gripper: 1
- right pose: `xyz + quat` = 7
- right gripper: 1
- total: 16

They are also awkward for:

- delta transforms
- direct perturbation
- normalization intuition

This is why we choose:

- pose observation compressed into `xyz + ref-rotvec + grip`
- action represented as `delta xyz + delta rotvec + grip`


## Precise Geometry Definitions

### Observation Orientation

For arm orientation at time `t`:

- let current orientation be `R_t`
- let fixed reference orientation be `R_ref`
- define observation orientation vector as:

`r_obs = log(R_ref^-1 * R_t)`

where `log(...)` denotes the SO(3) logarithm map, i.e. a 3D rotation vector.

This gives a compact 3D orientation state that is stable as long as the arm stays within a normal operating regime.


### Action Orientation

For action from time `t` to target:

- let current orientation be `R_t`
- let target orientation be `R_target`
- define action rotation vector as:

`r_act = log(R_t^-1 * R_target)`

This is a local relative orientation action.


### Position Action

For action from time `t`:

- let current position be `p_t`
- let target position be `p_target`
- define translational action as:

`dp = p_target - p_t`


### Gripper

For the first version:

- keep gripper action as an absolute target value
- do not convert gripper to delta

This is simpler and better aligned with current usage.


## Reconstruction At Execution Time

The policy outputs 14D delta-EE action.

The env server must reconstruct a target EE pose before execution.

Per arm:

- current pose: `(p_t, R_t)`
- action gives `(dp, dr, g)`
- reconstruct:
  - `p_target = p_t + dp`
  - `R_target = R_t * Exp(dr)`
  - `g_target = g`

Then convert `R_target` back to quaternion and form the EE execution target:

- `x y z qx qy qz qw grip`

Then call env execution in EE mode.

Important note:

- model action is 14D
- env execution target remains effectively 16D internally
- this conversion happens inside the env adapter / websocket server


## Why This Keeps The Current Advantages

This design keeps the main benefits of the current pipeline:

- action dimension remains 14, so many model assumptions stay stable
- exploration can be targeted at EE dimensions directly
- PPO and Pi0-FAST remain continuous-vector based at the action interface
- no need to rewrite the entire policy/trainer stack around a 16D quaternion action

It also adds important benefits:

- more meaningful exploration geometry
- less joint-specific bias
- easier reasoning about which action dimensions should be perturbed


## Training Transform Implications

The current joint-style delta transform logic should **not** be reused as-is for this new representation.

Reason:

- our action is already explicitly defined as a delta-EE action
- reapplying the current joint-style `DeltaActions/AbsoluteActions` logic would be conceptually wrong
- that old logic assumes a direct elementwise relation between state and action dimensions in joint space

For the EE delta design:

- the dataset itself should already store the desired 14D action representation
- training should consume it directly
- the current joint-specific delta transform should be disabled for the new EE config

This does **not** mean we stop using delta actions.

It means:

- delta is part of the data definition
- not part of a later transform layer


## Offline Dataset Design

### Data To Save

During data generation, it is recommended to save both:

- raw EE information for debugging and verification
- final 14D obs/action fields actually used by training

Recommended contents:

- `observation.state`: final 14D EE observation
- `action`: final 14D delta-EE action

Recommended debug fields:

- raw left EE pose as `xyz + quat`
- raw right EE pose as `xyz + quat`
- raw joint state
- optionally raw joint action


### For `/mnt/data/lhm/vla-rl/RoboTwin/collect_data.sh`

This script launches the environment-side collection process.

For the EE migration, the data source must eventually provide:

- current EE pose per arm
- current gripper state per arm
- enough adjacent-frame information to derive delta-EE actions

Since RoboTwin already exposes `endpose` in environment observation generation, the likely implementation strategy is:

- keep collecting the underlying raw trajectory
- extend the saved raw observation to include end-effector pose fields if not already included in final exported artifacts


### For `RoboTwin/policy/pi05/scripts/process_data.py`

This script is one of the key offline converters.

Today it reads joint-based HDF5 data and writes processed training episodes where:

- `qpos` is joint-based
- `action` is joint-based

For the EE delta design, this script will need to become the main place where the new representation is built.

Target change:

- read raw EE pose and gripper information from collected data
- compute 14D observation
- compute 14D delta-EE action
- write processed episodes using those fields instead of joint-based ones

This script is likely the best location for:

- quaternion to relative-rotation-vector conversion
- fixed-reference orientation encoding for observation
- adjacent-step delta construction for action


## Online Env Design

### For `scripts/serve_robotwin_env_ws.py`

This file is the central online environment adapter.

It currently:

- returns observation to policy
- accepts action chunk from policy
- executes env steps
- optionally writes LeRobot-format online data

For EE delta mode, it should eventually support:

- returning `observation.state` in 14D EE form
- receiving 14D delta-EE actions from policy
- reconstructing absolute EE targets internally
- calling `take_action(..., action_type='ee')`
- exporting online collected datasets with EE obs/action semantics

Recommended future switch:

- add an explicit action/state mode option, for example `joint` vs `ee_delta`

In `ee_delta` mode:

- `reset()` returns EE observation
- `step()` consumes delta-EE action
- action is internally lifted to absolute EE target
- env executes in EE mode
- LeRobot writing stores EE observation/action fields


### For `scripts/train_pi0_fast_online_v2.py`

This file is the main online PPO entry.

Conceptually, it should not require a full rewrite.

What changes:

- the observation coming from env becomes 14D EE state
- the policy action becomes 14D delta-EE action
- rollout samples and replay buffer now contain EE semantics rather than joint semantics

What should remain mostly unchanged:

- PPO loop
- chunk collection structure
- reward/value plumbing
- exploration structure

Important follow-up:

- current exploration masks / perturb rules should be reinterpreted over EE dimensions
- for example, exploration over the first three dims of each arm would now directly mean EE translation perturbation


## Offline Training And Evaluation

### For `/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/finetune.sh`

This script launches offline training.

It probably does not need structural changes beyond:

- pointing to a new EE-specific train config
- using a dataset generated in the new EE format
- using newly computed norm stats


### For `/mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/eval.sh`

This script launches evaluation.

Its main dependency is the deployed policy/config stack.

For EE delta migration, evaluation must ultimately run with:

- a policy checkpoint trained on EE obs/action
- a deploy / eval path that reconstructs EE execution targets instead of assuming joint targets

The shell wrapper itself is unlikely to be the main complexity.


## Value / Keyframe / Exploration Heads

After the EE migration:

- policy must be retrained
- value head must be retrained
- keyframe head must be retrained
- exploration network must be retrained

The reason is not necessarily that algorithm code must change a lot.

It is that:

- `observation.state` distribution changes
- `action` semantics change
- normalization changes

As long as those modules consume the new fields consistently, many parts of their framework code may stay structurally similar.


## Exploration Implications

This design is especially helpful for exploration.

In the current joint design:

- equal-magnitude perturbation on different joints can cause very unequal end-effector motion

In the new EE delta design:

- the first three dims of each arm directly represent translational motion
- the next three dims directly represent rotational increment

This enables simple and meaningful exploration rules such as:

- perturb only EE translation dims
- perturb only early rollout chunks in EE translation space
- use different scales for translation and rotation perturbation


## Recommended Phase Order

Implementation should not attempt to switch the entire stack at once.

Recommended order:

1. Define final 14D obs/action specification in code comments and docs
2. Modify offline data conversion so a processed EE dataset can be generated
3. Compute norm stats for EE dataset
4. Add a dedicated EE training config with joint-style delta transform disabled
5. Run offline training smoke test
6. Update env websocket server to support EE delta mode
7. Run online rollout-only smoke test
8. Retrain value / keyframe / exploration models on EE data
9. Run full online PPO


## Main Risks

### 1. Rotation Convention Errors

This is the most important correctness risk.

Be consistent about:

- quaternion component order
- quaternion multiplication order
- matrix vs quaternion convention
- rotvec logarithm / exponential implementation


### 2. Reference Orientation Consistency

Observation orientation uses a fixed reference orientation.

That reference must be:

- well defined
- stable across data generation, training, and inference


### 3. EE Planner Runtime

EE execution will use planner-based execution.

Compared with joint action execution, this may:

- be slower
- fail on some targets

So online PPO speed may change.


### 4. Large Rotation Deltas

This design assumes neighboring targets do not require huge relative orientation changes.

If large orientation jumps occur frequently, rotvec smoothness may degrade.


## Summary

The agreed first-version design is:

- observation: dual-arm 14D EE state
  - per arm: `xyz + ref-rotvec + grip`
- action: dual-arm 14D delta-EE action
  - per arm: `dxyz + drotvec + grip_target`
- env execution:
  - reconstruct absolute EE target pose internally
  - execute in `action_type='ee'` mode
- training:
  - use EE dataset directly
  - disable current joint-style delta transform for the new config
- all downstream heads:
  - retrain on EE data

This design is the current recommended baseline because it:

- keeps `action_dim=14`
- avoids absolute quaternion action regression
- preserves EE semantics for exploration
- minimizes unnecessary architectural disruption to the current pi05 stack
