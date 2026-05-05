# PPO Dimwise Sigma Simplification

## Goal

Simplify the current online PPO exploration path while preserving:

- keyframe-gated exploration
- stochastic exploration via per-location sigma
- low-frequency DCT perturbation
- optional action-dimension masking similar to `explore_action_noise_dims`

The main structural change is:

- remove rowwise decode / rowwise replay from PPO rollout
- keep standard full-chunk decode as the only decode path
- apply exploration in DCT space after the full chunk is decoded


## Current Problem

The current `rowwise_sigma` rollout path couples three things that do not need to be coupled:

1. rowwise / transposed FAST tokenization
2. per-action-dimension sigma prediction
3. online PPO exploration rollout

This makes the online rollout path structurally complex and slow:

- decode proceeds row-by-row instead of chunk-by-chunk
- each row is generated token-by-token outside the main JAX decode loop
- explored rows are replayed back through the model
- PPO rollout behavior becomes dependent on a special rowwise path that differs from the normal fast decode path

The main design goal is to remove this coupling.


## High-Level Design

### Decode Path

Use a single decode path for PPO rollout:

1. standard full-chunk FAST decode
2. extract full DCT coefficients from decoded tokens
3. if keyframe gate is inactive: use decoded chunk directly
4. if keyframe gate is active: apply DCT-space stochastic perturbation
5. encode perturbed DCT coefficients back to action tokens
6. recompute logprobs on the executed tokens

This keeps PPO aligned with the existing fast JAX decode path and removes rowwise token generation from rollout.


## Exploration Subspace

Let the decoded DCT coefficients be:

- `C in R^(T x A)`
- for current policy, typically `T=32`, `A=14`

Exploration is restricted to a configurable low-frequency action subspace:

- `F`: selected DCT frequency indices
- `D`: selected action-dimension indices

Default intended form:

- `F = {0, 1, 2, 3}` with size `k=4`
- `D` comes from configuration, similar to `explore_action_noise_dims`

Only the submatrix `C[F, D]` is perturbed.
All other coefficients remain unchanged.


## Sigma Network

### Input

The exploration network input is the selected DCT submatrix:

- `X = C[F, D]`

Implementation uses flattening:

- `x = flatten(X)`
- shape is `k * m`
- where `k = |F|`, `m = |D|`

Example:

- if `k=4` and `m=7`, then input dimension is `28`

### Output

The network outputs sigma values for the same selected subspace:

- `S = f_theta(x)`
- reshape to `sigma in R^(k x m)`

Important:

- output is `sigma` (standard deviation)
- not variance

Suggested parameterization:

- network predicts unconstrained values
- apply `softplus` and add `sigma_min`
- optionally clip to `sigma_max`


## Stochastic Perturbation

Sample:

- `eps ~ N(0, I)` with shape `(k, m)`

Compute low-frequency perturbation:

- `delta_sub = sigma * eps`

Construct full perturbation matrix:

- `delta_C in R^(T x A)`
- initialize all zeros
- assign `delta_C[F, D] = delta_sub`

Executed DCT coefficients:

- `C_exec = C + delta_C`

Optional action-space clipping can still be applied by:

1. converting `delta_C` to action-space delta
2. clipping action delta
3. projecting back to DCT space

This preserves the existing action-magnitude safety behavior if desired.


## PPO Rollout Flow

### Non-Exploration Branch

When keyframe gate is inactive:

1. decode standard chunk tokens
2. decode full DCT coefficients
3. use decoded chunk directly
4. no perturbation

### Exploration Branch

When keyframe gate is active:

1. decode standard chunk tokens
2. decode full DCT coefficients `C`
3. extract subspace `C[F, D]`
4. flatten and run sigma network
5. sample `eps`
6. construct `delta_C`
7. compute `C_exec = C + delta_C`
8. encode `C_exec` back to action tokens
9. recompute token logprobs using executed tokens
10. execute decoded actions from `C_exec`

This preserves stochastic exploration while keeping rollout on the standard decode path.


## Training Data for Sigma Network

Training samples are chunk-level, not rowwise token-level.

For each training chunk:

1. compute action chunk `A_chunk in R^(T x A)`
2. transform to DCT coefficients `C`
3. select subspace `C[F, D]`
4. flatten into input vector `x`

The network predicts sigma over the same flattened subspace.

Training objective remains stochastic and can keep the current spirit:

- encourage useful perturbation magnitude
- penalize action-space violations
- only perturb the configured low-frequency selected action dimensions

Compared with the original dimwise sigma training, the new sample unit is:

- one sample per chunk
- instead of one sample per `(chunk, action_dim)`


## Relationship to Current Dimwise Sigma Design

What is preserved:

- stochastic exploration using `sigma * eps`
- low-frequency-only perturbation
- configurable action dimensions
- DCT-space perturbation

What changes:

- network no longer operates on one action dimension at a time
- input is a selected DCT block rather than one DCT column plus onehot dim id
- rollout no longer depends on rowwise decode / replay

This is intended to keep the exploration semantics while simplifying the rollout architecture.


## SFT and Tokenizer Boundary

This PPO simplification should be decoupled from SFT tokenization choices.

Planned direction:

- PPO exploration should not require rowwise decode logic
- PPO exploration should not require transposed / rowwise serialization
- SFT tokenizer choices should be evaluated independently

Practical preference:

- if transpose / rowwise tokenization was introduced only to support the old rowwise PPO exploration path, revert SFT/tokenizer behavior toward the original implementation where possible

This keeps the base policy path closer to the original setup and avoids carrying PPO-specific complexity into SFT.


## Configuration Surface

Suggested exploration configuration parameters:

- `explore_dct_noise_k`
  - number of low-frequency rows to perturb
  - intended default: `4`

- `explore_action_noise_dims`
  - selected action dimensions to perturb
  - same role as current action-dimension filtering

- `sigma_min`
  - minimum sigma after parameterization

- `sigma_max`
  - optional sigma cap

- `action_delta_limit`
  - optional action-space clipping limit after DCT perturbation


## Benefits

- removes rowwise decode and row replay from PPO rollout
- keeps PPO on the standard full-chunk decode path
- preserves stochastic exploration
- preserves low-frequency perturbation structure
- preserves configurable action-dimension masking
- better matches chunk-level exploration semantics
- cleanly separates PPO exploration design from SFT tokenization design


## Open Questions

1. Whether sigma network input should contain only flattened `C[F, D]`, or also include small extra context such as keyframe probability.
2. Whether action-space clipping should remain exactly as in the current sigma-net implementation.
3. Whether the final sigma net should be implemented in NumPy first for simplicity, then moved into JAX if rollout integration needs it.
4. Whether SFT tokenizer config should be reverted immediately or only after PPO rollout refactor is complete.


## Recommended Implementation Order

1. Remove rowwise PPO rollout path from exploration usage.
2. Keep standard full-chunk decode and executed-token logprob recomputation.
3. Implement the new sigma net interface on flattened `C[F, D]`.
4. Update sigma-net training script to use chunk-level subspace samples.
5. Validate rollout speedup and exploration behavior.
6. Separately evaluate whether SFT/tokenizer config should return to the original non-transposed setup.
