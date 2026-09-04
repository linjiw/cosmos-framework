# Cosmos3-Edge humanoid navigation research note

## Outcome

The unadapted Cosmos3-Edge INT8WO generator is a useful visual prior, but this
pilot does not support using it as a navigation or collision-avoidance policy.
Across three prompts and three deterministic seeds, none of the nine generated
videos completed the full ordered behavior requested by its prompt. This is an
open-loop prompt-following result, not a closed-loop collision or success rate.

The recommended system boundary is therefore hierarchical: adapt Cosmos3-Edge
to predict short-horizon body-frame navigation commands, retain a separately
validated whole-body locomotion controller, and place a geometric safety filter
between them. Do not ask a diffusion video generator to directly close a
high-rate humanoid torque loop.

## Prompt pilot

### Contract

| Item | Setting |
| --- | --- |
| Checkpoint | `Cosmos3-Edge` |
| Inference path | streamed INT8WO, VAE CPU offload, `torch.compile`, no guardrails |
| Samples | 3 scenarios × seeds 17, 18, 19 = 9 videos |
| Video | 256p bucket (`320×192` at 16:9), 25 frames, 10 FPS, 2.5 seconds |
| Sampling | 35 UniPC steps, guidance 6.0, shift 10.0 |
| Hardware | NVIDIA GeForce RTX 5080, 16,303 MiB |
| Maximum process VRAM observed | 8,214 MiB, sampled every 100 ms |
| Steady denoising time | approximately 8.0–8.8 seconds per clip after compilation |

The exact structured prompts are in
[`inputs/experiments/humanoid_navigation_t2v.jsonl`](../inputs/experiments/humanoid_navigation_t2v.jsonl).
Reproduce the three-seed set with:

```shell
COSMOS_VAE_CPU_OFFLOAD=1 LD_LIBRARY_PATH='' \
python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i inputs/experiments/humanoid_navigation_t2v.jsonl \
  -o outputs/humanoid_navigation_edge_int8wo \
  --checkpoint-path=Cosmos3-Edge \
  --quantization-method=int8wo \
  --use-torch-compile --no-guardrails \
  --num-outputs=3 --seed=17
```

### Scoring rule

A clip counts as a full prompt completion only when (1) the requested agents and
obstacles are present, (2) the ordered action phases are visible, and (3) the
final requested state is visible. A manually inspected 13-frame temporal strip
was used for each clip. Appearance alone, a stationary agent, or omission of the
obstacle does not count as avoidance.

### Observations

| Scenario | Seed 17 | Seed 18 | Seed 19 | Full completion |
| --- | --- | --- | --- | --- |
| Static corridor slalom | Figure moves in a doorway; the box and slalom are absent. | A human-like figure stands beside a slab/box; the corridor and avoidance maneuver are absent. | Corridor and box appear, but no clear left pass and recenter sequence is completed. | 0/3 |
| Yield to crossing pedestrian | Robot and pedestrian appear; the robot remains stationary and does not show approach–stop–resume. | Two human figures appear instead of a robot–pedestrian pair; no yield sequence. | A pedestrian passes a stationary humanoid-like figure; no approach or resume phase. | 0/3 |
| Step over low barrier | Robot and barrier appear; both feet do not visibly clear the barrier. | Robot approaches/remains behind the barrier; no complete step-over. | Robot appears on top of the barrier rather than clearing it. | 0/3 |
| **Strict total** | 0/3 | 0/3 | 0/3 | **0/9** |

The 13-frame inspection strips for all nine clips are archived in
[`docs/assets/humanoid-navigation/`](./assets/humanoid-navigation/).

All nine inference jobs completed and produced valid videos. The failure is at
the requested spatial-temporal behavior, not at model execution. Several clips
render useful ingredients—a recognizable robot, a corridor, a pedestrian, or a
barrier—but do not compose them into the commanded sequence.

### What this result establishes

- The optimized INT8WO path can generate short humanoid-themed T2V clips inside
  the measured single-process memory envelope.
- The base model has partial scene and object priors relevant to navigation.
- Under this prompt contract, exact multi-stage behavior was not observed in
  nine open-loop samples.

It does **not** establish a robot collision rate, navigation success rate,
balance margin, controller latency, or sim-to-real transfer. Those require
action-conditioned closed-loop trials with simulator contact ground truth.

## Recommended system

```text
RGB/depth or lidar + body-frame goal + proprioception
                         │
                         ▼
       Cosmos3-Edge navigation policy / world model
       short-horizon (vx, vy, yaw-rate, step-height) chunk
                         │
                         ▼
        geometric command filter + emergency stop
                         │
                         ▼
      pretrained whole-body controller (50–200 Hz class)
                         │
                         ▼
                humanoid joint commands
```

The rate labels are design targets and must be profiled on the deployment
computer. A lower-dimensional command interface isolates semantic navigation
from contact-rich balance control and makes a classical safety layer possible.
The [HOVER](https://github.com/NVlabs/HOVER/) project is one primary-source
example of a reusable neural whole-body controller trained from motion data.

## Data needed

### 1. Motion and stabilization prior

Use motion-capture data such as [AMASS](https://amass.is.tue.mpg.de/) to train or
select the lower-body controller. AMASS supplies varied human motion, but it
does not supply robot-specific collision-avoidance decisions, actuator limits,
or onboard sensor observations; it is a locomotion prior, not the navigation
dataset.

### 2. Simulation navigation episodes

Collect synchronized episodes in Isaac Lab or MuJoCo with:

- egocentric RGB plus depth or local range observations;
- body-frame goal direction and distance;
- joint position/velocity, base velocity, projected gravity, and foot contact;
- expert short-horizon commands from a strengthened A*/MPC/local-planner teacher;
- executed joint commands from the fixed whole-body controller;
- obstacle and human trajectories, minimum clearance, contact force, fall,
  timeout, goal, intervention, and route-length labels;
- successful, near-contact, recovery, dead-end, and failure episodes—not only
  clean demonstrations.

A practical first data budget is a learning curve over 10k, 50k, and 100k
episodes spanning at least 1,000 randomized layouts. These are proposed study
points, not established sample-complexity requirements. Hold out entire layouts,
obstacle-motion profiles, and sensor corruptions; never split overlapping frames
from one trajectory across train and test.

### 3. Human interaction priors

[SCAND](https://arxiv.org/abs/2203.15041) provides 8.7 hours of socially
compliant navigation demonstrations with lidar, vision, odometry, inertial data,
and joystick commands, while
[THÖR](https://arxiv.org/abs/1909.04403) provides accurate human trajectories,
goals, gaze, obstacles, lidar, and robot motion. Both are useful for pedestrian
prediction and social-route priors; neither matches humanoid actuation, so do
not train the final joint policy from their actions.

### 4. Target-platform corrective data

After simulation screening, collect a staged 1/5/10/20-hour learning curve of
target-humanoid teleoperation and corrective interventions. Synchronize sensor
timestamps, commands, state, motor actions, safety-filter interventions, and
contact events. Begin without nearby people, then use soft obstacles or dummies,
and involve qualified oversight before supervised human-proximity trials.

## Learning recipe

1. **Train in BF16, quantize after export.** The existing Edge vision recipe is
   BF16 generator SFT. INT8WO is an inference transformation in this repository,
   not a validated quantization-aware training recipe.
2. **Domain-adapt the visual dynamics.** Start from `vision_sft_edge` and train
   T2V/I2V/V2V on egocentric humanoid clips. Keep the current 70/20/10 modality
   mix initially, then ablate it rather than assuming it is optimal.
3. **Add a humanoid action domain.** Register a new embodiment/domain ID, action
   width, normalizer, LeRobot-compatible dataset adapter, and Edge action-policy
   experiment. The model supports action tensors up to 64 channels, but a
   3–4-dimensional high-level command is the safer first interface.
4. **Behavior-clone the expert.** Train `mode="policy"` first on observation,
   proprioception, language/goal, and command chunks. Compare against an
   identical backbone with vision-only SFT and against a non-Cosmos behavior
   cloning policy under the same data and compute contract.
5. **Add predictive auxiliaries only after the policy baseline is stable.** Joint
   policy/forward-dynamics training can test whether predicted futures reduce
   collision and dead-end failures; the ablation must keep every other setting
   fixed. [Cosmos Policy](https://arxiv.org/abs/2601.16163) provides primary-source
   precedent for adapting a video model with robot demonstrations and predicting
   actions, future images, and values.
6. **Correct covariate shift in simulation.** Use DAgger-style expert relabeling
   or PPO/fine-tuning on policy rollouts, emphasizing near-collision recoveries,
   occluded goals, pushes, latency, friction, mass, lighting, and sensor dropout.
7. **Export, apply INT8WO, and repeat every gate.** Quantization promotion is a
   separate factor; compare BF16 and INT8WO with identical checkpoints, seeds,
   scenes, and policy timing.

## Evaluation plan

### Experiment A — visual prior (current tier)

- **Question:** Can the generator compose navigation objects and ordered actions?
- **Changed variable:** base versus navigation-video SFT; same prompts and seeds.
- **Outcomes:** agent/obstacle presence, ordered-event completion, visible
  interpenetration, temporal identity stability.
- **Evidence:** open-loop generated video only; no safety claim.

### Experiment B — closed-loop simulation

- **Question:** Does the adapted policy improve goal reaching without increasing
  collision or falls?
- **Comparators:** tuned classical planner + same whole-body controller;
  vision behavior cloning + same controller; Cosmos policy SFT; Cosmos policy
  plus forward-dynamics/value auxiliary; each with and without the same safety filter.
- **Cells:** static clutter, narrow passages, dynamic crossings, temporary goal
  occlusion, and low-obstacle/terrain transitions at three difficulty levels.
- **Replication:** at least three training seeds and 20 closed-loop episodes per
  method × condition × held-out environment family for the first directional
  comparison. Report dispersion over seeds and environments, not pooled frames.
- **Primary metrics:** goal success and any non-foot environmental contact.
- **Secondary metrics:** fall, timeout, stuck, intervention, minimum clearance,
  SPL/path efficiency, time among successful trials, command jerk, and inference
  deadline misses.
- **Failure partition:** success / collision / fall / stuck / timeout / emergency
  intervention, mutually exclusive with the full attempted denominator shown.

Isaac Lab exposes body-filtered contact reporting and contact-force histories;
use those physics signals rather than judging collision from pixels. See the
[official contact-sensor API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.sensors.html).

### Experiment C — staged physical evaluation

Run hardware-in-the-loop timing first, then soft-obstacle trials, then one venue
without people, and only then supervised pedestrian interaction. A physical
pilot demonstrates feasibility; a comparative claim needs multiple held-out
venues and a disclosed intervention protocol.

## Decision gates

Promote the approach only when:

1. policy inference plus filtering meets the command deadline on the onboard
   computer;
2. its closed-loop held-out success improvement survives three training seeds;
3. collision, fall, and intervention rates are reported separately and do not
   regress against the strengthened baseline;
4. the gain survives removal of privileged simulator signals;
5. INT8WO matches the BF16 policy within a predeclared tolerance on every primary
   metric.

## Provisional contribution statement

If the planned evidence supports it, the work would contribute:

1. a hierarchical Cosmos3-Edge humanoid navigation policy that predicts a
   four-dimensional body-frame command chunk while retaining a fixed whole-body
   controller and geometric safety filter;
2. a synchronized humanoid navigation dataset spanning at least 1,000 simulated
   layouts plus staged target-platform corrective trajectories, with exhaustive
   collision, fall, timeout, and intervention labels; and
3. a matched closed-loop study over three training seeds and five navigation
   condition families that isolates visual SFT, action SFT, predictive auxiliary
   training, the safety filter, and INT8 inference.

These are proposed contributions, not completed findings.
