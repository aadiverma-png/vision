# Plan A: ORCA features that make Visison more efficient

## Context

The user wants ORCA (Optimal Reciprocal Collision Avoidance, van den Berg et al. 2011) in Visison and asked which of its features can make the algorithm *more efficient*. Today Visison has:
- **tactical planning:** reservation-based bypass planning (`planner.py`, `controller.py`);
- **recovery:** emergency braking plus standoff resolution (`emergency.py`, `standoff.py`).

The inefficiencies we have seen:
1. **One robot takes the whole cost of a conflict.** R1 waits 12 s in `near`, while R2 never changes speed even though a small slowdown would let R1 go at once.
2. **Nothing sits between "fine" and "emergency brake".** A small timing error goes straight to a full stop of every robot involved. The WARN level is only logged.
3. **Rear-end conflicts:** followers catching up with a slower robot trigger emergency brakes (the `R8->R9` brakes seen in `fleet_10`), then a stop-and-go queue release.
4. **The coordinator does O(n²) pair loops** (`identify`, `_cascade`, `_detect_deadlocks`). That's fine for 10 robots but not for a large fleet.

ORCA offers four ideas that fix exactly these, plus the local filter itself:

| ORCA idea | Visison feature | fixes |
|---|---|---|
| **Reciprocity**: share the avoidance effort | **A. Reciprocal bypass windows**: oncoming robots slow a little so the bypasser doesn't wait long | 1 |
| **Minimal velocity change `u`** (the ORCA half-plane) | **B. Graded response**: ORCA soft-avoids first, and the emergency brake fires only if `u` can't be achieved | 2 |
| **ORCA in 1-D** (lateral locked by the lane) | **C. Cooperative speed keeping in lanes**: followers keep a safe gap on their own | 3 |
| **ORCA-DD** (Snape et al. 2010): ORCA on a differential-drive robot's *effective centre*, with its kinematic limits as constraints | **D. ORCA-DD filter** with dynamic lane walls, whose output always satisfies the robot's limits | 2, 3 |
| **Spatial neighbour index** (RVO2's k-d tree) | **E. Per-aisle sorted index** (1-D, s-sorted) | 4 |
| (architecture fix) | **F. MotionArbiter**: one command path per robot | command conflicts |

Some ORCA features are deliberately **not** used:
- **Free lateral dodging:** 0.6 m robots in 1.2 m lanes have no room. Lane walls block it, and deadlocks go to the existing resolver.
- **Holonomic ORCA plus a "convert (vx, vy) to (v, ω)" step: rejected.** A holonomic velocity is often not reachable by a diff-drive or car-like robot within one control period. Converting it afterwards breaks the collision guarantee and can exceed wheel, acceleration or turning limits. ORCA-DD solves directly in the robot's command space instead (see D).

## Architecture review and fix

The proposed chain was RMF → Visison planner → ORCA → emergency safety. The layers are right, but reviewing it against today's code found **six flaws**:

1. **Several components command the robot directly.** `BypassController` calls `robot.follow()` / `robot.stop()`, `EmergencyCoordinator` calls `robot.follow()` / `robot.emergency_brake()`, the Gazebo node calls `drive()`, and with RMF its adapter would command too. Their commands can interleave. For example, the controller's `_hold()` calls `robot.stop()` while the coordinator holds the robot in a standoff, and the result depends on call order: **command conflicts**.
2. **"Emergency safety" is drawn as the last *commander*.** It must be two things: a **mode input** (brake, hold, scripted reverse/rejoin) with the highest priority, and a **final gate that can only make motion safer**. It must never be a second source of motion.
3. **Nothing flows back.** When ORCA slows a robot or an emergency happens, the reservations and RMF still believe the old itinerary. That causes new conflicts later: the "miscalculation" problem, created by ourselves.
4. **Time-indexed tracking fights ORCA.** If ORCA slows the robot, the tracker sees growing lag and demands a catch-up speed. That surge undoes ORCA's slowdown and violates acceleration limits.
5. **ORCA can oppose safety-critical motion.** During an emergency brake ORCA must not "dodge". During scripted back-off and rejoin, and the bypass lane change next to the box, lateral deviation could push the robot toward the box.
6. **The layers run at different rates** (RMF about 1 Hz, planner event-driven, coordinator 10–20 Hz, control 20–50 Hz). Without a defined hand-over, a slow layer can leave a stale command active.

**Fixed architecture: every layer produces an *intent*, and exactly one component per robot turns intents into the motor command.**

```
RMF ──(task, goal)──▶ Visison planner / controller ──(reference trajectory, source=PLANNER)──┐
      ▲                                                                                      │
      │                  Emergency coordinator ──(mode: NORMAL | HOLD | BRAKE |              │
      │                                           SCRIPTED traj, source=EMERGENCY)────────────┤
      │                                                                                      ▼
      │               ┌──────────────── MotionArbiter (one per robot, the ONLY cmd_vel writer) ───────────────┐
      │               │ 1 choose the active reference by priority: BRAKE > HOLD/SCRIPTED > PLANNER > idle    │
      │               │ 2 tracker on a TIME-SCALED reference → preferred (v, ω)                                │
      │               │ 3 ORCA-DD → safe (v, ω) inside all kinematic limits (skipped in BRAKE;                 │
      │               │   speed-only along the path for SCRIPTED motion and lane changes next to obstacles)   │
      │               │ 4 safety gate: monotone (never faster, never a new direction), e-brake, limits        │
      │               │   check, stale-input watchdog → stop                                                   │
      │               │ 5 publish exactly one command per control tick (seq, stamp)                           │
      │               └──────────────────────────────────────────────────────────────────────────────────────┘
      │                                          │ feedback: actual progress, lateness, active reference
      └──── override_schedule / delay ◀── reservation rebook ◀┘
```

- **Flaws 1 and 2:** the arbiter is the only writer. The emergency layer is an input with top priority, plus the monotone gate.
- **Flaws 3 and 4:** the time-scaled reference means a slowdown becomes *lateness*, not a catch-up. Lateness goes back to the reservations and to RMF.
- **Flaw 5:** ORCA's freedom depends on the mode.
- **Flaw 6:** the arbiter samples the latest intent every tick. Intents carry timestamps, and a stale one expires to a safe stop.

## Design

### New `visison/motion_models.py`: kinematic models for any AMR
A `MotionModel` protocol. Each model gives:
- the **effective-centre map** J(θ);
- the **effective radius**;
- the **command constraints** for the current state and control period Δt, as half-planes in command space.

The models:
- **`DiffDrive(D, track_width, v_wheel_max, a_acc, a_dec, alpha_max, w_max, v_rev_max)`**
  - Effective centre P = (x, y) + D·(cos θ, sin θ), D > 0. Its velocity is ṗ = J(θ)·(v, ω) with J = [[cos θ, −D sin θ], [sin θ, D cos θ]]. J is invertible, so every ṗ corresponds to exactly one (v, ω), **with no approximation**. This is the ORCA-DD principle.
  - Wheel limits: |v ± ω·b/2| ≤ v_wheel_max, a diamond of 4 half-planes.
  - Acceleration window over Δt: v ∈ [v₀ − a_dec·Δt, v₀ + a_acc·Δt] and ω ∈ [ω₀ − α·Δt, ω₀ + α·Δt].
  - |ω| ≤ ω_max.
  - **Reverse:** v ≥ −v_rev_max, where v_rev_max = 0 unless the active intent is a SCRIPTED reverse. So ORCA can never invent a backwards move.
- **`Ackermann(D, wheelbase, R_min, v_max, v_rev_max, a_acc, a_dec, steer_rate)`** (tuggers, forklifts)
  - Same effective-point map.
  - **Turning radius:** |ω| ≤ |v|/R_min, and steering-rate limits on ω.
  - This set is non-convex (a bow-tie through the origin). It is solved as **two convex LPs**, forward (v ≥ 0) and reverse (v ≤ 0, only if reverse is allowed), and the better one is kept.
- **`Omni(vx_max, vy_max, w_max, accels)`** (mecanum): the command is (vx, vy) directly, with D = 0 and J = the rotation.
- **`validate_command(model, state, cmd, dt) -> list[str]`**: every limit the command breaks. The tests use it, and the arbiter's safety gate uses it (step 4).

### New `visison/orca.py`: ORCA-DD (standard library only)
- **`OrcaAgent(robot_id, p, p_dot, radius, cooperative)`**. `p` is the effective centre. `radius` = the largest distance from P to a footprint corner + margin, so the disc around P covers the real body.
- **`orca_half_plane(a, b, tau, dt, responsibility) -> (point, direction, u)`**. This is the RVO2 construction in ṗ-space: cut-off circle, leg projection, and the overlap case. It also returns the minimal change **u**, which feature B uses.
  - Responsibility is 0.5 for cooperative neighbours.
  - It is 1.0 for non-cooperative ones: unknown tracks, other fleets, and robots that are braking or under coordinator control.
- **`obstacle_half_plane(...)`**: the lane walls (the lateral band the robot may use now, the same logic as `ego_band`) and the edges of known obstacles.
- **The solve happens in command space, not in ṗ-space.**
  - Substituting ṗ = J(θ)·c turns every ORCA and wall half-plane into a half-plane in c = (v, D·ω). Scaling ω by D makes the Euclidean distance in c equal to the distance in ṗ.
  - The model's kinematic half-planes join the same set, so **one LP** returns the closest-to-preferred command that is both collision-free and physically executable.
  - The optimisation uses the RVO2 incremental 2D LP.
  - **Infeasible:** the 3D LP relaxes *only* the neighbour ORCA constraints (least penetration). Kinematic limits and walls stay hard.
  - **Still infeasible** (only possible if the acceleration window excludes every safe command): return the maximum-deceleration command and flag `feasible=False`, which feature B turns into BRAKE.
- **Mode-dependent freedom (flaw 5):**
  - `NORMAL`: full 2-D ORCA-DD within the lane walls.
  - `PATH_SPEED_ONLY`: only |v| along the reference path may change, and ω follows the path. Used for SCRIPTED back-off and rejoin, and for bypass lane changes within `clearance + 0.3 m` of an obstacle.
  - `OFF`: used during BRAKE.
- **`orca_1d(...)`**: the lane-locked case, where the constraints reduce to a speed interval. No LP. Used by feature C.
- **`OrcaConfig`**: τ = 2 s, τ_obs = 1 s, D (default 0.15 m), neighbour range, max neighbours, radius margin, and `u_soft_accel`.

### A. Reciprocal bypass windows (`planner.py`)
- In `BypassPlanner._candidates`, besides the current plans, add **cooperative candidates**. These let the reservations of the oncoming robots in the borrowed stretch be delayed by δ ∈ {1, 2, 3 s} (≤ `PlannerConfig.max_coop_delay`).
- The forbidden intervals are recomputed with those reservations shifted. This reuses `_build`, only feeding it the shifted intervals.
- The cost becomes the **total fleet delay**: R1's delay + Σ δ_j·(number of robots shifted) + `coop_penalty`. That compares one 12 s wait against, say, R2 slowing by 2 s.
- The chosen plan carries `coop_shifts: {robot_id: δ}`. `BypassController._decide` claims atomically, and a new `ReservationProvider.shift_many` or a sequence of `shift` calls applies the delays, rolled back if the claim fails.
- The shifted robot follows its new timing. In the sim/Gazebo that is its schedule re-timed through `displacement`/`resume_trajectory`, or simply slowed by feature C tracking its shifted reservation.
- New `Action.COOP_BYPASS` so reports and logs show it.
- Coop shifts only apply to **our own fleet's** robots. Robots of other fleets (RMF, Plan B) are never shifted by Visison; for them only the normal wait/bypass options exist.

### B. Graded response (`emergency.py`, `standoff.py`)
- For each pair `EmergencyMonitor` rates, also compute ORCA's `u`. The needed speed change is |u|, and ORCA can achieve `Δv = a_max·t_react + τ·a_max/2`.
- New level ladder:
  - `NONE`;
  - `AVOID` (ORCA handles it softly: `feasible` and |u| ≤ achievable);
  - `WARN`;
  - `BRAKE`, which now fires only when P ≥ `p_brake` **and** ORCA cannot fix it (infeasible, or |u| > achievable).
- `P ≥ p_hard` (0.8) always brakes, as a safety override.
- `EmergencyCoordinator` counts AVOID events separately, so efficiency shows up in the reports.

### C. Cooperative speed keeping in lanes
- `EmergencyCoordinator.safe_velocity()` uses `orca_1d` for every robot whose lateral band stays inside one lane. Every robot outside that case gets the full 2D filter (bypass, rejoin, lane change).
- A follower then matches the speed of a slower or stopped leader, keeping the `restart_gap`, so there are no rear-end emergency brakes.
- In `_release_queue`, a resumed follower doesn't need to wait for `restart_gap + 2 m`. It resumes, and the 1-D ORCA keeps the gap, so queues move off smoothly together.

### D. ORCA-DD inside the control loop
- `EmergencyCoordinator.orca_neighbours(rid)` supplies the inputs:
  - the agents (from `self.states`, converted to effective centres);
  - unknown tracks (non-cooperative);
  - the walls (lane band + `obstacles()`).

  The arbiter (F) calls ORCA-DD with them. The coordinator itself never commands motion.
- **Sim:** a `SimRobot` ORCA mode that **integrates the unicycle** (not a holonomic point) with the arbiter's (v, ω) each `dt`, under the same `MotionModel`. The pure sim then matches Gazebo, and adds a `--orca` flag and a `Scenario.orca` field.
- **Gazebo:** `GazeboRobot.drive()` is replaced by its arbiter's `step()`. A launch argument `orca:=true|false` (default false) switches the ORCA-DD step on. With it off, the pipeline is otherwise identical.

### F. MotionArbiter: one command path per robot (new `visison/arbiter.py`)
- **`MotionIntent(source, priority, kind, trajectory | None, stamp, ttl)`**
  - `kind` is one of `FOLLOW`, `HOLD`, `BRAKE` or `SCRIPTED`.
  - `source` is one of `PLANNER`, `EMERGENCY`, `FLEET` (RMF pause) or `OPERATOR`.
- **`MotionArbiter(robot_id, model, tracker, orca, publish)`**
  - `request(intent)` accepts an intent only if its priority is ≥ the active lock holder's. A rejected request is logged with its source, never silently mixed.
  - `release(source)` clears that source's intent.
  - `step(t, odom)` runs pipeline steps 1–5 and calls `publish(v, ω)` **exactly once**.
  - The **priority lock**: while EMERGENCY holds BRAKE, HOLD or SCRIPTED, PLANNER intents are stored but not executed. The newest PLANNER intent becomes active again on release.
- **The time-scaled reference (flaws 3 and 4):**
  - A virtual time τ advances by Δt·κ, with κ = clip(actual progress rate / reference speed, 0, 1).
  - The tracker follows `trajectory.sample(τ)`, so a slowdown delays the reference instead of causing a catch-up surge.
  - `lateness = t − τ`.
- **The safety gate (flaw 2)** is monotone:
  - |v_out| ≤ |v_orca|;
  - the sign of v never flips unless the intent is SCRIPTED;
  - `validate_command` must pass, otherwise it clamps to the nearest valid command toward zero;
  - in BRAKE, the command is maximum deceleration to zero;
  - a **watchdog**: if the intent's ttl has passed, or odometry is older than 2 ticks → HOLD.
- **Refactor, so nothing but the arbiter moves robots:**
  - `RobotInterface.follow` / `stop` / `emergency_brake` get a `source` argument, with defaults to stay backward-compatible;
  - the adapters (`SimRobot`, `GazeboRobot`, later RMF) turn these into arbiter intents;
  - `BypassController` sends PLANNER intents;
  - `EmergencyCoordinator` sends EMERGENCY intents and releases them when a standoff ends. That replaces the direct `follow` / `emergency_brake` calls in `_brake`, `_resolve_committed`, `_release_queue` and `_escalate`.
- **The feedback loop (flaw 3):**
  - the arbiter exposes `lateness` and the active reference;
  - `EmergencyCoordinator.tick` calls the existing `_rebook()` when a cruising robot's lateness exceeds `late_tolerance` (1 s);
  - the controller's `_finish` uses lateness for its shift;
  - **claims follow lateness.** If ORCA slows a robot while it is in the borrowed lane, its lane-B claims would expire before it is back. So the controller extends its remaining claims by the current lateness each tick (`try_reserve` of the extension). If the extension is refused, it raises a WARN to the coordinator, which then rates that pair with priority;
  - Plan B's RMF adapter sends the active reference with `override_schedule` whenever it changes.

### E. Per-aisle neighbour index (`standoff.py`)
- `EmergencyCoordinator` keeps `self._by_s`, the robots sorted by s, rebuilt each tick in O(n log n).
- `identify`, `_cascade`, `_detect_deadlocks`, `_nearest_ahead` and ORCA's neighbour query use range scans within ±range. That drops the O(n²) all-pairs loops.

### Efficiency metrics (proves the gains)
- `SimResult` gets `fleet_delay`, which is Σ (actual time out of the aisle − nominal time) over robots, plus `emergency_stops`, `avoid_events`, `hard_decel_events` and the coordinator's CPU time per tick.
- `python3 -m sim.aisle_sim --compare` runs each scenario three ways (baseline, `+orca`, `+orca+coop`) and prints a table of those metrics.
- A `fleet_50` stress scenario (50 robots) only measures the tick CPU time for E.

## Tests
- New `tests/test_orca.py`:
  - the half-plane geometry matches the closed form;
  - the LP matches a brute-force grid over the *command* space;
  - a non-cooperative neighbour gets responsibility 1;
  - lane walls bound the lateral motion;
  - the 3D fallback relaxes only the neighbour constraints;
  - `orca_1d` gives exactly the speed bound;
  - `PATH_SPEED_ONLY` never changes the path.
- New `tests/test_motion_models.py`, **validating the conversion**:
  - J(θ) is exactly inverse-consistent;
  - a seeded **property test** over 10 000 random states, preferred commands and neighbour sets, for DiffDrive, Ackermann and Omni. Every ORCA-DD output passes `validate_command`: wheel speed, acceleration and deceleration window, ω_max, turning radius, and no reverse unless SCRIPTED;
  - a comparison test: the naive "holonomic ORCA + convert" breaks those limits in a measurable share of cases, which the gate catches;
  - a closed-loop unicycle integration of two DiffDrive robots head-on, and of an Ackermann robot pair: no overlap, all limits held at every step.
- New `tests/test_arbiter.py`:
  - exactly one publish per `step`;
  - priority and lock: a PLANNER request during an EMERGENCY lock is stored, not executed, and becomes active on release;
  - BRAKE skips ORCA and decelerates at maximum;
  - the gate never increases |v| or flips direction;
  - the watchdog holds on a stale intent or stale odometry;
  - the time-scaled reference: under an ORCA slowdown, lateness grows and speed never goes above the reference (no catch-up surge);
  - a regression for flaw 1: controller `_hold` plus a coordinator scripted reverse in the same tick gives the coordinator's command, deterministically.
- `tests/test_planner.py`:
  - `near` with coop picks COOP_BYPASS with a lower total fleet delay than the 12 s wait;
  - coop is never chosen when it would push a shifted robot into a conflict with a third robot's reservation;
  - the shifts are rolled back if the claim fails.
- `tests/test_emergency.py`: the risk ladder. A small mismatch gives AVOID and no brake. A large mismatch still gives BRAKE, and `P ≥ p_hard` always brakes.
- `tests/test_scenarios.py`:
  - all existing scenarios pass in all three modes;
  - `fleet_10` with ORCA has **0 rear-end emergencies**;
  - a new `orca_small_error` scenario gives 0 emergencies with ORCA and 1 without;
  - `miscalc_rejoin` with ORCA still resolves by chain reverse;
  - `--compare` shows fleet delay: coop ≤ baseline on `near`.
- `test_offline.py`: the Gazebo DiffDrive model (from `amr.urdf.xacro`: track width, wheel limits, the plugin's 1 m/s² limit) through the arbiter on a head-on and a follow pair, with every command passing `validate_command`.

## Files
- **New:**
  - `visison/motion_models.py`, `visison/orca.py`, `visison/arbiter.py`;
  - `tests/test_orca.py`, `tests/test_motion_models.py`, `tests/test_arbiter.py`.
- **Changed:**
  - `visison/adapters.py` (a `source` argument on the robot commands);
  - `visison/planner.py` (A);
  - `visison/controller.py` (PLANNER intents, claim + coop shifts, lateness);
  - `visison/reservation.py` (`shift_many` with rollback);
  - `visison/emergency.py` (B);
  - `visison/standoff.py` (B, C, D, E, EMERGENCY intents instead of direct robot calls);
  - `visison/__init__.py`;
  - `sim/aisle_sim.py` (arbiter-driven `SimRobot`, unicycle integration in ORCA mode, metrics, `--compare`, new scenarios);
  - the Gazebo node (`GazeboRobot` driven by its arbiter, `orca` argument) and the launch file;
  - the tests above.
- **Docs:**
  - `docs/MATH.md` §10: the ORCA half-plane, the 1-D form, the graded response, and reciprocal windows with the fleet-delay cost;
  - the README (the layer diagram and the new options).

## Verification
1. `python3 -m pytest` and the Gazebo offline tests: everything passes.
2. `python3 -m sim.aisle_sim --compare`: the metrics table shows ORCA/coop reduce fleet delay and emergency stops with no collisions. Any scenario where that doesn't hold gets reported honestly.
3. Gazebo, headless:
   - `./run_gazebo_scenario.sh 30 orca:=true`: coop bypass and a shorter wait than today;
   - `./run_gazebo_scenario.sh 50 r2_early:=<small> orca:=true`: AVOID and no brake;
   - `./run_gazebo_scenario.sh 50 r2_early:=22 lane_b_robots:=3 orca:=true`: chain reverse still works, with no collisions.

