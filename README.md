# Visison — reservation-aware obstacle bypass for AMRs

An autonomous mobile robot (R1) is driving along lane A of a two-lane
warehouse aisle when a box falls off a shelf and blocks its lane. Usually it has
two options: trace back and reroute (for example with PIBT), or turn around. Both
cost time and battery and break the timestamps other robots rely on.

Visison adds a third option. **Borrow lane B, pass the box, and return to lane A
before anyone who has reserved lane B needs that space.** It combines
closed-form kinematics with an exact search over the space-time reservation
table, and it picks one of these actions:

| action | when |
|---|---|
| `CONTINUE` | the obstacle is not in the robot's path |
| `SQUEEZE` | the gap next to the box is wide enough to pass inside lane A |
| `BYPASS_NOW` | lane B is free long enough right now (e.g. R2 is 20 s away) |
| `WAIT_THEN_BYPASS` | lane B is busy now (e.g. R2 is 5 s away). R1 stops, waits for the earliest safe window `t*`, then bypasses |
| `COOP_BYPASS` | bypass now, because oncoming robots of our own fleet slip by 1–5 s and that costs less than waiting |
| `REROUTE` | a reroute is cheaper, the wait is too long, or both lanes are blocked |
| `HOLD` | nothing is possible and no reroute exists. R1 waits and re-evaluates |
| `REJOIN` | emergency only: creep back into the robot's own lane (see the emergency layer) |

In every case R1 **reports the obstacle** to the fleet so a person can remove it.

```
 lane B  ◄── R2 ─────────────────────────────  (reserved space-time cells)
         ─────────────╮▁▁▁▁▁▁▁▁╭───────────────
 lane A  R1 ──────────╯  [box] ╰──────────────►
              u_a  change out  pass  change in  u_b
```

---

## Taking just the core

`visison/` **is** the algorithm: 15 modules, ~4,100 lines, **pure Python standard
library, zero dependencies**. Nothing in it imports `sim/`, `tests/` or anything
under `ROS2_Gazebo_Bridge/` — those are a simulator, a test suite and a reference
ROS 2 integration built *on* the library, not part of it. So the core drops into
any stack two ways:

```bash
pip install "git+https://github.com/aadiverma-png/vision"   # installs the visison package only
cp -r visison/ /path/to/your/project/                       # or just copy the folder
```

Then implement only the protocols your system needs — `RobotInterface` to drive
the robot and `ReservationProvider` to reach your fleet's schedule are the two
required ones; see [Integrating with a robot and fleet](#integrating-with-a-robot-and-fleet).
`InMemoryReservationTable` and `LoggingReporter` are working reference
implementations you can start against before wiring the real fleet.

| you want | you need |
|---|---|
| the bypass decision only | `BypassPlanner.evaluate()` — pure function, no I/O, no threads |
| decision + execution | add `BypassController` and `RobotInterface` |
| multi-robot safety | add `EmergencyCoordinator` (one per fleet) and `MotionArbiter` (one per robot) |
| reciprocal avoidance | add ORCA-DD via `MotionArbiter`'s `orca` hook |

Each layer is usable without the ones below it. The planner is the natural
starting point: it owns no state beyond its config and answers one question.

---

## Architecture

Visison is a **library, not a framework**: it owns no threads, no middleware and
no I/O. The host calls it, and it calls back through four small protocols.
Five layers run inside one robot, plus one fleet-level coordinator shared by all
of them.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ HOST / FLEET     Open-RMF · vendor fleet manager · operator · sim          │
└─────────────┬─────────────────────────────────────────────────────────┬────┘
              │ ReservationProvider · FleetReporter · ReroutePlanner    │ FLEET /
              │ RobotInterface       (visison.adapters)                 │ OPERATOR
              │                                                         │ intents
              │                                                         │
              ▼                                                         │
┌────────────────────────────────┐                                      │
│ 1  PERCEPTION                  │                                      │
│    ObjectTracker (α-β filter)  │                                      │
│    lidar → TrackedObject       │    ┌──────────────────────────────┐  │
│    emergency.py                │    │ 5  FLEET SAFETY (1/fleet)    │  │
└─────────────┬──────────────────┘    │    EmergencyMonitor          │  │
  Obstacle    │     tracks ──────────▶│    EmergencyCoordinator      │  │
              │                       │    standoffs, chain reverse  │  │
              ▼                       │    emergency.py, standoff.py │  │
┌────────────────────────────────┐    └───────────┬──────────────────┘  │
│ 2  DECISION                    │                │ EMERGENCY intents   │
│    BypassPlanner               │                │                     │
│    geometry · kinematics       │                │                     │
│    intervals · reservation     │                │                     │
│    planner.py → Decision       │                │                     │
└─────────────┬──────────────────┘                │                     │
  BypassPlan  │                                   │                     │
              │                                   │                     │
              ▼                                   │                     │
┌────────────────────────────────┐                │                     │
│ 3  EXECUTION                   │                │                     │
│    BypassController            │                │                     │
│    claim · re-check · rejoin   │                │                     │
│    controller.py               │                │                     │
└─────────────┬──────────────────┘                │                     │
  PLANNER     │                                   │                     │
  intents     │                                   │                     │
              ▼                                   ▼                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 4  COMMAND — MotionArbiter: the ONLY writer of motor commands              │
│    priority → time-scaled reference → tracker → ORCA-DD → gate             │
│    arbiter.py · tracking.py · orca.py · motion_models.py                   │
└─────────────────────────────┬──────────────────────────────────────────────┘
                              ▼  exactly one (v, ω) command per tick
                        RobotInterface → wheels
```

### The one-command-path invariant

Every layer above the motors only states what it *wants*, as a `MotionIntent`.
The arbiter resolves them and publishes **exactly one command per tick**. This is
what keeps the planner, the fleet and the emergency layer from fighting over the
same robot.

| source | priority | typical intent |
|---|---|---|
| `OPERATOR` | 4 (highest) | `HOLD` — a person took over |
| `EMERGENCY` | 3 | `BRAKE`, `SCRIPTED` (back-off, rejoin), `HOLD` |
| `FLEET` | 2 | `HOLD` — an Open-RMF pause |
| `PLANNER` | 1 | `FOLLOW` — the bypass trajectory |

A lower-priority intent that arrives while a higher one is active is stored, not
executed, and takes over when the higher one is released. `release(..., handover=True)`
turns an in-progress emergency motion into the planner's intent so it finishes
cleanly.

### Module responsibilities

| layer | module | responsibility |
|---|---|---|
| — | `geometry.py` | aisle/lane frame, footprints, oriented boxes, squeeze offset |
| — | `kinematics.py` | trapezoidal 1-D profiles (invertible), quintic + holonomic lane-change models |
| — | `intervals.py` | open-interval algebra, `earliest_free_point` |
| — | `reservation.py` | `ReservationProvider` protocol, thread-safe in-memory table, swept occupancy |
| 1 | `emergency.py` | α-β lidar tracker, collision probability, per-robot `EmergencyMonitor` |
| 2 | `planner.py` | `BypassPlanner`: candidates, exact `t*`, cost comparison, `Decision` |
| 3 | `controller.py` | 13-state machine, atomic claim, schedule shift, abort, rejoin |
| 4 | `arbiter.py` | intent priority, time-scaled reference, safety gate, watchdog |
| 4 | `orca.py` | ORCA-DD half-planes, RVO2 2-D/3-D linear programs |
| 4 | `motion_models.py` | `DiffDrive`, `Ackermann`, `Omni`: command-space limits, validation |
| 4 | `tracking.py` | Kanayama pose tracker (handles a standing-still or reversing reference) |
| 5 | `standoff.py` | `EmergencyCoordinator`, standoff grouping, chain reverse, deadlock detection |
| — | `adapters.py` | the four host protocols |
| — | `reporting.py` | obstacle and emergency reports |

### Controller state machine

```
CRUISING → DETECTED → EVALUATING ─┬→ APPROACH → WAITING → CHANGE_OUT → PASSING → CHANGE_IN → REJOINED
                                  ├→ REROUTING
                                  └→ HOLDING  (re-evaluates every hold_retry s)

any state ──emergency_stop──▶ EMERGENCY_STOP ──rejoin───▶ REJOINING → REJOINED (or HOLDING)
                                              ──yield_lane─▶ HOLDING
                                              ──resume─────▶ CRUISING (or HOLDING)
```

The commit point is `CHANGE_OUT`. Before it, every tick re-checks the claimed
window and perception, and either check failing releases the claim, undoes the
schedule shift and re-plans. After it, the manoeuvre is completed — the robot is
already partly in lane B, so stopping there is the worse option.

### Control and data flow per tick

1. Lidar → `ObjectTracker.update` → confirmed tracks.
2. Tracks → `EmergencyMonitor.assess` → `RiskAssessment` per pair, up to the fleet.
3. `EmergencyCoordinator.tick` → standoffs, brake/scripted intents, ORCA inputs.
4. `BypassController.tick` → state machine → `FOLLOW`/`HOLD` intent.
5. `MotionArbiter.step` → tracker → ORCA-DD → limit gate → one `Command`.
6. `RobotInterface` applies it; odometry comes back as `RobotState`.

Reservations flow the other way: `BypassPlanner` reads the table, the controller
claims atomically, shifts its own later bookings by the delay, and releases on
rejoin.

---

## Mathematical stack

Each layer is solved in closed form or by an exact combinatorial method. Nothing
in the decision path time-steps, iterates to convergence or samples randomly, so
the same inputs always give the same decision. `docs/MATH.md` has the full
derivations and worked examples; the section numbers below point into it.

| # | layer | problem | method | cost |
|---|---|---|---|---|
| 0 | squeeze | fits beside the box? | lateral gap comparison | O(1) |
| 1 | lane change | time for a lateral shift `w` under `a_lat` | quintic polynomial, closed-form peak acceleration | O(1) |
| 2 | bypass geometry | which stretch of lane B is borrowed | inflated-footprint sweep | O(1) |
| 3 | crossing time | time from here to back-in-lane | time-optimal trapezoid, energy-balance peak speed | O(1) |
| 4 | quick check | one oncoming robot | inequality on arrival times | O(1) |
| 5 | **exact window** | earliest safe departure `t*` | invert the profile per cell → forbidden open intervals → merge → first free point | O(n log n) |
| 6 | choice | bypass vs reroute | weighted delay + energy cost, schedule-disturbance penalty | O(1) |
| 7 | commit | keeping the claim honest | atomic compare-and-set, re-check, rollback | O(n) |
| 8 | risk | will we touch after both brake? | α-β filter + Gaussian margin, `P = Φ(−m/σ)` | O(1)/pair |
| 9 | standoff | who reverses, how far | recursive back-off chain with feasibility test | O(k) robots |
| 10 | avoidance | executable collision-free command | ORCA half-planes mapped into command space, RVO2 2-D/3-D LP | O(m) half-planes |

### The equations that matter

**Lane-change time** (§1) — the quintic `y(x) = w(10x³ − 15x⁴ + 6x⁵)` has zero
lateral velocity and acceleration at both ends; its peak lateral acceleration is
`(10/√3)·w/T²`, so capping it at `a_lat` gives

```
T_lc = √( 10 w / (√3 · a_lat) )        D_lc = v_b · T_lc
```

**Borrowed stretch** (§2) — in the travel frame `u = d·s`, with half-length
`h = ℓ/2`, clearance `c` and space margin `σ`:

```
u_a = o₁ − c − h − D_lc                 lane change out starts
u_b = o₂ + c + h + D_lc                 fully back in lane A
lane B is used over  [u_a − h − σ,  u_b + h + σ]
```

**Trapezoid peak speed** (§3) — from energy balance, capped at `v_max`:

```
v_p² = (2 a_acc a_dec d + a_dec v₀² + a_acc v_end²) / (a_acc + a_dec)
```

**Exact earliest window** (§5) — the heart of the planner. Inverting the motion
profile gives, for every lane-B cell `k`, the interval `[α_k, β_k]` in which the
inflated footprint covers it. A reservation `(r_s, r_e)` on that cell then
forbids exactly the departure times

```
F_{k,r} = ( r_s − β_k − τ ,  r_e − α_k + τ )
t*      = min { t ≥ t_arrive : t ∉ ⋃ F_{k,r} }
```

Sorting and merging the forbidden intervals finds `t*` in O(n log n) — no
time-stepping, no fixed threshold. Intervals are **open**, so two that merely
touch do not conflict, and merging is conservative so an isolated instant is
never offered as a window. `tests/test_planner.py` checks `t*` against a 10 ms
brute-force scan over random tables.

**Decision cost** (§6):

```
J_bypass  = delay + λ_E·E_bypass,      delay = t_back − t_nominal(u_b)
J_reroute = T_reroute + P_reroute + λ_E·E_reroute
```

`P_reroute` (default 10 s) prices the disturbance a reroute causes to other
robots' schedules, which a bypass inside a claimed window does not.

**Collision probability** (§8) — the probability that two bodies still touch
after both have braked as hard as they can:

```
g      = |s_o − s_e| − (L_e + L_o)/2            bumper gap
x      = v·t_r + v|v| / (2b)                    signed travel until stopped
d_stop = sign(s_o − s_e) · (x_e − x_o)
m      = g − d_stop − g_min                     margin left
σ      = √( σ_p² + (σ_v · t_stop)² )
P      = Φ( −m / σ )
```

`P ≥ 0.25` brakes, `P ≥ 0.05` warns, `P ≥ 0.8` always brakes even if ORCA-DD
could still resolve it. At `m = 0`, `P = 0.5`. `σ` grows with stopping time,
because a velocity error accumulates over it.

**Chain reverse** (§9) — the robot stuck between the lanes needs the other lane
up to `F = u_back + h`. The robots ahead of it back off, nearest first:

```
req₁   = F + g                      g = g_min + space
revᵢ   = max(0, reqᵢ + hᵢ − uᵢ)
reqᵢ₊₁ = uᵢ + revᵢ + hᵢ + g
```

The push stops at the first robot with `rev = 0`; any `rev > 6 m`, or a robot
that would leave the aisle, escalates to the fleet instead.

**ORCA-DD** (§10) — a differential-drive robot cannot move its axle sideways,
but the point `P = (x, y) + D(cos θ, sin θ)` can move in any direction:

```
Ṗ = R(θ) · (v, D·ω)
```

The map is exact and invertible, so avoidance is solved directly for the
*command* `c = (v, Dω)`. Every ORCA half-plane on `Ṗ` becomes a half-plane on
`c`, and the robot's own limits — wheel speeds `|v ± ωb/2| ≤ v_w`, the
acceleration window, `|ω| ≤ ω_max`, no reverse unless `SCRIPTED`, and for
car-like robots `|ω| ≤ |v|/R_min` — are more half-planes on `c`. The RVO2 2-D LP
returns the executable command closest to the preferred one; when the neighbour
constraints cannot all be met, the 3-D LP relaxes **only** those, never the
limits or walls. So the answer is always executable — there is no after-the-fact
holonomic-to-unicycle conversion, which a property test over 10,000 random cases
confirms.

**Reciprocal windows** (§10) — if slipping oncoming robots of our own fleet by
1–5 s opens a rolling bypass, its cost is `own delay + Σ δⱼ + penalty`, compared
with waiting. The slowed robot drops to half speed just long enough to lose
exactly `δ` (ramp loss `v/8a_dec + v/8a_acc`, the rest cruising at `v/2`). All of
it is claimed atomically and rolled back on any failure. Robots of other fleets
are never asked.

---

## Requirements and install

Python ≥ 3.10. The `visison` package uses **only the standard library** —
`pytest` is needed for the tests and `matplotlib` only for `--plot`.

```bash
pip install -e .                 # or just add the repo root to PYTHONPATH
pip install -e '.[test,plot]'    # tests and plotting
```

The Gazebo and Open-RMF layers are optional and live in `ROS2_Gazebo_Bridge/`.
They need ROS 2 Jazzy, Gazebo Harmonic and `ros-jazzy-rmf-dev`.

## Quickstart

```bash
python3 -m sim.aisle_sim                      # all 10 scenarios (normal, emergency, ORCA, cooperative)
python3 -m sim.aisle_sim --scenario convoy_reverse --plot convoy.png
python3 -m sim.aisle_sim --compare --stress   # baseline vs ORCA-DD vs ORCA-DD + reciprocal windows
```

| sim flag | effect |
|---|---|
| `--scenario NAME` | one of `far`, `near`, `both_blocked`, `squeeze`, `miscalc_rejoin`, `miscalc_yield`, `convoy_reverse`, `fleet_10`, `orca_small_error`, `coop_window`, `all` |
| `--orca` | every robot runs ORCA-DD (unicycle integration) |
| `--coop` | allow reciprocal bypass windows |
| `--compare` | metrics table: baseline vs `+orca` vs `+orca+coop` |
| `--stress` | with `--compare`, add a 50-robot CPU test |
| `--plot PNG` | save a top view and a space-time plot |
| `--dt` | integration step in seconds (default `0.05`) |

**Gazebo:** `./run_gazebo_scenario.sh 50` builds and launches scenario 1 with the
GUI and RViz; the number is when R2 passes the fallen box. See
[`ROS2_Gazebo_Bridge/src/visison_gazebo`](ROS2_Gazebo_Bridge/src/visison_gazebo/README.md)
for the launch arguments and the expected decision for each arrival time.

## Using the planner directly

```python
from visison import *

aisle = Aisle(lane_width=1.2, length=40.0)
spec = RobotSpec(Footprint(0.9, 0.6), MotionLimits(v_max=1.5, accel=0.8, decel=1.2),
                 QuinticLaneChange(max_lateral_accel=0.5), bypass_speed=1.0)
planner = BypassPlanner(aisle, spec)
table = InMemoryReservationTable()                # or your fleet manager's table

decision = planner.evaluate(
    RobotState("R1", s=8.0, v=1.5, lane="A"),
    Obstacle(20.0, 20.6, 0.1, 1.1),
    t_now=0.0,
    reservations=table,                           # any ReservationProvider
    reroute=RerouteEstimate(extra_time=45, extra_distance=60),
)
decision.action, decision.reason
decision.plan.trajectory.sample(t)                # reference pose for your tracker
decision.plan.claims                              # lane-B reservations to claim
```

## Integrating with a robot and fleet

`BypassController` runs the whole cycle:

1. It reports the obstacle and evaluates the options.
2. It claims the window atomically, and plans again if the claim is refused.
3. It shifts the robot's later reservations by the delay.
4. Until the lane change starts, every tick re-checks the window and perception. If either fails, it aborts safely.
5. When the robot is back in lane A, it releases the lane-B claims.

You implement four small protocols:

| protocol | module | what you provide |
|---|---|---|
| `RobotInterface` | `visison.adapters` | `state()`, `follow(trajectory)`, `stop()` |
| `ReservationProvider` | `visison.reservation` | `query`, atomic `try_reserve`, `release`, `shift` |
| `ReroutePlanner` (optional) | `visison.adapters` | `estimate()` → `RerouteEstimate`, `execute()` |
| `FleetReporter` (optional) | `visison.adapters` | `report_obstacle()`, `report_schedule_conflicts()` |

`InMemoryReservationTable` and `LoggingReporter` are reference
implementations. `examples/ros2_adapter.py` is an untested skeleton that
shows the ROS 2 wiring.

```python
ctl = BypassController(robot, planner, reservations, reporter, rerouter)
ctl.on_obstacle(obstacle, t_now, other_lane_obstacles)   # from perception
ctl.tick(t_now)                                          # 10–50 Hz
```

Differential-drive and car-like robots use `QuinticLaneChange`. Mecanum and
omni robots use `HolonomicLaneChange`. Both robot directions (`direction=±1`)
are supported.

## Tuning

Every tunable is a frozen dataclass with documented SI defaults; pass your own
instead of relying on the defaults.

| dataclass | module | what it covers | notable defaults |
|---|---|---|---|
| `RobotSpec` | `planner` | footprint, limits, lane-change model, bypass speed, energy rates | `energy_per_meter=1.0`, `energy_per_second=0.2` |
| `SafetyMargins` | `planner` | the buffers behind `τ` and `σ` | `τ = 0.2 + 0.5 + 0.75 = 1.45 s`, `σ = 0.05 + 0.30 = 0.35 m` |
| `PlannerConfig` | `planner` | clearance, speeds to try, wait/horizon limits, costs, cooperation | `max_wait=90 s`, `horizon=300 s`, `reroute_penalty=10 s`, `coop_delays=()` |
| `EmergencyConfig` | `emergency` | reaction latencies, sensor noise, risk thresholds, standoff rules | `p_brake=0.25`, `p_hard=0.8`, `max_reverse=6 m`, `cascade_range=12 m`, `standoff_timeout=60 s` |
| `OrcaConfig` | `orca` | avoidance horizons, neighbour range, margins, headway | `tau=2 s`, `neighbour_range=6 m`, `max_neighbours=8`, `headway=0.4 s` |

Cooperation is **off by default**: set `coop_delays=(1, 2, 3)` to let the planner
ask oncoming robots of your own fleet to slip.

## When the schedule is wrong: emergency layer

Reservations can be wrong: a robot is early or late, or a timing was
miscalculated. Every robot rates its lidar tracks with a collision
probability (`visison/emergency.py`, MATH.md §8). A high risk brakes **every robot involved**
(`visison/standoff.py`, §9), and then the standoff is resolved:

| situation | resolution |
|---|---|
| a robot is stuck **between the lanes** | **chain reverse**: the robots in front of it back off exactly as far as needed, one after another, and it rejoins its lane |
| a robot was **about to** bypass | it gives way and plans again |
| a queue caught up with a stopped robot | the queue restarts, front first |
| an unknown object (e.g. a person) | hold until it has left the path |
| facing robots stopped for 3 s | deadlock → the same resolver |
| nothing works / timeout | escalate to the fleet / an operator |

It works for any number of robots (`EmergencyCoordinator.register`), and the
coordinator's cost per tick stays under 7 ms for 50 robots thanks to an
`s`-sorted neighbour index.

## ORCA-DD and one command path (Plan A)

- **`MotionArbiter`** (`visison/arbiter.py`): planner, fleet and emergency layer only send *intents*. One arbiter per robot turns them into the single motor command. It uses priorities, a time-scaled reference, a limit gate and a watchdog.
- **ORCA-DD** (`visison/orca.py`, `visison/motion_models.py`): reciprocal collision avoidance solved in the robot's *command space*. Every command respects the wheel, acceleration, turn-rate, turning-radius and reverse limits, for diff-drive, Ackermann and omni robots. In aisles it runs lane-locked (speed only) with a time headway.
- **Graded response**: small timing errors are absorbed by ORCA without an emergency brake.
- **Reciprocal windows**: oncoming robots of our own fleet slip by 1–5 s when that saves more than the bypasser would wait.

Measured with `python3 -m sim.aisle_sim --compare` (no collisions in any run):

| scenario | baseline | + ORCA-DD | + ORCA-DD + windows |
|---|---|---|---|
| `orca_small_error` (R2 8 s early) | 1 emergency brake, 14.2 s delay | **0 brakes, 4.6 s** | 0 brakes, 4.6 s |
| `coop_window` (R1 would wait 21 s) | 26.9 s | 26.8 s | **5.2 s** (R2 slows 2 s) |

`fleet_10`, `miscalc_rejoin` and `convoy_reverse` are in MATH.md §10.

ORCA-DD removes emergency brakes for small errors and rear-end conflicts. When
two robots meet head-on in one lane, speed alone cannot solve it: ORCA stops them
gently and the chain reverse takes over, which is slower than braking hard.

See MATH.md §10 and `docs/plans/PLAN_A_ORCA.md`. In the sim: `--orca`, `--coop`. In Gazebo: `orca:=true coop:=true`.

## Open-RMF (Plan B)

`ROS2_Gazebo_Bridge/src/visison_rmf` connects Visison to Open-RMF (ROS 2 Jazzy, `ros-jazzy-rmf-dev`):

- **Tasks and routes come from RMF**, through an EasyFullControl fleet adapter. RMF never drives a robot directly: a `navigate` goal becomes the controller's mission, and the controller issues it through the robot's `MotionArbiter`. An RMF `stop` is a FLEET hold.
- **RMF sees Visison's real motion.** A bypass through the other lane, a rejoin or an emergency hold is published with `override_schedule`, so the other fleets' planners negotiate around it. In the demo, RMF holds the other fleet's robot while R1 borrows its lane.
- **Visison sees other fleets.** `RmfTrafficWatcher` turns their itineraries from the RMF traffic schedule into read-only reservations, so bypass windows respect robots Visison cannot command.
- **Other fleets are never commanded.** The emergency layer brakes, backs off or resumes only our own robots. If another fleet's robot would have to move, the standoff escalates.

```bash
./run_gazebo_scenario.sh 35 rmf:=true      # R1 = Visison fleet, lane B = another vendor's fleet
```

## The complete demo

```bash
./run_demo.sh                 # every case, one after another, Gazebo + RViz, 2x speed
./run_demo.sh cases:=3,4,9    # a selection;  speed:=1.0 for real time
```

A demo director removes and spawns the robots of each case in the same Gazebo
session, runs the case, and shows its title and result in RViz. No terminal
command is needed between cases. The nine cases are defined in
`visison_gazebo/demo_director.py` (`CASES`): normal bypass; wait then bypass;
emergency after a timing miscalculation; convoy chain reverse; an ORCA-DD small
error; an ORCA-DD convoy; a reciprocal window; a fleet of 6 robots; an Open-RMF
mixed fleet.

RViz shows every robot's lidar scan, the lidar tracks with velocity arrows, the lane-B
reservations, the plans, the emergency zones and the results board.

## Tests

```bash
python3 -m pytest                                              # 138 tests, pure Python
python3 -m pytest ROS2_Gazebo_Bridge/src/visison_gazebo/test    # scenario checks, no ROS needed
python3 -m pytest ROS2_Gazebo_Bridge/src/visison_rmf/test       # RMF conversions, no RMF needed
```

The suite is mostly differential, not golden: closed-form kinematics are checked
against numerical integration, `t*` against a 10 ms brute-force scan over 60 random
reservation tables, and every ORCA-DD command against the motion model's hard
limits over ~10,000 random cases spread across the three motion models.

| file | checks |
|---|---|
| `test_kinematics.py` | profiles and lane-change models vs numerical integration |
| `test_intervals.py` | open-interval merge and `earliest_free_point` |
| `test_planner.py` | `t*` vs brute force, action selection, cost comparison |
| `test_controller.py` | state machine, claim refusal, abort, rejoin |
| `test_emergency.py` | risk model, tracker association, thresholds |
| `test_motion_models.py` | command-space limits for diff-drive, Ackermann, omni (3400 random cases each) |
| `test_orca.py` | half-plane construction, LP feasibility, executability |
| `test_arbiter.py` | priority, hand-over, time-scaled reference, watchdog |
| `test_scenarios.py` | end-to-end sim runs, oriented-box collision check every step |

## Layout

```
visison/   the library (see "Module responsibilities" above)
sim/       aisle_sim.py: two-lane simulator, oriented-box collision check every step
tests/     kinematics vs numerical integration, t* vs brute force, controller, scenarios
docs/      MATH.md, plans/PLAN_A_ORCA.md, plans/PLAN_B_OPEN_RMF.md
examples/  ros2_adapter.py
ROS2_Gazebo_Bridge/src/
           amr_sim/        the AMR model, world and bridge config
           visison_gazebo/ scenario + demo nodes, lidar perception, Kanayama tracker, RViz
           visison_rmf/    Open-RMF fleet adapter, traffic watcher, read-only reservations
```

## Scope and limitations

- The aisle model is **two lanes**. Wider corridors, junctions and open floor are out of scope; the host's global planner owns those and Visison only answers the bypass question.
- The reservation search needs a **fleet table**. Without one, `InMemoryReservationTable` makes the robot the only booker and the answer degenerates to "bypass if perception is clear".
- Reciprocal windows apply **only to robots Visison commands**. Other fleets are read-only, by design.
- `examples/ros2_adapter.py` is an **untested skeleton**, not a supported adapter.
- The ORCA-DD results above come from the simulator and headless Gazebo runs, not from hardware.

## License

MIT — see [LICENSE](LICENSE). You may copy `visison/` into a closed-source
product; the only requirement is that the copyright notice travels with it.
