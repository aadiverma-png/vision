# Visison — the mathematics

All quantities are in SI units. The numbers in the worked examples are
the defaults used by `sim/aisle_sim.py`, and the simulator reproduces them.

## Setting

A two-lane aisle in lane coordinates: `s` along the aisle, `y` across it.
Lane A is `y ∈ [0, w]` and lane B is `y ∈ [w, 2w]`. Each lane is split along
`s` into reservation cells of length `Δs`. A reservation
`(robot, lane, k, r_s, r_e)` means *robot may occupy cell k of lane during the
open interval (r_s, r_e)*.

Robot R1 drives in lane A towards `+s` (the direction `d = −1` is handled by
working in the travel frame `u = d·s`). It has footprint length `ℓ` (half length
`h = ℓ/2`), width `b`, limits `v_max, a_acc, a_dec`, and a preferred bypass
speed `v_b`. An obstacle box `[o₁, o₂] × [y₁, y₂]` is detected ahead.

| symbol | meaning | sim value |
|---|---|---|
| `w` | lane width | 1.2 m |
| `ℓ × b` | footprint | 0.9 × 0.6 m |
| `v_max, a_acc, a_dec` | limits | 1.5 m/s, 0.8, 1.2 m/s² |
| `v_b` | bypass speed | 1.0 m/s |
| `a_lat` | max lateral acceleration | 0.5 m/s² |
| `c` | clearance to the obstacle | 0.15 m |
| `τ` | time margin = comm latency + jitter + other robot's braking | 0.2 + 0.5 + 0.75 = 1.45 s |
| `σ` | space margin = localization error + min gap | 0.05 + 0.30 = 0.35 m |

## 0. Squeeze check

In its own lane, the free gaps next to the obstacle are `y₁ − 0` and `w − y₂`.
If the larger gap is at least `b + 2c`, R1 shifts to the middle of that gap and
passes without entering lane B. It needs no reservation (`Action.SQUEEZE`).

## 1. Lane-change time

**Differential drive / car-like: quintic lateral profile.**
With `x = t/T`, `y(x) = w (10x³ − 15x⁴ + 6x⁵)`. Lateral velocity and
acceleration are zero at both ends. Differentiating twice,

```
ÿ(t) = (w/T²)·60x(1 − 3x + 2x²)
```

The peak is at `x = ½ ± 1/(2√3)`, where `|ÿ|max = (10/√3)·w/T²`. Setting
this equal to `a_lat` gives

```
T_lc = √( 10 w / (√3 · a_lat) )
```

With the sim values, `T_lc = √(12 / 0.866) = 3.72 s`. The distance covered
along the aisle during the lane change is `D_lc = v_b · T_lc = 3.72 m`.

**Holonomic (mecanum/omni).** A sideways trapezoid with `v_y,max, a_y,max`.
The peak speed is `v_p = min(v_y,max, √(a_y,max · w))` and
`T_lc = 2v_p/a + (w − v_p²/a)/v_p`.

Both models implement the `LaneChangeModel` protocol, so other models can be plugged in.

## 2. Bypass geometry

In the travel frame:

```
u_a = o₁ − c − h − D_lc      start of the lane change out
u_b = o₂ + c + h + D_lc      R1 fully back in lane A
```

The front of R1 stays `c` behind the obstacle until it is fully in lane B. Its
rear is `c` past the obstacle before it starts moving back. Lane B is used
over `[u_a − h − σ, u_b + h + σ]`.

Sim: `u_a = 20 − 0.15 − 0.45 − 3.72 = 15.68 m` and `u_b = 20.6 + 0.6 + 3.72 = 24.92 m`.
The borrowed stretch of lane B is `[14.88, 25.72] m`.

## 3. Crossing time

The approach is a time-optimal trapezoid from `(u₁, v₁)` to `(u_a, v_b)`. The
same `trapezoid(d, v₀, v_end)` also covers a start from standstill after
waiting (`v₀ = 0`). Its peak speed follows from energy balance:

```
v_p² = (2 a_acc a_dec d + a_dec v₀² + a_acc v_end²) / (a_acc + a_dec),   capped at v_max
```

Then

```
T_cross = T_approach + (u_b − u_a) / v_b
        = T_approach + 2 T_lc + (o₂ − o₁ + 2c + ℓ) / v_b
```

## 4. Simple safety rule (one oncoming robot)

R2 drives in lane B towards R1. Its centre is a distance `g` from the far edge
of the borrowed stretch, and it moves at speed `v₂`. It reaches that stretch at

```
t_arr2 = (g − ℓ₂/2) / v₂ − jitter
```

The bypass is safe if

```
T_cross + τ_comm + τ_brake + σ / v₂  ≤  t_arr2
```

This is `oncoming_is_clear()`. It is a quick check for one robot. §5 solves
the general case exactly.

## 5. Exact earliest window over the reservation table

For a candidate manoeuvre, invert the motion profile to find, for every lane-B
cell `k`, the interval `[α_k, β_k]` during which R1's inflated footprint
(`h + σ`) covers it. For the waiting plan the times are measured from the
departure time `t_d`. For the rolling plan they are absolute.

R1's occupancy `(t_d + α_k − τ, t_d + β_k + τ)` must not overlap any other
robot's reservation `(r_s, r_e)` on the same cell. Two open intervals overlap
exactly when `t_d + α_k − τ < r_e` and `r_s < t_d + β_k + τ`. So each
reservation forbids the departure times

```
F_{k,r} = ( r_s − β_k − τ ,  r_e − α_k + τ )
```

and the earliest safe departure is

```
t* = min { t ≥ t_arrive : t ∉ ⋃ F_{k,r} }
```

Sort the intervals, merge them, and walk from `t_arrive`. That costs
O(n log n) for `n` relevant reservations, and it is exact: no time-stepping
and no iteration. It works for R2 in either direction and for any number of
robots, and `tests/test_planner.py` checks it against a 10 ms brute-force scan
on random tables.

* A rolling candidate is accepted only if `t_now ∉ ⋃F`, which gives **BYPASS_NOW**.
* Otherwise the stop-and-go candidate uses `t*`, which gives **WAIT_THEN_BYPASS**.
  R1 stops at `u_w = u_a − v_b²/(2a_acc)` so that it is back at `v_b` exactly
  at `u_a`.
* The planner tries several bypass speeds (`speed_fractions`). Slower passes
  need a shorter window but hold lane B longer. It keeps the cheapest.

## 6. Decision cost

```
J_bypass  = delay + λ_E · E_bypass
delay     = t_back − t_nominal(u_b)        (includes the wait)
E_bypass  = e_m · 2·(arc − D_lc) + e_s · delay
J_reroute = T_reroute + P_reroute + λ_E · E_reroute
```

`P_reroute` (default 10 s) prices the disturbance to other robots' schedules,
which a reroute causes and a bypass inside a claimed window does not.
REROUTE is chosen when `J_reroute < J_bypass`, or when the wait exceeds
`W_max`. If lane B is blocked too, or no window exists within the horizon,
the result is REROUTE when a reroute is available and HOLD otherwise.

## 7. Commit, re-check, rejoin

1. The claims `(k, t* + α_k − τ, t* + β_k + τ)` go to `try_reserve`, which is
   atomic. If another robot won the window, R1 plans again.
2. R1's later reservations are shifted by `delay` straight away, so other
   robots see the new timestamps. The provider returns any conflicts it
   creates to the fleet.
3. Until the lane change out starts, every tick re-checks the claimed window
   and perception. If either fails, R1 releases the claim, undoes the shift
   and plans again.
4. At rejoin, R1 releases the lane-B claims early. Any lateness beyond
   tolerance is shifted as well.
5. The obstacle is reported to the fleet once, with lanes blocked, box,
   confidence and the action taken.

## Worked examples (sim defaults)

R1 detects the box at `t = 4 s`. It is then at `s = 8 m` and driving at `1.5 m/s`.
Bypassing without stopping, R1 is back in lane A at `t = 18.4 s`, so
`T_cross = 14.4 s`. The right-hand side margins add
`0.2 + 0.75 + 0.35/1.0 = 1.3 s`, so the bypass needs `t_arr2 ≥ 15.7 s`.

**R2 20 s away (`far`).** At detection R2 is at `s = 46 m`, so
`g = 46 − 25.72 = 20.3 m` and `t_arr2 = 19.83 − 0.5 = 19.3 s ≥ 15.7 s`.
Result: **BYPASS_NOW**. The delay is 3.2 s, against 55 s for the reroute.

**R2 5 s away (`near`).** At detection R2 is at `s = 31 m`, so
`g = 5.3 m` and `t_arr2 = 4.3 s < 15.7 s`, so it is not safe now. The interval
search gives `t* = 21.5 s`. R1 stops at `s = 15.05 m`, waits 12.2 s while R2
goes past, then bypasses. It is back in lane A at `t = 32.0 s`.
Result: **WAIT_THEN_BYPASS**. The delay is 16.7 s, against 55 s for the reroute.

## 8. Collision probability and the emergency brake

Reservations can be wrong: a robot is early or late, or a timing was
miscalculated. Every robot therefore also rates what its lidar sees
(`visison/emergency.py`). Clusters are tracked with an α-β filter, which gives a
position and velocity per object. For the ego `e` and an object `o` along the
aisle:

    g      = |s_o − s_e| − (L_e + L_o)/2                 bumper gap
    x      = v·t_r + v|v| / (2b)                         signed travel until stopped
    d_stop = sign(s_o − s_e) · (x_e − x_o)               gap used up while both stop
    m      = g − d_stop − g_min                          margin left
    σ      = √(σ_p² + (σ_v · t_stop)²)
    P      = Φ(−m/σ)                                     collision probability

- The ego reacts after `t_re` = lidar period + compute (0.15 s).
- A fleet robot coming towards it brakes after `t_ro = t_re` + fleet latency (0.35 s).
- An object moving *away* is assumed to brake at once as hard as it can, which is the worst case for a follower.
- An unknown object coming towards the ego is assumed to keep going.

Only pairs whose lateral bands overlap are rated. The ego's band covers its own
planned lane change for the next 2 s, so a bypass is rated *before* the robot
leaves its lane. The results:
- `P ≥ 0.25`: BRAKE.
- `P ≥ 0.05`: WARN.
- `g ≤ g_min` while closing: `P = 1`.

At `m = 0`, `P = 0.5`.

**Who brakes.** In a head-on pair both robots brake. In a same-direction pair only
the closing (rear) robot brakes, because braking the leader would make it worse.
Robots driving into a braking group within 12 m brake too (a cascade).

## 9. Standoff resolution and the chain reverse

Once everyone involved has stopped, `visison/standoff.py` decides:

1. **A robot between the lanes has priority.** Its body overlaps the other lane's
   corridor, so it can neither wait nor go back safely.
   - `rejoin_plan` creeps it (0.2 m/s) past the obstacle and back into its lane, or back in front of the obstacle if that uses less of the other lane.
   - Its front stays in the other lane up to `F = u_back + h`, in its travel frame.
   - The robots `j₁, j₂, …` ahead of it in that lane back off, nearest first:

         req₁     = F + g        g = g_min + space
         revᵢ     = max(0, reqᵢ + hᵢ − uᵢ)
         reqᵢ₊₁   = uᵢ + revᵢ + hᵢ + g

   - The push stops at the first robot with `rev = 0`.
   - The chain is infeasible if any `rev > 6 m`, or a robot would leave the aisle or hit a static obstacle. It then **escalates** to the fleet.
2. **A robot about to bypass, still in its own lane, gives way.** It releases its claim, undoes its schedule shift and plans again.
3. **Otherwise, queues restart front first.** Each robot waits for free road ahead.
4. **An unknown object** (for example a person) is waited out until it has left every robot's path, not merely until the alarm stops.

**Deadlock:** facing robots stopped in each other's way for 3 s form a standoff
with no alarm. All pairs that become ready together are grouped (union-find).

**Example (`miscalc_rejoin`):** R2 runs 14 s early.
- Both robots brake at `t = 13.8 s` (P = 0.90, gap 1.0 m).
- R1 is beside the box, so R2 backs off `rev = 1.39 m`.
- R1 creeps back into lane A, and R2 resumes. The closest gap was 0.51 m.

## 10. ORCA-DD, the motion arbiter and reciprocal windows (Plan A)

**One command path.** The planner, the fleet (RMF) and the emergency layer only
send *intents* (FOLLOW, HOLD, BRAKE, SCRIPTED). Each robot's `MotionArbiter` is
the only component that commands its motors. Each control tick it:

1. picks the active intent by priority: operator > emergency > fleet > planner;
2. tracks it at a **time-scaled reference time** τ, with `τ̇ = clip(v / v_ref, 0, 1)`, so a slowdown becomes lateness `t − τ` instead of a catch-up surge;
3. filters with ORCA-DD;
4. gates the command against the robot's limits, and brakes if odometry is stale.

**ORCA-DD.** A differential-drive robot cannot move its axle sideways, but the point
`P = (x, y) + D(cos θ, sin θ)` can move in any direction:

    Ṗ = R(θ) · (v, D·ω)

The map is exact and invertible, so avoidance is solved directly for the command
`c = (v, Dω)`. Every ORCA half-plane on `Ṗ` (van den Berg's construction:
cut-off circle, legs, or the overlap case) becomes a half-plane on `c`. The robot's
limits are more half-planes on `c`:
- wheel speeds `|v ± ωb/2| ≤ v_w`;
- the acceleration window;
- `|ω| ≤ ω_max`;
- no reverse unless the intent is SCRIPTED;
- for car-like robots, `|ω| ≤ |v|/R_min`, solved as two convex programs (forward and reverse).

The RVO2 2-D LP gives the executable command closest to the preferred one. When
the robots cannot all be satisfied, the 3-D LP relaxes only the robot constraints,
never the limits or walls. A property test over 10,000 random cases checks that
every returned command is executable. A naive "holonomic ORCA, then convert"
breaks the limits in a measurable share of cases.

**Lane-locked ORCA (1-D).** In an aisle the lanes already keep robots apart
sideways. So:
- ORCA runs **speed-only along the path**.
- Only robots whose lateral band overlaps count, projected onto the lane with radius `ℓ/2 + g_min/2 + margin`.
- A same-direction leader is treated `headway · v` closer (0.4 s), as adaptive cruise control does.
- A robot reacts to where others physically are, plus its own plan, so a robot about to change lanes yields to the lane's owner.
- The coordinator's scripted back-off and rejoin are not filtered against their own standoff's members, whose geometry is designed.

**Graded response.** An alarm with `P < 0.8` does not brake if ORCA-DD is already
resolving it within every closing robot's limits. That means the robot's last solve was feasible, or it is still out of the other's lane band and slowing. `P ≥ 0.8` always brakes.

**Reciprocal windows.** If a small slip of oncoming robots of *our own fleet*
(1–5 s) opens a rolling bypass, that option costs:

    cost = own delay + Σ δⱼ + penalty

The planner compares it with waiting. The other robots' bookings shift and the
claim is made atomically, and the slower robot slows to half speed long enough
to lose exactly δ (ramp loss `v/8a_dec + v/8a_acc`, the rest cruising at `v/2`).
If any step fails, everything is rolled back. Robots of other fleets are never asked.

**Results** (`python3 -m sim.aisle_sim --compare`; no collisions in any run):

| scenario | baseline | + ORCA-DD | + ORCA-DD + windows |
|---|---|---|---|
| `orca_small_error` (R2 8 s early) | 1 emergency brake, 14.2 s delay | **0 brakes, 4.6 s** | 0 brakes, 4.6 s |
| `coop_window` (R1 would wait 21 s) | 26.9 s | 26.8 s | **5.2 s** (R2 slows 2 s) |
| `fleet_10` | 2 brakes, 115.8 s | 1 brake, 93.3 s | 1 brake, 93.3 s |
| `miscalc_rejoin` (R2 14 s early) | 1 brake, 33.6 s | 0 brakes (deadlock hand-over), 41.8 s | same |
| `convoy_reverse` | 1 brake, 70.3 s | 1 brake, 89.4 s | same |

ORCA-DD removes emergency brakes for small errors and rear-end conflicts. When the
error is so large that two robots meet head-on in one lane, speed alone cannot
solve it. ORCA then stops them gently and the chain reverse resolves it, which is
slower than braking hard. The coordinator's cost per tick stays under 7 ms for 50
robots, thanks to an s-sorted neighbour index.
