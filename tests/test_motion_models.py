"""ORCA-DD must always return a command the robot can execute."""

import math
import random

import pytest

from visison import Ackermann, Command, DiffDrive, Omni, OrcaAgent, OrcaConfig, OrcaMode, orca_solve, validate_command
from visison.motion_models import rotate

DD = DiffDrive(track_width=0.4, v_wheel_max=1.0, a_acc=0.8, a_dec=1.2, alpha_max=3.0, w_max=1.5)
ACK = Ackermann(wheelbase=0.8, R_min=1.2, v_max=1.2, a_acc=0.6, a_dec=1.0, yaw_accel_max=1.5)
OMNI = Omni(vx_max=1.0, vy_max=0.6, w_max=1.5, a_max=0.8, alpha_max=3.0)
DT = 0.05


@pytest.mark.parametrize("model", [DD, ACK])
def test_effective_centre_map_is_exact(model):
    rng = random.Random(0)
    for _ in range(200):
        th, v, w = rng.uniform(-math.pi, math.pi), rng.uniform(-1, 1), rng.uniform(-1, 1)
        # numerical derivative of P = (x, y) + D (cos th, sin th) under the unicycle
        x, y, h = 0.3, -0.2, 1e-6
        p0 = (x + model.D * math.cos(th), y + model.D * math.sin(th))
        x1, y1, th1 = x + v * math.cos(th) * h, y + v * math.sin(th) * h, th + w * h
        p1 = (x1 + model.D * math.cos(th1), y1 + model.D * math.sin(th1))
        num = ((p1[0] - p0[0]) / h, (p1[1] - p0[1]) / h)
        exact = rotate(*model.to_c(Command(v, w)), th)
        assert num == pytest.approx(exact, abs=1e-4)
        back = model.from_c(model.to_c(Command(v, w)))
        assert (back.v, back.w) == pytest.approx((v, w))


def _random_case(rng, model):
    th = rng.uniform(-math.pi, math.pi)
    if model is ACK:
        v0 = rng.uniform(0.0, 1.0)
        cur = Command(v0, rng.uniform(-1, 1) * v0 / ACK.R_min)
    elif model is OMNI:
        cur = Command(rng.uniform(-0.1, 0.9), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3))
    else:
        cur = Command(rng.uniform(-0.1, 0.9), rng.uniform(-0.5, 0.5))
    pref = Command(rng.uniform(-1.0, 1.5), rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0))
    x, y = rng.uniform(0, 10), rng.uniform(0, 2)
    ego = OrcaAgent("e", (x + model.D * math.cos(th), y + model.D * math.sin(th)), rotate(*model.to_c(cur), th), 0.45)
    nbs = [OrcaAgent(f"n{j}", (x + rng.uniform(-3, 3), y + rng.uniform(-2, 2)),
                     (rng.uniform(-1, 1), rng.uniform(-1, 1)), 0.45, rng.random() < 0.7)
           for j in range(rng.randint(0, 4))]
    return th, cur, pref, ego, nbs


@pytest.mark.parametrize("model", [DD, ACK, OMNI], ids=["diff_drive", "ackermann", "omni"])
def test_every_orca_dd_command_respects_every_limit(model):
    rng = random.Random(7)
    for _ in range(3400):  # ~10 000 cases over the three models
        th, cur, pref, ego, nbs = _random_case(rng, model)
        reverse = rng.random() < 0.3
        mode = rng.choice([OrcaMode.NORMAL, OrcaMode.PATH_SPEED_ONLY])
        r = orca_solve(model, cur, pref, th, ego, nbs, [], OrcaConfig(), DT, mode, reverse)
        assert not validate_command(model, cur, r.cmd, DT, reverse), (cur, r.cmd)


def test_no_reverse_unless_allowed():
    rng = random.Random(3)
    for _ in range(500):
        th, cur, pref, ego, nbs = _random_case(rng, DD)
        cur = Command(abs(cur.v), cur.w)
        r = orca_solve(DD, cur, Command(-0.5, pref.w), th, ego, nbs, [], OrcaConfig(), DT, OrcaMode.NORMAL, False)
        assert r.cmd.v >= -1e-9


def test_naive_holonomic_conversion_breaks_limits_that_orca_dd_keeps():
    """Holonomic ORCA + 'convert (vx, vy) to (v, w)' is not executable in general."""
    rng = random.Random(11)
    naive_bad = 0
    for _ in range(1000):
        th, cur, pref, ego, nbs = _random_case(rng, DD)
        # naive: holonomic velocity -> v = |p_dot| cos(err), w = k * err, no limits in the solve
        vx, vy = rotate(pref.v, 0.0, th)
        err = math.atan2(math.sin(math.atan2(vy, vx) - th), math.cos(math.atan2(vy, vx) - th))
        naive = Command(math.hypot(vx, vy) * math.cos(err), 3.0 * err)
        naive_bad += bool(validate_command(DD, cur, naive, DT))
        r = orca_solve(DD, cur, pref, th, ego, nbs, [], OrcaConfig(), DT)
        assert not validate_command(DD, cur, r.cmd, DT)
    assert naive_bad > 100  # a measurable share of naive commands are not executable


def test_ackermann_turning_radius_is_never_violated():
    rng = random.Random(5)
    for _ in range(1000):
        th, cur, pref, ego, nbs = _random_case(rng, ACK)
        r = orca_solve(ACK, cur, Command(pref.v, 3.0 * pref.w), th, ego, nbs, [], OrcaConfig(), DT)
        assert abs(r.cmd.w) <= abs(r.cmd.v) / ACK.R_min + 1e-6
