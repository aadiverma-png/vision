import pytest

from sim.aisle_sim import AISLE, SCENARIOS, brakes, fleet_n, run
from visison import Action, EmergencyConfig, Resolution

EMERGENCY = [n for n, s in SCENARIOS.items() if s.resolutions]
NORMAL = [n for n, s in SCENARIOS.items() if not s.resolutions]
BASE = [n for n, s in SCENARIOS.items() if not s.orca]


@pytest.fixture(scope="module")
def results():
    return {name: run(sc, quiet=True) for name, sc in SCENARIOS.items()}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_is_collision_free_and_decides_as_expected(results, name):
    res = results[name]
    assert not res.collisions, res.collisions
    assert res.action is SCENARIOS[name].expected, res.reason
    assert res.reports and len({r.reporter_id for r in res.reports}) == len(res.reports)
    assert all(res.finished.values()), res.finished
    assert res.ok


@pytest.mark.parametrize("name", NORMAL)
def test_correct_reservations_never_trigger_an_emergency(results, name):
    assert not results[name].emergencies


@pytest.mark.parametrize("name", EMERGENCY)
def test_miscalculated_timing_is_caught_and_resolved(results, name):
    res = results[name]
    assert [e.event for e in res.emergencies][0] == "brake"
    for r in SCENARIOS[name].resolutions:
        assert r in res.resolutions
    assert Resolution.ESCALATE not in res.resolutions
    assert all(so.phase.value == "done" for so in res.standoffs)
    # the robots stopped with the planned gap, not just "did not touch"
    assert res.min_gap >= EmergencyConfig().min_gap


def test_miscalc_rejoin_brakes_both_and_r2_backs_off_for_r1(results):
    res = results["miscalc_rejoin"]
    brake = res.emergencies[0]
    assert set(brake.robots) == {"R1", "R2"}
    so = res.standoffs[0]
    assert so.resolution is Resolution.CHAIN_REVERSE and so.rejoined == ["R1"]
    assert set(so.moved) == {"R2"} and 0 < so.moved["R2"] < EmergencyConfig().max_reverse
    assert res.max_y_r1 > AISLE.lane_width  # R1 really was between the lanes


def test_convoy_backs_off_as_a_chain(results):
    so = results["convoy_reverse"].standoffs[0]
    moves = [so.moved.get(r, 0.0) for r in ("R2", "R3", "R4")]
    # each robot backs off less than the one in front of it
    assert moves[0] > moves[1] > moves[2] >= 0.0


def test_fleet_of_ten_resolves_every_standoff(results):
    res = results["fleet_10"]
    assert len(res.finished) == 10
    assert res.standoffs and all(so.phase.value == "done" for so in res.standoffs)


def test_near_waits_until_r2_has_left_the_zone():
    res = run(SCENARIOS["near"], quiet=True)
    plan = res.plan
    assert plan.wait_time > 0
    # R2 drives towards smaller s; it must be past the borrowed stretch
    assert res.r2_s_at_change_out < plan.s_out


def test_far_does_not_stop():
    res = run(SCENARIOS["far"], quiet=True)
    assert res.plan.rolling and res.plan.wait_time == 0


def test_squeeze_never_enters_lane_b():
    res = run(SCENARIOS["squeeze"], quiet=True)
    assert res.max_y_r1 <= AISLE.lane_width


def test_bypass_saves_time_over_reroute():
    for name in ("far", "near"):
        d = run(SCENARIOS[name], quiet=True).decision
        assert d.cost_bypass < d.cost_reroute


# ------------------------------------------------------------------ Plan A: ORCA-DD mode

@pytest.fixture(scope="module")
def orca_results():
    return {name: run(SCENARIOS[name], quiet=True, orca=True) for name in BASE}


@pytest.mark.parametrize("name", BASE)
def test_every_scenario_is_collision_free_with_orca(orca_results, name):
    res = orca_results[name]
    assert not res.collisions, res.collisions
    assert all(res.finished.values()), res.finished
    assert Resolution.ESCALATE not in res.resolutions


@pytest.mark.parametrize("name", [n for n in NORMAL if n in BASE])
def test_orca_changes_nothing_when_nothing_is_wrong(results, orca_results, name):
    base, orca = results[name], orca_results[name]
    assert orca.action is base.action and not orca.emergencies
    assert orca.fleet_delay <= base.fleet_delay + 0.5


def test_orca_never_adds_emergency_brakes(results, orca_results):
    for name in BASE:
        assert brakes(orca_results[name]) <= brakes(results[name]), name


def test_fleet_of_ten_with_orca_has_no_rear_end_brakes(orca_results):
    res = orca_results["fleet_10"]
    rear = [e for e in res.emergencies if e.event == "brake" and "queue" in e.message]
    assert not rear and brakes(res) <= 1


def test_coop_window_cuts_fleet_delay(results):
    res = results["coop_window"]
    base = run(SCENARIOS["coop_window"], quiet=True, orca=False, coop=False)
    assert res.action is Action.COOP_BYPASS and res.coop_events
    assert res.fleet_delay < 0.5 * base.fleet_delay and not res.collisions


def test_fifty_robots_are_collision_free_and_cheap():
    res = run(fleet_n(50), quiet=True, orca=True, t_max=40.0)
    assert not res.collisions
    assert res.tick_cpu < 0.05  # one coordinator tick for 50 robots
