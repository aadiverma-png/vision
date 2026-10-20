"""ROS/RMF-free checks of the Open-RMF integration pieces (plain pytest)."""

import pytest

from visison import (
    Aisle, BypassPlanner, EmergencyConfig, EmergencyCoordinator, FleetMember, Obstacle, RobotState,
    plan_chain_reverse,
)
from visison_gazebo.scenario import AISLE, CONFIG, FALLEN_BOX, LANE_A_Y, LANE_B_Y, SPEC
from visison_rmf.core import goal_trajectory, itinerary_reservations, override_path


def test_rmf_goal_becomes_a_lane_trajectory_that_stops_at_the_goal():
    st = RobotState("R1", 2.0, 0.0, "A", 1, LANE_A_Y)
    traj = goal_trajectory(AISLE, st, SPEC, 39.0, 5.0)
    end = traj.sample(traj.profile.t_end + 1.0)
    assert end.s == pytest.approx(39.0, abs=1e-6) and end.v == pytest.approx(0.0)
    assert max(traj.sample(5.0 + 0.5 * k).v for k in range(200)) <= SPEC.limits.v_max + 1e-9
    assert goal_trajectory(AISLE, st, SPEC, -5.0, 5.0) is None  # behind: RMF's next leg handles it


def test_goal_trajectory_recentres_a_robot_that_is_off_its_lane_centre():
    st = RobotState("R1", 25.0, 0.0, "A", 1, LANE_A_Y + 0.2)
    traj = goal_trajectory(AISLE, st, SPEC, 39.0, 0.0)
    assert traj.sample(traj.profile.t_end).y == pytest.approx(LANE_A_Y, abs=1e-6)


def test_bypass_is_published_with_its_excursion_into_the_other_lane():
    st = RobotState("R1", 12.0, SPEC.limits.v_max, "A", 1, LANE_A_Y)
    plan = BypassPlanner(AISLE, SPEC, CONFIG).evaluate(st, FALLEN_BOX, 13.0).plan
    pts = override_path(plan.trajectory, 13.0, plan.t_back + 2.0)
    ys = [p[1] for p in pts]
    assert max(ys) == pytest.approx(LANE_B_Y, abs=0.02)  # it really goes through lane B
    assert ys[0] == pytest.approx(LANE_A_Y) and ys[-1] == pytest.approx(LANE_A_Y, abs=1e-6)
    assert all(len(p) == 3 for p in pts)


def test_other_fleet_itinerary_becomes_timed_reservations_on_its_lane():
    # an RMF itinerary down lane B: from x=38 at t=10 to x=2 at t=55 (0.8 m/s)
    wps = [(10.0, 38.0, LANE_B_Y), (55.0, 2.0, LANE_B_Y)]
    res = itinerary_reservations(AISLE, "R2", wps, SPEC.footprint.length, CONFIG.margins.space)
    assert res and all(r.lane == "B" and r.robot_id == "R2" for r in res)
    box_cell = AISLE.cells(20.3, 20.5)[0]
    r = next(r for r in res if r.cell == box_cell)
    t_pass = 10.0 + (38.0 - 20.4) / 0.8
    assert r.start < t_pass < r.end and r.end - r.start < 4.0


def test_waypoints_outside_the_lanes_are_ignored():
    wps = [(0.0, 10.0, 5.0), (10.0, 20.0, 5.0)]  # y = 5 is off the aisle
    assert itinerary_reservations(AISLE, "X", wps, 0.6, 0.35) == []


class _Still:
    def __init__(self, st):
        self.st, self.robot_id = st, st.robot_id
        self.commands = []

    def state(self):
        return self.st

    def follow(self, *a, **k):
        self.commands.append("follow")

    def emergency_brake(self, *a, **k):
        self.commands.append("brake")

    def stop(self, *a, **k):
        self.commands.append("stop")


def test_another_fleets_robot_is_never_commanded_and_blocks_a_chain_reverse():
    """Cross-vendor motion commands are impossible: the standoff escalates instead."""
    from visison import BypassController, InMemoryReservationTable

    r1 = RobotState("R1", 20.4, 0.0, "A", 1, LANE_B_Y)  # between the lanes, beside the box
    r2 = RobotState("R2", 21.6, 0.0, "B", -1, LANE_B_Y)  # other fleet, in the way
    rob1, rob2 = _Still(r1), _Still(r2)
    planner = BypassPlanner(AISLE, SPEC, CONFIG)
    ctl = BypassController(rob1, planner, InMemoryReservationTable())
    ctl.obstacle = FALLEN_BOX
    coord = EmergencyCoordinator(planner, None, None, EmergencyConfig())
    coord.register("R1", FleetMember(rob1, SPEC, ctl))
    coord.register("R2", FleetMember(rob2, SPEC, None, cooperative=False))
    coord.tick(0.0)
    from visison.standoff import Standoff
    so = Standoff(1, {"R1", "R2"}, 0.0)
    coord.standoffs.append(so)
    coord._brake(0.0, so, so.members, "test")
    coord._resolve(0.0, so)
    assert rob2.commands == []  # never braked, never moved
    assert so.resolution.value == "escalate" and "another fleet" in so.escalation
