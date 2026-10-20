"""Skeleton: running Visison on a ROS 2 robot.

UNTESTED integration sketch. It shows where each Visison interface
connects to typical ROS 2 topics. Adapt the message types, frames and the
reservation client to your fleet manager.

    odom (nav_msgs/Odometry) ──► RosRobot.state()
    perception obstacles     ──► node._on_obstacles() ──► controller.on_obstacle()
    RosRobot.follow()        ──► timer publishes trajectory.sample(now) as a pose/twist setpoint
    RosFleetReporter         ──► /fleet/obstacle_reports
"""

from __future__ import annotations

import json
import math

import rclpy  # type: ignore
from geometry_msgs.msg import Twist  # type: ignore
from nav_msgs.msg import Odometry  # type: ignore
from rclpy.node import Node  # type: ignore
from std_msgs.msg import String  # type: ignore

from visison import (
    Aisle,
    BypassController,
    BypassPlanner,
    Footprint,
    MotionLimits,
    Obstacle,
    QuinticLaneChange,
    RobotSpec,
    RobotState,
)


class AisleFrame:
    """Map <-> aisle-frame transform for one straight aisle (origin, heading)."""

    def __init__(self, x0: float, y0: float, yaw: float):
        self.x0, self.y0, self.c, self.s = x0, y0, math.cos(yaw), math.sin(yaw)

    def to_aisle(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.x0, y - self.y0
        return self.c * dx + self.s * dy, -self.s * dx + self.c * dy


class RosRobot:
    """Implements visison.adapters.RobotInterface on top of odom + cmd_vel."""

    def __init__(self, node: Node, robot_id: str, frame: AisleFrame, lane: str, direction: int):
        self.node, self.robot_id, self.frame = node, robot_id, frame
        self.lane, self.direction = lane, direction
        self.odom: Odometry | None = None
        self.trajectory = None
        self.cmd = node.create_publisher(Twist, "cmd_vel", 10)
        node.create_subscription(Odometry, "odom", self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def state(self) -> RobotState:
        p = self.odom.pose.pose.position
        s, y = self.frame.to_aisle(p.x, p.y)
        v = abs(self.odom.twist.twist.linear.x)
        return RobotState(self.robot_id, s, v, self.lane, self.direction, y)

    def follow(self, trajectory) -> None:
        self.trajectory = trajectory  # the tracking timer below takes over

    def stop(self) -> None:
        self.trajectory = None
        self.cmd.publish(Twist())  # replace with a proper deceleration ramp

    def track(self, t: float) -> None:
        """Very simple feed-forward + heading tracking; use your own controller."""
        if self.trajectory is None:
            return
        ref = self.trajectory.sample(t)
        cmd = Twist()
        cmd.linear.x = ref.v
        # add lateral/heading feedback from self.state() vs ref here
        self.cmd.publish(cmd)


class RosFleetReporter:
    def __init__(self, node: Node):
        self.pub = node.create_publisher(String, "/fleet/obstacle_reports", 10)

    def report_obstacle(self, report) -> None:
        self.pub.publish(String(data=json.dumps(report.to_dict())))

    def report_schedule_conflicts(self, robot_id, conflicts) -> None:
        self.pub.publish(String(data=json.dumps({"robot": robot_id, "conflicts": len(conflicts)})))


class VisisonNode(Node):
    def __init__(self):
        super().__init__("visison")
        aisle = Aisle(lane_width=1.2, length=40.0, aisle_id="aisle-7")
        spec = RobotSpec(Footprint(0.9, 0.6), MotionLimits(1.5, 0.8, 1.2), QuinticLaneChange(0.5), 1.0)
        self.robot = RosRobot(self, "amr-12", AisleFrame(0.0, 0.0, 0.0), lane="A", direction=+1)
        reservations = None  # plug in a client for your fleet manager's reservation table
        self.ctl = BypassController(self.robot, BypassPlanner(aisle, spec), reservations,
                                    RosFleetReporter(self))
        self.create_timer(0.05, self._tick)
        # subscribe to your perception output and call self._on_obstacles(...)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_obstacles(self, boxes: list[tuple[float, float, float, float, str]]) -> None:
        """Boxes already transformed into the aisle frame: (s_min, s_max, y_min, y_max, id)."""
        t = self.now()
        obstacles = [Obstacle(*b[:4], obstacle_id=b[4]) for b in boxes]
        lane_b = self.ctl.planner.aisle.lane_bounds("B")
        others = tuple(o for o in obstacles if o.overlaps_lateral(*lane_b))
        for o in obstacles:
            if o.overlaps_lateral(*self.ctl.planner.aisle.lane_bounds(self.robot.lane)):
                self.ctl.on_obstacle(o, t, others)

    def _tick(self) -> None:
        t = self.now()
        self.ctl.tick(t)
        self.robot.track(t)


def main():
    rclpy.init()
    rclpy.spin(VisisonNode())


if __name__ == "__main__":
    main()
