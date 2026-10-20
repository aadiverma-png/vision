"""Plays every Visison case, one after another, in one Gazebo + RViz session.

    ros2 launch visison_gazebo demo_all.launch.py          # or ./run_demo.sh

For each case the director
1. shows a title card in RViz (``/visison/demo``),
2. removes the previous case's robots from Gazebo and spawns the new ones
   (every robot with its own 2-D lidar),
3. runs the Visison scenario node for that case (lidar perception, bypass
   planning, emergency layer, ORCA-DD, RMF ...) until it reports its result,
4. keeps the result and moves on. At the end it shows a results table.

No terminal command is needed between cases. The simulation runs at the
real-time factor given to the launch file (default 2x).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import rclpy
import xacro
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from . import scenario as sc
from .bypass_scenario_node import BypassScenario

WORLD = "aisle_world"
MAX_ROBOTS = 8

# Every case: a title, one line of explanation, and the scenario parameters.
CASES = [
    ("Normal bypass", "Lane B is free long enough: R1 borrows it at once (BYPASS_NOW)",
     dict(r2_arrival=50.0)),
    ("Wait, then bypass", "R2 is coming soon: R1 stops, lets R2 pass, then bypasses (WAIT_THEN_BYPASS)",
     dict(r2_arrival=30.0)),
    ("Emergency: timing miscalculated", "R2 runs 22 s ahead of its booking and meets R1 mid-bypass: both brake, "
     "R2 backs off, R1 rejoins lane A", dict(r2_arrival=50.0, r2_early=22.0)),
    ("Emergency: convoy chain reverse", "3 robots run early: the whole chain backs off just enough for R1",
     dict(r2_arrival=50.0, r2_early=22.0, lane_b_robots=3)),
    ("ORCA-DD: small timing error", "R2 runs early, but ORCA-DD slows both robots smoothly: no emergency brake",
     dict(r2_arrival=50.0, r2_early=16.0, orca=True)),
    ("ORCA-DD: convoy", "ORCA-DD stops the convoy gently; the deadlock is handed to the chain reverse",
     dict(r2_arrival=50.0, r2_early=22.0, lane_b_robots=3, orca=True)),
    ("Reciprocal window", "Instead of R1 waiting long, R2 slows down by a few seconds and R1 bypasses at once",
     dict(r2_arrival=39.0, orca=True, coop=True)),
    ("Fleet of 6 with ORCA-DD", "R1 + 2 followers in lane A, 3 robots in lane B: queues, bypasses, no collisions",
     dict(r2_arrival=55.0, lane_b_robots=3, lane_a_followers=2, orca=True)),
    ("Open-RMF mixed fleet", "Tasks from Open-RMF; lane B is another vendor's fleet; R1's bypass window respects "
     "R2's RMF itinerary and is published to the RMF schedule", dict(r2_arrival=35.0, rmf=True)),
]


def _run(cmd: list[str], timeout: float = 20.0) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        pass


class DemoDirector(Node):
    def __init__(self):
        super().__init__("visison_demo_director")
        self.declare_parameter("cases", "all")  # "all" or comma-separated case numbers, e.g. "1,3,9"
        self.declare_parameter("case_timeout", 260.0)  # s of simulation time per case
        self.declare_parameter("pause", 4.0)  # s of wall time between cases
        pick = str(self.get_parameter("cases").value)
        idx = range(len(CASES)) if pick == "all" else [int(i) - 1 for i in pick.split(",") if i.strip()]
        self.cases = [(i, CASES[i]) for i in idx]
        self.timeout = float(self.get_parameter("case_timeout").value)
        self.pause = float(self.get_parameter("pause").value)
        self.demo_pub = self.create_publisher(MarkerArray, "/visison/demo", 10)
        self.status_pub = self.create_publisher(String, "/visison/demo_status", 10)
        self.model = os.path.join(get_package_share_directory("visison_gazebo"), "urdf", "amr.urdf.xacro")
        try:
            self.rmf_share = get_package_share_directory("visison_rmf")
        except Exception:  # RMF integration not built: the RMF case is skipped
            self.rmf_share = ""
        self.results: list[tuple[str, str, list[str]]] = []
        self.current = ""
        self.create_timer(0.5, self._publish_board)

    # ------------------------------------------------------------------ gazebo

    def _remove_all(self) -> None:
        threads = [threading.Thread(target=_run, args=([
            "gz", "service", "-s", f"/world/{WORLD}/remove", "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean", "--timeout", "2000", "--req", f'name: "r{i}" type: MODEL'],))
            for i in range(1, MAX_ROBOTS + 1)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    def _spawn(self, layout) -> None:
        threads = []
        for plan in layout:
            urdf = xacro.process_file(self.model, mappings={"ns": plan.ns, "lidar": "true",
                                                            "color": plan.color}).toxml()
            x, y, yaw = plan.spawn
            cmd = ["ros2", "run", "ros_gz_sim", "create", "-world", WORLD, "-string", urdf, "-name", plan.ns,
                   "-x", str(x), "-y", str(y), "-z", "0.05", "-Y", str(yaw)]
            threads.append(threading.Thread(target=_run, args=(cmd, 40.0)))
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    # -------------------------------------------------------------------- cases

    def run_all(self, executor) -> None:
        time.sleep(3.0)  # let Gazebo, the bridge and RViz come up
        for n, (i, (title, text, params)) in enumerate(self.cases, 1):
            if params.get("rmf") and not self.rmf_share:
                self.results.append((title, "SKIPPED", ["visison_rmf is not built"]))
                continue
            self.current = f"Case {i + 1}/{len(CASES)}: {title}\n{text}"
            self.get_logger().warning("=" * 70)
            self.get_logger().warning(f"CASE {i + 1}: {title} - {text}")
            self._remove_all()
            time.sleep(1.5)
            layout = sc.fleet_layout(int(params.get("lane_b_robots", 1)), int(params.get("lane_a_followers", 0)),
                                     float(params.get("r2_arrival", 50.0)), 0.8, float(params.get("r2_early", 0.0)))
            self._spawn(layout)
            overrides = dict(use_sim_time=True, rmf_share=self.rmf_share, **params)
            node = BypassScenario(f"visison_case_{i + 1}", overrides)
            executor.add_node(node)
            while rclpy.ok():
                time.sleep(0.2)
                if node.result_pass is not None:  # the case has finished and written its summary
                    break
                if node.t0 is not None and node.scenario_time() > self.timeout:
                    node.get_logger().error(f"case timed out after {self.timeout:.0f} s of simulation time")
                    break
            verdict = {True: "PASS", False: "CHECK", None: "TIMEOUT"}[node.result_pass]
            self.results.append((title, verdict, node.result_lines))
            self.get_logger().warning(f"CASE {i + 1} RESULT: {verdict}")
            node.shutdown()
            executor.remove_node(node)
            time.sleep(1.0)  # let callbacks already running on the executor finish
            node.destroy_node()
            self.current = f"Case {i + 1}/{len(CASES)}: {title} -> {verdict}"
            time.sleep(self.pause)
        self._remove_all()
        self.current = "All cases done"
        self.get_logger().warning("=" * 70)
        for title, verdict, _ in self.results:
            self.get_logger().warning(f"  {verdict:<8} {title}")
        self.get_logger().warning("=" * 70)
        self.status_pub.publish(String(data="\n".join(f"{v:<8} {t}" for t, v, _ in self.results)))

    # -------------------------------------------------------------------- rviz

    def _publish_board(self) -> None:
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def text(mid, x, y, z, size, rgba, body):
            m = Marker()
            m.header.frame_id, m.header.stamp = "world", stamp
            m.ns, m.id, m.type, m.action = "demo", mid, Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
            m.pose.orientation.w = 1.0
            m.scale.z = float(size)
            m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
            m.text = body
            ma.markers.append(m)

        text(0, sc.AISLE.length / 2, sc.AISLE.lane_width, 5.2, 0.7, (0.05, 0.15, 0.55, 1.0), self.current)
        board = "\n".join(f"{v:<7} {t}" for t, v, _ in self.results) or "results appear here"
        text(1, -8.0, sc.AISLE.lane_width, 4.0, 0.45, (0.1, 0.4, 0.1, 1.0), "RESULTS\n" + board)
        self.demo_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    director = DemoDirector()
    executor = SingleThreadedExecutor()
    executor.add_node(director)
    worker = threading.Thread(target=director.run_all, args=(executor,), daemon=True)
    worker.start()
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
