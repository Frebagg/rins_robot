#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import math
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, TwistStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

#ZA DRUGI KROG
from rins_robot.msg import FaceCoords
from rins_robot.msg import RingCoords
from std_msgs.msg import String
#from rins_robot.srv import Speech
import numpy as np

class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

# Ce je True, prvi krog po tabeli koordinate ne uporablja Nav2 ciljev,
# ampak se robot prisilno premika z /cmd_vel med tockami.
# To je uporabno na majhni mapi, kjer Nav2 pogosto aborta zaradi costmap/inflation.
USE_DIRECT_CMD_VEL_ROUTE = True

# Nastavitve za pocasno in varno prisilno voznjo.
DIRECT_LINEAR_MAX = 0.10          # m/s; na pravem robotu raje pocasi
DIRECT_LINEAR_MIN = 0.035         # m/s
DIRECT_ANGULAR_MAX = 0.55         # rad/s
DIRECT_DISTANCE_TOL = 0.10        # m
DIRECT_YAW_TOL = 0.08             # rad
DIRECT_TURN_KP = 1.8
DIRECT_DRIVE_KP = 0.55
DIRECT_HEADING_KP = 1.8

# Drugi krog: robot gre po istih waypointih nazaj.
# Pri vsakem waypointu obisce samo obraze/ringe, ki so temu waypointu najblizji.
VISIT_MAX_DISTANCE_FROM_WAYPOINT = 1.60   # m; ce je detekcija dlje od vseh waypointov, jo preskoci
VISIT_FACE_STANDOFF = 0.25                # m; koliko pred obrazom naj se ustavi
VISIT_RING_STANDOFF = 0.25                # m; koliko pred ringom naj se ustavi
VISIT_APPROACH_TOL = 0.11                 # m; toleranca pri direktnem obisku objekta
VISIT_RETURN_TOL = 0.11                   # m; toleranca pri vrnitvi na waypoint

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False

        # ROS2 subscribers
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)

        #-----------------------------------------------------------------------------------------
        self.create_subscription(FaceCoords,"/face_coords",self.updateFaceCoords,10)
        self.create_subscription(RingCoords,"/ring_coords",self.updateRingCoords,10)

        # self.greetClient = self.create_client(Speech,"/greet_service")
        # self.sayColorClient = self.create_client(Speech,"/sayColor_service")

        self.speakPublisher  = self.create_publisher(String, "/speak", 10)

        self.faces = []
        self.rings = []
        #-----------------------------------------------------------------------------------------
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.get_logger().info(f"NEW Robot commander has been initialized!")
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    # ------------------------------------------------------------------
    # Direktno premikanje z /cmd_vel.
    # To obide Nav2 planner/costmap in je zato primerno samo za vnaprej
    # preverjene odseke z dovolj prostora. Robot mora biti dobro lokaliziran.
    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _angle_diff(self, target, current):
        return self._wrap_angle(target - current)

    def _clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def get_robot_yaw(self):
        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publishCmdVel(self, linear=0.0, angular=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

    def stopRobot(self):
        for _ in range(5):
            self.publishCmdVel(0.0, 0.0)
            time.sleep(0.04)

    def spinSome(self, seconds=0.5):
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def yawCodeToAngle(self, yaw_code):
        # Enaka interpretacija kot v tvoji tabeli.
        # 8 pomeni scan/obrat na mestu, 9 pomeni samo vozi do tocke brez koncnega obrata.
        if yaw_code == 0:   # gor
            return 0.0
        if yaw_code == 1:   # desno
            return -1.57
        if yaw_code == 2:   # dol
            return 3.14
        if yaw_code == 3:   # levo
            return 1.57
        if yaw_code == 4:   # gor-desno
            return -0.785
        if yaw_code == 5:   # dol-desno
            return -2.356
        if yaw_code == 6:   # gor-levo
            return 0.785
        if yaw_code == 7:   # dol-levo
            return 2.356
        return None

    def turnDirectToYaw(self, target_yaw, yaw_tolerance=DIRECT_YAW_TOL, timeout=12.0):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot turn directly because current_pose is not available.")
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            error = self._angle_diff(target_yaw, self.get_robot_yaw())

            if abs(error) <= yaw_tolerance:
                self.stopRobot()
                return True

            angular = self._clamp(DIRECT_TURN_KP * error, -DIRECT_ANGULAR_MAX, DIRECT_ANGULAR_MAX)
            self.publishCmdVel(0.0, angular)

        self.stopRobot()
        self.warn("Direct turn timed out.")
        return False

    def turnDirectRelative(self, delta_yaw, yaw_tolerance=DIRECT_YAW_TOL, timeout=20.0):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot turn relatively because current_pose is not available.")
            return False

        turned = 0.0
        last_yaw = self.get_robot_yaw()
        direction = 1.0 if delta_yaw >= 0.0 else -1.0
        target_abs = abs(delta_yaw)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            current_yaw = self.get_robot_yaw()
            step = self._wrap_angle(current_yaw - last_yaw)
            turned += abs(step)
            last_yaw = current_yaw

            remaining = target_abs - turned
            if remaining <= yaw_tolerance:
                self.stopRobot()
                return True

            speed = min(DIRECT_ANGULAR_MAX, max(0.20, 0.9 * remaining))
            self.publishCmdVel(0.0, direction * speed)

        self.stopRobot()
        self.warn("Relative direct turn timed out.")
        return False

    def driveDirectToXY(self, target_x, target_y, distance_tolerance=DIRECT_DISTANCE_TOL, timeout=None):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot drive directly because current_pose is not available.")
            return False

        start = self.get_robot_position()
        start_dist = math.hypot(target_x - start[0], target_y - start[1])
        if timeout is None:
            timeout = max(10.0, start_dist / max(DIRECT_LINEAR_MIN, DIRECT_LINEAR_MAX) * 2.5)

        self.info(f"Direct cmd_vel drive to ({target_x:.2f}, {target_y:.2f}), distance {start_dist:.2f} m")
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            robot_position = self.get_robot_position()
            dx = target_x - robot_position[0]
            dy = target_y - robot_position[1]
            distance = math.hypot(dx, dy)

            if distance <= distance_tolerance:
                self.stopRobot()
                return True

            target_yaw = math.atan2(dy, dx)
            heading_error = self._angle_diff(target_yaw, self.get_robot_yaw())
            angular = self._clamp(DIRECT_HEADING_KP * heading_error, -DIRECT_ANGULAR_MAX, DIRECT_ANGULAR_MAX)

            # Ce je robot prevec narobe obrnjen, se najprej samo obrne.
            if abs(heading_error) > 0.45:
                linear = 0.0
            else:
                linear = self._clamp(DIRECT_DRIVE_KP * distance, DIRECT_LINEAR_MIN, DIRECT_LINEAR_MAX)

            self.publishCmdVel(linear, angular)

        self.stopRobot()
        self.warn(f"Direct drive to ({target_x:.2f}, {target_y:.2f}) timed out.")
        return False

    def driveDirectWaypoint(self, waypoint_id, x, y, yaw_code):
        self.info(f"Direct waypoint {waypoint_id}: x={x:.2f}, y={y:.2f}, yaw_code={yaw_code}")

        ok = self.driveDirectToXY(x, y)
        if not ok:
            self.warn(f"Waypoint {waypoint_id}: direct movement did not fully reach target; continuing.")

        if yaw_code == 8:
            self.info(f"Waypoint {waypoint_id}: 360 degree scan with /cmd_vel.")
            self.turnDirectRelative(2.0 * math.pi, timeout=25.0)
            return ok

        if yaw_code == 9:
            # Samo pridi do tocke; ne spreminjaj koncne orientacije.
            return ok

        target_yaw = self.yawCodeToAngle(yaw_code)
        if target_yaw is not None:
            self.turnDirectToYaw(target_yaw)

        return ok

    def executeDirectWaypointRoute(self, waypoints):
        self.info("Executing first route with direct /cmd_vel waypoint following.")
        self.spinSome(0.5)
        for waypoint_id, x, y, yaw_code in waypoints:
            self.driveDirectWaypoint(waypoint_id, x, y, yaw_code)
            self.stopRobot()
            time.sleep(0.15)
        self.info("Direct /cmd_vel route finished.")
    # ------------------------------------------------------------------

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
    #---------------------------------------------------------------------------
    def updateFaceCoords(self,data):
        self.faces = list(zip(data.points, data.ids))

    def updateRingCoords(self,data):
        self.rings = list(zip(data.points, data.ids,data.colors))

    def get_robot_position(self):
        return np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])

    # ------------------------------------------------------------------
    # Drugi krog po isti poti nazaj + kratki izleti do detekcij.
    # Ideja:
    # 1) prvi krog zbira /face_coords in /ring_coords,
    # 2) za vsak obraz/ring izbere najblizji waypoint iz tabele koordinate,
    # 3) po koncu prvega kroga gre robot po waypointih nazaj,
    # 4) na vsakem waypointu gre direktno do pripetih detekcij in nazaj.
    def _waypoint_xy(self, waypoint):
        return float(waypoint[1]), float(waypoint[2])

    def _find_nearest_waypoint_index(self, point, waypoints):
        best_index = None
        best_distance = float('inf')
        for index, waypoint in enumerate(waypoints):
            wx, wy = self._waypoint_xy(waypoint)
            distance = math.hypot(point.x - wx, point.y - wy)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index, best_distance

    def _add_unique_visit(self, visits_by_waypoint, waypoint_index, visit, seen):
        key = visit[0], visit[2]
        if key in seen:
            return
        seen.add(key)
        visits_by_waypoint[waypoint_index].append(visit)

    def buildVisitsByNearestWaypoint(self, waypoints):
        visits_by_waypoint = {i: [] for i in range(len(waypoints))}
        seen = set()

        facesCopy = self.faces.copy()
        ringsCopy = self.rings.copy()

        self.info(f"Assigning {len(facesCopy)} faces and {len(ringsCopy)} rings to nearest waypoints.")

        for point, face_id in facesCopy:
            waypoint_index, distance = self._find_nearest_waypoint_index(point, waypoints)
            if waypoint_index is None:
                continue
            if distance > VISIT_MAX_DISTANCE_FROM_WAYPOINT:
                self.warn(
                    f"Skipping face {face_id}: nearest waypoint is {distance:.2f} m away "
                    f"(limit {VISIT_MAX_DISTANCE_FROM_WAYPOINT:.2f} m)."
                )
                continue
            self._add_unique_visit(
                visits_by_waypoint,
                waypoint_index,
                ("face", point, face_id, None, distance),
                seen
            )

        for point, ring_id, color in ringsCopy:
            waypoint_index, distance = self._find_nearest_waypoint_index(point, waypoints)
            if waypoint_index is None:
                continue
            if distance > VISIT_MAX_DISTANCE_FROM_WAYPOINT:
                self.warn(
                    f"Skipping ring {ring_id}: nearest waypoint is {distance:.2f} m away "
                    f"(limit {VISIT_MAX_DISTANCE_FROM_WAYPOINT:.2f} m)."
                )
                continue
            self._add_unique_visit(
                visits_by_waypoint,
                waypoint_index,
                ("ring", point, ring_id, color, distance),
                seen
            )

        for waypoint_index in visits_by_waypoint:
            visits_by_waypoint[waypoint_index].sort(key=lambda visit: visit[4])
            if visits_by_waypoint[waypoint_index]:
                waypoint_id = waypoints[waypoint_index][0]
                self.info(
                    f"Waypoint {waypoint_id} has {len(visits_by_waypoint[waypoint_index])} assigned visits."
                )

        return visits_by_waypoint

    def computeApproachPointFromWaypoint(self, waypoint_x, waypoint_y, target_x, target_y, standoff_distance):
        dx = target_x - waypoint_x
        dy = target_y - waypoint_y
        distance = math.hypot(dx, dy)

        if distance <= 1e-3:
            return waypoint_x, waypoint_y

        # Ustavi se pred objektom, gledano iz waypointa proti objektu.
        # Ce je objekt zelo blizu waypointa, se ne premikaj do njega, samo obrni robota.
        if distance <= standoff_distance + 0.05:
            return waypoint_x, waypoint_y

        ux = dx / distance
        uy = dy / distance
        approach_x = target_x - standoff_distance * ux
        approach_y = target_y - standoff_distance * uy
        return approach_x, approach_y

    def turnDirectToPoint(self, target_x, target_y, timeout=12.0):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot turn to point because current_pose is not available.")
            return False

        robot_position = self.get_robot_position()
        target_yaw = math.atan2(target_y - robot_position[1], target_x - robot_position[0])
        return self.turnDirectToYaw(target_yaw, timeout=timeout)

    def visitSingleDetectionFromWaypoint(self, waypoint_x, waypoint_y, visit):
        kind, point, object_id, color, distance_from_waypoint = visit

        if kind == "face":
            standoff = VISIT_FACE_STANDOFF
            speak_text = "Hello, human"
            label = f"face {object_id}"
        else:
            standoff = VISIT_RING_STANDOFF
            speak_text = str(color)
            label = f"ring {object_id} ({color})"

        target_x = float(point.x)
        target_y = float(point.y)
        approach_x, approach_y = self.computeApproachPointFromWaypoint(
            waypoint_x,
            waypoint_y,
            target_x,
            target_y,
            standoff
        )

        self.info(
            f"Visiting {label} from waypoint: target=({target_x:.2f}, {target_y:.2f}), "
            f"approach=({approach_x:.2f}, {approach_y:.2f}), "
            f"nearest waypoint distance={distance_from_waypoint:.2f} m."
        )

        # Kratek direktni izlet iz varne waypoint tocke do objekta.
        moved_to_object = self.driveDirectToXY(
            approach_x,
            approach_y,
            distance_tolerance=VISIT_APPROACH_TOL
        )
        if not moved_to_object:
            self.warn(f"Could not fully approach {label}; trying to continue anyway.")

        # Pred govorom se vedno obrni proti dejanski koordinati objekta.
        self.turnDirectToPoint(target_x, target_y)
        self.stopRobot()
        time.sleep(0.15)
        self.speakPublisher.publish(String(data=speak_text))
        time.sleep(0.8)

        # Vrni se po isti smeri nazaj na waypoint, nato nadaljuj po reverse route.
        returned = self.driveDirectToXY(
            waypoint_x,
            waypoint_y,
            distance_tolerance=VISIT_RETURN_TOL
        )
        if not returned:
            self.warn(f"Could not fully return to waypoint after visiting {label}; continuing route.")

        self.stopRobot()
        time.sleep(0.15)
        return moved_to_object and returned

    def visitDetectionsOnReverseRoute(self, waypoints):
        self.spinSome(1.0)
        visits_by_waypoint = self.buildVisitsByNearestWaypoint(waypoints)

        total_visits = sum(len(v) for v in visits_by_waypoint.values())
        self.info(f"Starting reverse route with {total_visits} assigned face/ring visits.")

        if not waypoints:
            self.warn("No waypoints supplied for reverse route visits.")
            return

        last_index = len(waypoints) - 1

        # Edge case je namenoma pokrit: ce je detekcija najblizja zadnjemu waypointu,
        # se obdela takoj, ker je robot po prvem krogu ze tam.
        for index in range(last_index, -1, -1):
            waypoint_id, wx, wy, yaw_code = waypoints[index]
            wx = float(wx)
            wy = float(wy)

            if index != last_index:
                self.info(f"Reverse route: returning to waypoint {waypoint_id} at ({wx:.2f}, {wy:.2f}).")
                self.driveDirectToXY(wx, wy)
                self.stopRobot()
                time.sleep(0.15)
            else:
                self.info(f"Reverse route starts at last waypoint {waypoint_id}.")

            assigned_visits = visits_by_waypoint.get(index, [])
            if not assigned_visits:
                continue

            self.info(f"Waypoint {waypoint_id}: visiting {len(assigned_visits)} assigned detections.")
            for visit in assigned_visits:
                self.visitSingleDetectionFromWaypoint(wx, wy, visit)

        self.info("Reverse route visits finished.")
    # ------------------------------------------------------------------

    def _build_standoff_goal(self, target_x, target_y, standoff_distance=0.30):
        robotPos = self.get_robot_position()
        dx = target_x - robotPos[0]
        dy = target_y - robotPos[1]
        dist = np.hypot(dx, dy)

        if dist > 1e-3:
            yaw = np.arctan2(dy, dx)
        else:
            yaw = 0.0

        if dist > standoff_distance:
            goal_x = target_x - standoff_distance * np.cos(yaw)
            goal_y = target_y - standoff_distance * np.sin(yaw)
        else:
            goal_x = target_x
            goal_y = target_y

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation = self.YawToQuaternion(yaw)

        return goal_pose

    def _go_close_enough(self, target_x, target_y, standoff_distance=0.30, close_enough_distance=0.60):
        """Navigate to a standoff point and accept failures if we are still close enough."""
        goal_pose = self._build_standoff_goal(target_x, target_y, standoff_distance)
        self.goToPose(goal_pose)

        self.info("Waiting for the task to complete...")
        while not self.isTaskComplete():
            time.sleep(1)

        task_result = self.getResult()
        if task_result == TaskResult.SUCCEEDED:
            return True

        robotPos = self.get_robot_position()
        dist_to_target = np.hypot(target_x - robotPos[0], target_y - robotPos[1])
        if dist_to_target <= close_enough_distance:
            self.warn(
                f"Navigation to exact detection point failed, but robot is close enough "
                f"({dist_to_target:.2f} m). Continuing interaction."
            )
            return True

        self.warn(
            f"Navigation failed and robot is too far from detection "
            f"({dist_to_target:.2f} m). Skipping interaction."
        )
        return False

    def visitDetections(self):
        #najprej obrazi
        ringsCopy = self.rings.copy()
        facesCopy = self.faces.copy()

        for point,id in facesCopy:
            self.info(f"Going towards face {id}.")
            x = point.x
            y = point.y

            if not self._go_close_enough(x, y, standoff_distance=0.30, close_enough_distance=0.60):
                continue
            
            """future = self.greetClient.call_async(request)
            rclpy.spin_until_future_complete(self,future)
            time.sleep(1)
            #time.sleep(2.0)
            response = future.result()
            if response is not None and response.success == True:
                self.info("Sucessfuly talked to a human!")
            else:
                self.info("Failed to talk to a human!")"""
            self.speakPublisher.publish(String(data="Hello, human"))

        #obisci ringe
        for point,id,color in ringsCopy:
            self.info(f"Going towards ring {id}.")
            x = point.x
            y = point.y

            if not self._go_close_enough(x, y, standoff_distance=0.30, close_enough_distance=0.60):
                continue
            
            """future = self.sayColorClient.call_async(request)
            rclpy.spin_until_future_complete(self,future)
            time.sleep(1)
            #time.sleep(2.0)
            response = future.result()
            if response is not None and response.success == True:
                self.info("Sucessfuly talked to a ring!")
            else:
                self.info("Failed to talk to a ring!")"""
            self.speakPublisher.publish(String(data=color))
    #--------------------------------------------------------------------------

def main(args=None):
    #print("Running new Commander!")
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()
    
    # "yaw" == 0 : gor
    # "yaw" == 1 : desno
    # "yaw" == 2 : dol
    # "yaw" == 3 : levo
    # "yaw" == 4 : gor-desno
    # "yaw" == 5 : dol-desno
    # "yaw" == 6 : gor-levo
    # "yaw" == 7 : dol-levo
    # "yaw" == 8 : obrat/scan na mestu
    # "yaw" == 9 : vozi do tocke brez koncnega obrata
    
    #koordinate = [
    #    (0, 0.0, 0.0, 1),
    #    (1, 1.3, 1.8, 4),
    #    (2, 0.5, 2.75, 1),
    #    (3, -1.5, 2.5, 0),
    #    (3, -1.5, 2.5, 7),
    #    (4, -2.6, 2.75, 4),
    #    (4.5, -2.5, 0.4, 0),
    #    (5, -1.9, -0.45, 3),
    #    (6, -1.85, -1.3, 3),
    #    (6, -1.85, -1.3, 2),
    #    (6.5, -1.2, -1.3, 1),
    #    (6.5, -1.2, -2.5, 0),
    #    (7, 0.0, -1.6, 3),
    #    (7.5, 0.6, -3.8, 0),
    #    (8, 1.4, -2.5, 2),
    #    (8, 1.4, -2.5, 1),
    #    (9, 2.55, -1.3, 2),
    #    (10, 1.0, 0.3, 1),
    #    (10.5, -0.4, 1.2, 2),
    #    (11, -1.75, -0.3, 3)
    #]
    # koordinate = [ (1, -1.2, -0.3, 7),
    #               (1, -1.2, -0.3, 0),
    #               (1, -1.2, -0.3, 2),
    #               (2, -1.55, -0.6, 4),
    #               (3, -0.1, 0.1, 5),
    #               (3, -0.1, 0.1, 7),
    #               (4, -0.1, 0.9, 2),
    #               (5, 1.1, 1.25, 4),
    #               (5, 1.1, 1.25, 7),
    #               (6, 0.75, 1.85, 5),
    #               (7, 1.35, 2.45, 6)
    #              ]
    koordinate = [ (1, -1.5, -0.55, 8),
                   (2, -0.9, 0.2, 8),
                   (3, -0.35, -0.35, 8),
                   (4, 0.25, 0.25, 8),
                   (5, 0.05, 0.7, 8),
                   (6, 0.55, 1.05, 8),
                   (7, 1.05, 1.5, 8),
                   (8, 0.8, 1.8, 8),
                   (9, 1.35, 2.35, 8)
                  ]
    
    #---------------------------------------------------------------------
    #PRVI KROG - DETEKCIJE
    # Na mali mapi Nav2 vcasih aborta tudi v odprtem prostoru.
    # Zato lahko prvi krog med tockami iz tabele izvede direktno z /cmd_vel.
    if USE_DIRECT_CMD_VEL_ROUTE:
        rc.executeDirectWaypointRoute(koordinate)
    else:
        for id, x, y, yaw in koordinate:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = rc.get_clock().now().to_msg()

            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            if yaw == 0: # gor
                goal_pose.pose.orientation = rc.YawToQuaternion(0)
            elif yaw == 1: # desno
                goal_pose.pose.orientation = rc.YawToQuaternion(-1.57)
            elif yaw == 2: # dol
                goal_pose.pose.orientation = rc.YawToQuaternion(3.14)
            elif yaw == 3: # levo
                goal_pose.pose.orientation = rc.YawToQuaternion(1.57)
            elif yaw == 4: # gor-desno
                goal_pose.pose.orientation = rc.YawToQuaternion(-0.785)
            elif yaw == 5: # dol-desno
                goal_pose.pose.orientation = rc.YawToQuaternion(-2.356)
            elif yaw == 6: # gor-levo
                goal_pose.pose.orientation = rc.YawToQuaternion(0.785)
            elif yaw == 7: # dol-levo
                goal_pose.pose.orientation = rc.YawToQuaternion(2.356)
            elif yaw == 8:
                if hasattr(rc, 'current_pose'):
                    goal_pose.pose.orientation = rc.current_pose.pose.orientation
                else:
                    goal_pose.pose.orientation = rc.YawToQuaternion(0)
            elif yaw == 9:
                if hasattr(rc, 'current_pose'):
                    goal_pose.pose.orientation = rc.current_pose.pose.orientation
                else:
                    goal_pose.pose.orientation = rc.YawToQuaternion(0)

            rc.goToPose(goal_pose)

            rc.info("Waiting for the task to complete...")
            while not rc.isTaskComplete():
                #rc.info("Waiting for the task to complete...")
                time.sleep(1)

            if yaw == 8:
                rc.spin(2 * np.pi, time_allowance=20)
                rc.info("Waiting for the spin to complete...")
                while not rc.isTaskComplete():
                    time.sleep(1)
    
    time.sleep(2)
    #-----------------------------------------------------------------
    #DRUGI KROG - OBISKI DETECTIONOV
    rc.info("Going to visit detections on reverse waypoint route now")
    rc.visitDetectionsOnReverseRoute(koordinate)
    rc.info("Finishing, give good grade!")
    #-------------------------------------------------------------------

    rc.destroyNode()
    

    # rc.destroyNode()
    # rclpy.shutdown()
    
    # # Finally send it a goal to reach
    # goal_pose = PoseStamped()
    # goal_pose.header.frame_id = 'map'
    # goal_pose.header.stamp = rc.get_clock().now().to_msg()

    # goal_pose.pose.position.x = 0.8
    # goal_pose.pose.position.y = 4.4
    # goal_pose.pose.orientation = rc.YawToQuaternion(0.57)

    # rc.goToPose(goal_pose)

    # while not rc.isTaskComplete():
    #     rc.info("Waiting for the task to complete...")
    #     time.sleep(1)

    # rc.spin(-0.57)

    # rc.destroyNode()

    # And a simple example
if __name__=="__main__":
    main()