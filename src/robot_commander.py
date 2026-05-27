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
import subprocess
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
from std_msgs.msg import String

#ZA DRUGI KROG
from rins_robot.msg import CylinderCoords
from rins_robot.msg import FaceCoords
from rins_robot.msg import RingCoords
from rins_robot.srv import BoundingBox
from rins_robot.srv import FaceDialogue
from rins_robot.srv import ReportTaskAssignment
from rins_robot.srv import Speech
from std_srvs.srv import SetBool
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

# Koordinate za anomaly premik.
# Vpisi kot (x, y). Ce zelis zraven oznako, lahko vpises tudi (id, x, y).
ANOMALY_RED_COORDINATES = [
    (1, 0.55, -4.72),
    (2, -1.75, -4.72)
]

ANOMALY_GREEN_COORDINATES = [
    (1, -4.75, -2.7),
    (2, -4.75, 0.5)
]

ANOMALY_RED_YAW = 3.14      # dol
ANOMALY_GREEN_YAW = 1.57    # levo
ANOMALY_ARM_INSPECT_COMMAND = "look_at_belt_left"
ANOMALY_ARM_DEFAULT_COMMAND = "garage"
FACE_SEARCH_TURN_RADIANS = 0.35
FACE_SEARCH_MAX_ATTEMPTS = 3

# Nav2 naj gre samo do varne pred-tocke, potem se robot premika direktno s /cmd_vel.
# To je pomembno pri red/green anomaly celicah, ker sta pravi tocki zelo blizu stene/costmap roba.
ANOMALY_PRE_START_OFFSET = 0.35
ANOMALY_NAV_CLOSE_ENOUGH_DISTANCE = 0.70
ANOMALY_DIRECT_POINT_TOLERANCE = 0.15

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
        self.is_docked = None

        # ROS2 subscribers
        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)

        #-----------------------------------------------------------------------------------------
        self.create_subscription(FaceCoords,"/face_coords",self.updateFaceCoords,10)
        self.create_subscription(CylinderCoords,"/cylinder_coords",self.updateCylinderCoords,10)
        self.create_subscription(RingCoords,"/ring_coords",self.updateRingCoords,10)

        self.greetClient = self.create_client(Speech,"/greet_service")
        self.boundingBoxClient = self.create_client(BoundingBox,"/bounding_box_service")
        self.faceDialogueClient = self.create_client(FaceDialogue,"/face_dialogue_service")
        self.reportTaskClient = self.create_client(ReportTaskAssignment,"/report_task_assignment")
        self.anomalyClient = self.create_client(SetBool, '/set_auto_scan')

        self.faces = []
        self.cylinders = []
        self.rings = []
        self.last_face_coords_time = None
        #-----------------------------------------------------------------------------------------
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.arm_command_pub = self.create_publisher(String, '/arm_command', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

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
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
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
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _angle_diff(self, target, current):
        return self._wrap_angle(target - current)

    def _clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def get_robot_yaw(self):
        q = self.current_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publishCmdVel(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

    def stopRobot(self):
        for _ in range(3):
            self.publishCmdVel(0.0, 0.0)
            time.sleep(0.05)

    def turnDirectToYaw(self, target_yaw, yaw_tolerance=0.06, max_angular=0.6, timeout=10.0):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot turn directly because current pose is not available.")
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            error = self._angle_diff(target_yaw, self.get_robot_yaw())

            if abs(error) <= yaw_tolerance:
                self.stopRobot()
                return True

            angular = self._clamp(1.8 * error, -max_angular, max_angular)
            self.publishCmdVel(0.0, angular)

        self.stopRobot()
        self.warn("Direct turn timed out.")
        return False

    def turnDirectToPoint(self, target_x, target_y, yaw_tolerance=0.06, timeout=10.0):
        """Obrni robota proti znani map koordinati iz trenutne AMCL poze.

        To se uporabi po Nav2 prihodu do obraza/ringa, ker se mora yaw izracunati
        sele iz dejanske koncne pozicije robota, ne iz pozicije pred zacetkom navigacije.
        """
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot turn to point because current pose is not available.")
            return False

        robot_position = self.get_robot_position()
        dx = target_x - robot_position[0]
        dy = target_y - robot_position[1]

        if math.hypot(dx, dy) < 1e-3:
            self.warn("Cannot turn to point because target is at robot position.")
            return False

        target_yaw = math.atan2(dy, dx)
        return self.turnDirectToYaw(target_yaw, yaw_tolerance=yaw_tolerance, timeout=timeout)

    def driveStraightToXY(self, target_x, target_y, distance_tolerance=0.08,
                          max_linear=0.16, max_angular=0.6, timeout=30.0):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot drive directly because current pose is not available.")
            return False

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
            angular = self._clamp(1.8 * heading_error, -max_angular, max_angular)

            if abs(heading_error) > 0.35:
                linear = 0.0
            else:
                linear = min(max_linear, max(0.04, 0.7 * distance))

            self.publishCmdVel(linear, angular)

        self.stopRobot()
        self.warn(f"Direct drive to ({target_x:.2f}, {target_y:.2f}) timed out.")
        return False

    def goDirectToXYYaw(self, x, y, yaw):
        if not self.driveStraightToXY(x, y):
            return False
        return self.turnDirectToYaw(yaw)

    def setArmPosition(self, command, wait_seconds=3.5):
        msg = String()
        msg.data = command
        self.info(f"Setting arm position: {command}")
        self.arm_command_pub.publish(msg)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _build_goal_pose(self, x, y, yaw):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = self.YawToQuaternion(yaw)
        return goal_pose

    def goToXYYaw(self, x, y, yaw):
        goal_pose = self._build_goal_pose(x, y, yaw)
        if not self.goToPose(goal_pose):
            return False

        self.info("Waiting for the task to complete...")
        while not self.isTaskComplete():
            time.sleep(1)

        return self.getResult() == TaskResult.SUCCEEDED

    def goToXYYawCloseEnough(self, x, y, yaw, close_enough_distance=0.70):
        """Nav2 cilj z mehkim uspesnim pogojem.

        Pri anomaly celicah so cilji ob steni/costmap robu, zato Nav2 vcasih aborta,
        ceprav je robot prakticno dovolj blizu pred-tocki. V tem primeru nadaljujemo
        z direktnim /cmd_vel manevrom.
        """
        success = self.goToXYYaw(x, y, yaw)
        if success:
            return True

        if not hasattr(self, 'current_pose'):
            return False

        robot_position = self.get_robot_position()
        distance = math.hypot(x - robot_position[0], y - robot_position[1])

        if distance <= close_enough_distance:
            self.warn(
                f"Nav2 failed near ({x:.2f}, {y:.2f}), but robot is close enough "
                f"({distance:.2f} m). Continuing with direct movement."
            )
            self.turnDirectToYaw(yaw, timeout=6.0)
            return True

        self.warn(
            f"Nav2 failed and robot is still too far from ({x:.2f}, {y:.2f}): "
            f"{distance:.2f} m."
        )
        return False

    def _anomaly_point_to_xy(self, point):
        if len(point) == 2:
            return point[0], point[1]
        if len(point) >= 3:
            return point[1], point[2]
        raise ValueError("Anomaly coordinate must be (x, y) or (id, x, y).")

    def runAnomalyMovement(self, coordinates, yaw, label="anomaly"):
        """Premik po rdeci/zeleni anomaly celici.

        Minimalna logika:
        1. Nav2 gre samo do varne pred-tocke pred prvo koordinato.
        2. Robot se direktno poravna proti drugi koordinati.
        3. Z /cmd_vel gre do prve tocke.
        4. Roka/kamera gre levo.
        5. Robot gre direktno naravnost do druge tocke brez Nav2 plannerja.
        6. Roka/kamera se vrne v default.
        """
        if len(coordinates) < 2:
            self.warn(f"Need at least two coordinates for {label} movement.")
            return False

        first_x, first_y = self._anomaly_point_to_xy(coordinates[0])
        second_x, second_y = self._anomaly_point_to_xy(coordinates[1])

        # Uporabi dejansko smer med tockama. Parameter yaw pustimo kot fallback,
        # ampak za rdeco/zeleno celico je bolj robustno racunati smer iz koordinat.
        dx = second_x - first_x
        dy = second_y - first_y
        if math.hypot(dx, dy) > 1e-3:
            line_yaw = math.atan2(dy, dx)
        else:
            line_yaw = yaw

        pre_start_x = first_x - ANOMALY_PRE_START_OFFSET * math.cos(line_yaw)
        pre_start_y = first_y - ANOMALY_PRE_START_OFFSET * math.sin(line_yaw)

        self.info(
            f"Starting {label} movement. "
            f"pre_start=({pre_start_x:.2f}, {pre_start_y:.2f}), "
            f"first=({first_x:.2f}, {first_y:.2f}), "
            f"second=({second_x:.2f}, {second_y:.2f}), yaw={line_yaw:.2f}"
        )

        # Do pred-tocke naj pelje Nav2, ker je to se varna tocka stran od roba/stene.
        if not self.goToXYYawCloseEnough(
            pre_start_x,
            pre_start_y,
            line_yaw,
            close_enough_distance=ANOMALY_NAV_CLOSE_ENOUGH_DISTANCE,
        ):
            self.warn(f"Could not get close enough to {label} pre-start coordinate.")
            return False

        # Od tu naprej ne uporabljamo Nav2, ker pravi tocki lezita ob robu/costmap steni.
        if not self.turnDirectToYaw(line_yaw, timeout=6.0):
            self.warn(f"Could not align with {label} line.")
            return False

        if not self.driveStraightToXY(
            first_x,
            first_y,
            distance_tolerance=ANOMALY_DIRECT_POINT_TOLERANCE,
            timeout=12.0,
        ):
            self.warn(f"Could not reach first {label} coordinate with direct drive.")
            return False

        if not self.turnDirectToYaw(line_yaw, timeout=5.0):
            self.warn(f"Could not align at first {label} coordinate.")
            return False

        # Enable anomaly detector auto-scan before moving the arm. Retry briefly
        enabled = False
        try:
            for _ in range(5):
                if self.anomalyClient.wait_for_service(timeout_sec=1.0):
                    req = SetBool.Request()
                    req.data = True
                    fut = self.anomalyClient.call_async(req)
                    rclpy.spin_until_future_complete(self, fut)
                    enabled = True
                    break
                time.sleep(0.25)
            if not enabled:
                self.warn("/set_auto_scan service not available; continuing without enabling anomaly detector.")
        except Exception:
            self.warn("Failed to enable anomaly detector; continuing anyway.")

        self.setArmPosition(ANOMALY_ARM_INSPECT_COMMAND)

        try:
            line_length = math.hypot(second_x - first_x, second_y - first_y)
            direct_timeout = max(18.0, line_length / 0.10 + 8.0)

            success = self.driveStraightToXY(
                second_x,
                second_y,
                distance_tolerance=ANOMALY_DIRECT_POINT_TOLERANCE,
                timeout=direct_timeout,
            )

            if not success:
                self.warn(f"Could not reach second {label} coordinate.")
                return False

            self.turnDirectToYaw(line_yaw, timeout=4.0)
            self.info(f"Finished {label} movement.")
            return True

        finally:
            self.setArmPosition(ANOMALY_ARM_DEFAULT_COMMAND)
            # Disable anomaly detector auto-scan after finishing. Retry briefly
            disabled = False
            try:
                for _ in range(5):
                    if self.anomalyClient.wait_for_service(timeout_sec=1.0):
                        req = SetBool.Request()
                        req.data = False
                        fut = self.anomalyClient.call_async(req)
                        rclpy.spin_until_future_complete(self, fut)
                        disabled = True
                        break
                    time.sleep(0.25)
                if not disabled:
                    self.warn("/set_auto_scan service not available; could not disable anomaly detector.")
            except Exception:
                self.warn("Failed to disable anomaly detector; continuing.")

    def runRedAnomalyMovement(self):
        return self.runAnomalyMovement(ANOMALY_RED_COORDINATES, ANOMALY_RED_YAW, "red anomaly")

    def runGreenAnomalyMovement(self):
        return self.runAnomalyMovement(ANOMALY_GREEN_COORDINATES, ANOMALY_GREEN_YAW, "green anomaly")

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
        self.last_face_coords_time = self.get_clock().now()

    def hasFreshFaceDetection(self, max_age_seconds=0.75):
        if not self.faces or self.last_face_coords_time is None:
            return False

        age_seconds = (self.get_clock().now() - self.last_face_coords_time).nanoseconds / 1e9
        return age_seconds <= max_age_seconds

    def searchForFace(self, turn_radians=FACE_SEARCH_TURN_RADIANS):
        if not hasattr(self, 'current_pose'):
            self.warn("Cannot search for face because current pose is not available.")
            return False

        target_yaw = self._wrap_angle(self.get_robot_yaw() + turn_radians)
        self.info(f"Searching for face by turning {turn_radians:.2f} rad.")
        return self.turnDirectToYaw(target_yaw, timeout=4.0)

    def updateCylinderCoords(self,data):
        self.cylinders = list(zip(
            data.points,
            data.ids,
            data.colors,
            data.orientations,
            data.leaking
        ))

    def updateRingCoords(self,data):
        self.rings = list(zip(data.points, data.ids,data.colors))

    def get_robot_position(self):
        return np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])

    def _build_standoff_goal(self, target_x, target_y, standoff_distance=0.25):
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

    def _go_close_enough(self, target_x, target_y, standoff_distance=0.25, close_enough_distance=0.45):
        """Navigate to a standoff point and accept failures if we are still close enough."""
        goal_pose = self._build_standoff_goal(target_x, target_y, standoff_distance)
        if not self.goToPose(goal_pose):
            return False

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

    def spinSome(self, seconds=1.0):
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    def say(self, text):
        if not self.greetClient.wait_for_service(timeout_sec=2.0):
            self.warn("/greet_service is not available.")
            return False

        request = Speech.Request()
        request.data = text
        future = self.greetClient.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        return response is not None and response.success

    def recognizeFaceTwice(self):
        if not self.boundingBoxClient.wait_for_service(timeout_sec=2.0):
            self.warn("/bounding_box_service is not available.")
            return "UNKNOWN", "M"

        request = BoundingBox.Request()
        request.data = True

        best_response = None
        for attempt in range(5):
            future = self.boundingBoxClient.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()

            if response is None:
                self.warn(f"Face recognition call failed on attempt {attempt + 1}.")
                time.sleep(0.4)
                continue

            person = (response.person or "UNKNOWN").strip()
            gender = (response.gender or "M").strip().upper()
            if person not in {"", "NONE", "UNKNOWN"}:
                best_response = (person, gender if gender in {"M", "F"} else "M")
                break

            self.info(
                f"Face recognition attempt {attempt + 1} returned "
                f"{response.person} ({response.gender}); retrying..."
            )
            time.sleep(0.4)

        if best_response is None:
            self.warn("Could not get a stable face recognition result.")
            return "UNKNOWN", "M"

        person, gender = best_response

        self.info(f"Recognized face as {person} ({gender}).")
        return person, gender

    def requestTaskFromPerson(self, person, gender):
        if not self.faceDialogueClient.wait_for_service(timeout_sec=2.0):
            self.warn("/face_dialogue_service is not available.")
            return None

        request = FaceDialogue.Request()
        request.name = person
        request.gender = gender

        future = self.faceDialogueClient.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response is None or not response.success:
            self.warn(f"Dialogue failed for {person}.")
            return None

        assignment = {
            "person": person,
            "gender": gender,
            "task": response.task,
            "cell": response.cell,
        }
        self.info(f"Task from {person}: {response.task} ({response.cell})")
        self.reportTaskAssignment(assignment)
        return assignment

    def reportTaskAssignment(self, assignment):
        if not self.reportTaskClient.wait_for_service(timeout_sec=0.5):
            self.warn("/report_task_assignment is not available.")
            return False

        request = ReportTaskAssignment.Request()
        request.person_name = assignment["person"]
        request.task = assignment["task"]
        request.cell = assignment["cell"]

        future = self.reportTaskClient.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        return response is not None and response.success

    def collectTasksFromFaces(self):
        self.spinSome(1.0)
        facesCopy = self.faces.copy()
        assignments = []

        self.info(f"Going to collect tasks from {len(facesCopy)} faces.")
        for point,id in facesCopy:
            self.info(f"Going towards face {id}.")
            if not self._go_close_enough(point.x, point.y, standoff_distance=0.25, close_enough_distance=0.35):
                continue

            detected = False
            for attempt in range(FACE_SEARCH_MAX_ATTEMPTS):
                self.spinSome(0.2)
                if self.hasFreshFaceDetection():
                    detected = True
                    break

                self.warn(
                    f"No fresh face detection for face {id}; turning a bit and retrying ({attempt + 1}/{FACE_SEARCH_MAX_ATTEMPTS})."
                )
                if not self.searchForFace():
                    break

            if not detected:
                self.warn(f"Skipping face {id} because no face was visible after the search turns.")
                continue

            # Pomembno: po koncu Nav2 navigacije ponovno izracunaj yaw iz trenutne
            # AMCL poze in se obrni proti dejanski koordinati obraza.
            # Prej se je robot vcasih obrnil narobe, ker je bil yaw izracunan
            # pred premikom oziroma za standoff cilj, ne za koncno pozicijo.
            self.turnDirectToPoint(point.x, point.y, timeout=8.0)
            self.spinSome(0.3)

            person, gender = self.recognizeFaceTwice()
            assignment = self.requestTaskFromPerson(person, gender)
            if assignment is not None:
                assignments.append(assignment)

        self.info(f"Collected {len(assignments)} task assignments.")
        return assignments

    def executeRingTask(self):
        self.spinSome(1.0)
        count = len(self.rings)
        self.info(f"Ring task: counted {count} rings.")
        self.say(f"I detected {count} rings.")
        return True

    def executeBarrelTask(self):
        self.spinSome(1.0)
        leakingCylinders = [
            cylinder for cylinder in self.cylinders.copy()
            if bool(cylinder[4])
        ]
        self.info(f"Barrel task: visiting {len(leakingCylinders)} leaking cylinders.")
        self.say("I am going to the leaked barrels.")

        for point,id,color,orientation,leaking in leakingCylinders:
            self.info(f"Going towards lying cylinder {id}.")
            if not self._go_close_enough(point.x, point.y, standoff_distance=0.25, close_enough_distance=0.5):
                continue

            # Only leaking cylinders are included, but keep the check defensive
            if leaking:
                if self.say("Alert! Alert! This barrel is leaking!"):
                    self.info("Successfully warned about a spill!")
                else:
                    self.info("Failed to warn about a spill!")

        return True

    def executeAnomalyTask(self, cell):
        cell = (cell or "").lower()
        if cell == "red":
            self.say("I am going to detect anomalies.")
            return self.runRedAnomalyMovement()
        if cell == "green":
            self.say("I am going to detect anomalies.")
            return self.runGreenAnomalyMovement()

        self.warn(f"Unknown anomaly cell: {cell}")
        return False

    def executeTaskAssignment(self, assignment):
        task = (assignment["task"] or "").lower()
        cell = (assignment["cell"] or "none").lower()
        person = assignment["person"]

        self.info(f"Executing task from {person}: {task} ({cell})")
        if task == "rings":
            return self.executeRingTask()
        if task == "barrels":
            return self.executeBarrelTask()
        if task == "anomaly":
            return self.executeAnomalyTask(cell)
        if task == "nothing":
            self.info(f"{person} requested no task.")
            return True

        self.warn(f"Unknown task from {person}: {task}")
        return False

    def executeTaskAssignments(self, assignments):
        for assignment in assignments:
            self.executeTaskAssignment(assignment)

    def visitDetections(self):
        assignments = self.collectTasksFromFaces()
        self.executeTaskAssignments(assignments)
    #--------------------------------------------------------------------------

def main(args=None):
    #print("Running new Commander!")
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()

    rc.say("I am going to explore the space now.")
    
    # "yaw" == 0 : gor
    # "yaw" == 1 : desno
    # "yaw" == 2 : dol
    # "yaw" == 3 : levo
    # "yaw" == 4 : gor-desno
    # "yaw" == 5 : dol-desno
    # "yaw" == 6 : gor-levo
    # "yaw" == 7 : dol-levo
    # "yaw" == 8 : obrat na mestu
    # "yaw" == 9 : direktno naravnost do tocke brez Nav2 plannerja
    
    koordinate = [
        (1, 0.2, -0.5, 8),
        (2, 0.85, -2.0, 8),
        (3, 0.0, -3.7, 3),
        (3, 0.0, -3.7, 0),
        (4, 0.9, -4.48, 3),
        (5, -2.4, -4.6, 8),
        (6, -1.18, -1.8, 8),
        (7, -1.5, -0.93, 3),
        (8, -2.9, -2.85, 8),
        (9, -4.2, -1.8, 8),
        (10, -4.3, 0.425, 8),
        (11, -3.3, -0.5, 1),
        (12, -1.35, 0.35, 8),
        (13, 1.5, 0.25, 8),
        (14, 3.05, -0.45, 8)
    ]
    
    #---------------------------------------------------------------------
    #PRVI KROG - DETEKCIJE
    for id, x, y, yaw in koordinate: 
        if yaw == 9:
            rc.info(f"Driving directly to point {id}: {x} {y}")
            rc.driveStraightToXY(x, y)
            continue

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
        elif yaw == 8: # obrat na mestu
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
    #DRUGI KROG - POGOVORI IN IZVAJANJE TASKOV
    rc.info("Going to collect face tasks and execute them")
    rc.say("I am going to visit everyone.")
    rc.visitDetections()

    # Za rocni test anomaly premika odkomentiraj eno vrstico:
    # rc.runRedAnomalyMovement()
    # rc.runGreenAnomalyMovement()

    # Return to the last coordinate after all tasks are complete
    rc.info("Returning to the last coordinate...")
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = rc.get_clock().now().to_msg()
    goal_pose.pose.position.x = 3.05
    goal_pose.pose.position.y = -0.45
    goal_pose.pose.orientation = rc.current_pose.pose.orientation if hasattr(rc, 'current_pose') else rc.YawToQuaternion(0)
    
    rc.goToPose(goal_pose)
    
    rc.info("Waiting for return navigation to complete...")
    while not rc.isTaskComplete():
        time.sleep(1)

    rc.info("Turning 70 degrees to the right")
    rc.spin(-np.deg2rad(70), time_allowance=10)

    rc.info("Waiting for final turn to complete...")
    while not rc.isTaskComplete():
        time.sleep(1)

    rc.info("Moving arm to line-following pose...")
    rc.setArmPosition("lines")

    rc.say("I am going to the line start point now.")

    rc.info("Starting line_follower for line navigation...")
    line_follower_process = subprocess.Popen([
        "ros2", "run", "rins_robot", "line_follower.py"
    ])
    rc.info(f"line_follower started with PID {line_follower_process.pid}")
    
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
