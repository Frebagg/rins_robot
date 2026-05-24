#!/usr/bin/env python3
"""
Line-following node with fork exploration for TurtleBot3/4.

Uses the forward-facing Gemini camera (assumed floor-facing if a dedicated
downward camera is not wired up in the URDF/topics yet).

Topics
------
  Sub  /gemini/color/image_raw   RGB image for line + fork detection
       /gemini/depth/image_raw   Depth image for obstacle detection
       /odom                     Odometry for checkpoint navigation
  Pub  /cmd_vel                  Twist velocity commands

State machine
-------------
  FOLLOWING        -> PID line follow; fork detection active
  BRANCH_TRAVERSAL -> PID line follow on left branch; fork detection OFF
  TURNING          -> 180-degree spin-in-place
  RETURNING        -> Odometry-based drive back to saved checkpoint
  REALIGNING       -> Turn to face the main-path direction at the checkpoint
  DONE             -> All branches visited; stop
"""

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# ---------------------------------------------------------------------------
# Camera / topic configuration
# ---------------------------------------------------------------------------

# Change these if a dedicated floor camera is available.
COLOR_TOPIC = 'top_camera/rgb/preview/image_raw'
DEPTH_TOPIC = 'top_camera/rgb/preview/depth'
ODOM_TOPIC  = '/odom'
CMD_VEL_TOPIC = '/cmd_vel'

# ---------------------------------------------------------------------------
# Blue line HSV range (tune per lighting / line colour)
# ---------------------------------------------------------------------------
BLUE_HSV_LO = np.array([95,  80,  50],  dtype=np.uint8)
BLUE_HSV_HI = np.array([135, 255, 255], dtype=np.uint8)

# ---------------------------------------------------------------------------
# PID gains (tune on the real robot)
# ---------------------------------------------------------------------------
KP = 0.004
KI = 0.0
KD = 0.0015

# ---------------------------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------------------------
LINEAR_SPEED         = 0.15   # m/s, forward speed
TURN_SPEED           = 0.8    # rad/s, 180-degree spin speed
RETURN_ANGULAR_SPEED = 0.5    # rad/s, heading correction while returning
RETURN_LINEAR_SPEED  = 0.12   # m/s, forward speed while returning

# ---------------------------------------------------------------------------
# Image-region parameters (fractions of image height/width)
# ---------------------------------------------------------------------------
# Strip for centroid: measured from the top of the image
CENTROID_STRIP_TOP_FRAC    = 0.75   # 75 % down from top
CENTROID_STRIP_BOTTOM_FRAC = 0.95   # 95 % down from top

# Scanline heights for fork detection (fraction from top of image)
SCANLINE_FRACS = [0.45, 0.60, 0.72]

# Minimum contiguous blue pixels to count as one valid segment
SEGMENT_MIN_WIDTH_PX = 12

# ---------------------------------------------------------------------------
# Obstacle / depth parameters
# ---------------------------------------------------------------------------
OBSTACLE_DIST_M     = 1.0   # meters; triggers end-of-branch / mid-path turn
DEPTH_STRIP_Y_FRAC  = 0.20  # height of centre strip used for depth sampling
DEPTH_STRIP_X_FRAC  = 0.30  # width of centre strip used for depth sampling
DEPTH_MIN_VALID_MM  = 50    # ignore depth readings below this (noise)
DEPTH_MAX_VALID_MM  = 8000  # ignore depth readings above this (out of range)

# ---------------------------------------------------------------------------
# Navigation tolerances
# ---------------------------------------------------------------------------
CHECKPOINT_DIST_TOL = 0.15  # metres; "close enough" to checkpoint
TURN_YAW_TOL        = 0.06  # radians (~3.5°); done spinning

# Cooldown between fork detections (seconds) to prevent re-triggering
FORK_COOLDOWN_S = 3.0

# Approximate horizontal field-of-view for mapping pixel offset → angle
CAMERA_HFOV_DEG = 60.0

# Minimum blue pixels in centroid strip to consider the line found
MIN_LINE_PIXELS = 100

# ---------------------------------------------------------------------------


class State(Enum):
    FOLLOWING        = auto()
    BRANCH_TRAVERSAL = auto()
    TURNING          = auto()
    RETURNING        = auto()
    REALIGNING       = auto()
    DONE             = auto()


@dataclass
class Checkpoint:
    x: float
    y: float
    yaw: float           # heading of the robot when fork was first detected
    left_cx: float       # x-pixel centre of the left (branch) segment
    right_cx: float      # x-pixel centre of the right (main-path) segment
    img_width: int


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _segments_in_row(row: np.ndarray, min_width: int) -> List[Tuple[int, int]]:
    """Return (start, end) x-indices for every True run of width >= min_width."""
    segments: List[Tuple[int, int]] = []
    n = len(row)
    i = 0
    while i < n:
        if row[i]:
            start = i
            while i < n and row[i]:
                i += 1
            if (i - start) >= min_width:
                segments.append((start, i - 1))
        else:
            i += 1
    return segments


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_diff(a: float, b: float) -> float:
    """Signed difference (a - b) normalised to (-π, π]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LineFollower(Node):

    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()

        # Subscribers (colour + depth synchronised)
        color_sub = message_filters.Subscriber(
            self, Image, COLOR_TOPIC, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, DEPTH_TOPIC, qos_profile=qos_profile_sensor_data)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=6, slop=0.12)
        self._sync.registerCallback(self._camera_cb)

        self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_cb, qos_profile_sensor_data)

        # Publisher
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        # State
        self._state: State = State.FOLLOWING
        self._checkpoint_stack: deque[Checkpoint] = deque()

        # PID
        self._pid_error_prev = 0.0
        self._pid_integral   = 0.0
        self._pid_time_prev  = self.get_clock().now()

        # Odometry
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._yaw   = 0.0

        # Turn management
        self._turn_target_yaw: Optional[float] = None
        self._turn_next_state: State = State.FOLLOWING

        # Fork detection cooldown
        self._last_fork_ns = 0

        self.get_logger().info('LineFollower started — state: FOLLOWING')

    # -----------------------------------------------------------------------
    # Odometry
    # -----------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pos_x = p.x
        self._pos_y = p.y
        self._yaw   = _yaw_from_quat(msg.pose.pose.orientation)

    # -----------------------------------------------------------------------
    # Main camera callback — dispatches per state
    # -----------------------------------------------------------------------

    def _camera_cb(self, color_msg: Image, depth_msg: Image):
        try:
            bgr   = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough').astype(np.float32)
        except CvBridgeError as e:
            self.get_logger().warning(f'CvBridge: {e}')
            return

        h, w = bgr.shape[:2]
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, BLUE_HSV_LO, BLUE_HSV_HI)

        s = self._state
        if s == State.FOLLOWING:
            self._do_following(blue, depth, w, h)
        elif s == State.BRANCH_TRAVERSAL:
            self._do_branch_traversal(blue, depth, w, h)
        elif s == State.TURNING:
            self._do_turning()
        elif s == State.RETURNING:
            self._do_returning()
        elif s == State.REALIGNING:
            self._do_realigning(blue, w, h)
        elif s == State.DONE:
            self._stop()

        self._debug_display(bgr, blue, w, h)

    # -----------------------------------------------------------------------
    # State: FOLLOWING
    # -----------------------------------------------------------------------

    def _do_following(self, blue, depth, w, h):
        # Mid-path obstacle → 180° turn, keep following
        if self._obstacle_ahead(depth, h, w):
            self.get_logger().info('Obstacle mid-path → 180° turn')
            self._start_turn(next_state=State.FOLLOWING)
            return

        # Fork detection (with cooldown)
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_fork_ns) > int(FORK_COOLDOWN_S * 1e9):
            fork = self._detect_fork(blue, h, w)
            if fork is not None:
                self._last_fork_ns = now_ns
                self._handle_fork(fork, w)
                return

        # PID line follow
        ang = self._pid(blue, w, h)
        self._publish(LINEAR_SPEED, ang)

    # -----------------------------------------------------------------------
    # State: BRANCH_TRAVERSAL
    # -----------------------------------------------------------------------

    def _do_branch_traversal(self, blue, depth, w, h):
        # End of branch: obstacle within threshold
        if self._obstacle_ahead(depth, h, w):
            self.get_logger().info('End of branch → 180° turn, then RETURNING')
            self._start_turn(next_state=State.RETURNING)
            return

        ang = self._pid(blue, w, h)
        self._publish(LINEAR_SPEED, ang)

    # -----------------------------------------------------------------------
    # State: TURNING (spin 180° in place)
    # -----------------------------------------------------------------------

    def _do_turning(self):
        if self._turn_target_yaw is None:
            return
        diff = _angle_diff(self._turn_target_yaw, self._yaw)
        if abs(diff) < TURN_YAW_TOL:
            self.get_logger().info(
                f'Turn done → {self._turn_next_state.name}')
            self._stop()
            self._state = self._turn_next_state
            self._turn_target_yaw = None
            self._reset_pid()
        else:
            # Always spin CCW (positive angular.z)
            self._publish(0.0, math.copysign(TURN_SPEED, diff))

    # -----------------------------------------------------------------------
    # State: RETURNING (drive toward checkpoint using odometry)
    # -----------------------------------------------------------------------

    def _do_returning(self):
        if not self._checkpoint_stack:
            self.get_logger().info('Checkpoint stack empty → DONE')
            self._state = State.DONE
            self._stop()
            return

        cp = self._checkpoint_stack[-1]
        dx = cp.x - self._pos_x
        dy = cp.y - self._pos_y
        dist = math.hypot(dx, dy)

        if dist < CHECKPOINT_DIST_TOL:
            self.get_logger().info('Checkpoint reached → REALIGNING')
            self._state = State.REALIGNING
            self._stop()
            return

        # Turn toward checkpoint, then drive
        target_heading = math.atan2(dy, dx)
        yaw_err = _angle_diff(target_heading, self._yaw)

        if abs(yaw_err) > 0.2:
            # Large heading error: rotate in place
            self._publish(0.0, math.copysign(RETURN_ANGULAR_SPEED, yaw_err))
        else:
            # Small heading error: proportional correction while driving
            correction = RETURN_ANGULAR_SPEED * (yaw_err / 0.2)
            self._publish(RETURN_LINEAR_SPEED, correction)

    # -----------------------------------------------------------------------
    # State: REALIGNING (turn to face the main path, then switch to FOLLOWING)
    # -----------------------------------------------------------------------

    def _do_realigning(self, blue, w, h):
        if not self._checkpoint_stack:
            self._state = State.DONE
            self._stop()
            return

        cp = self._checkpoint_stack[-1]

        # Map pixel position of right (main-path) segment to a world heading.
        # right_cx > img_width/2 → main path is to the right of robot axis.
        img_cx    = cp.img_width / 2.0
        dx_pix    = cp.right_cx - img_cx
        fov_rad   = math.radians(CAMERA_HFOV_DEG)
        pix_angle = (dx_pix / cp.img_width) * fov_rad   # angular offset in camera
        target_yaw = _wrap(cp.yaw + pix_angle)

        yaw_err = _angle_diff(target_yaw, self._yaw)

        if abs(yaw_err) > TURN_YAW_TOL * 1.5:
            self._publish(0.0, math.copysign(RETURN_ANGULAR_SPEED, yaw_err))
            return

        # Facing the right direction — check if the line is in view
        strip_top = int(h * CENTROID_STRIP_TOP_FRAC)
        strip_bot = int(h * CENTROID_STRIP_BOTTOM_FRAC)
        strip = blue[strip_top:strip_bot, :]
        M = cv2.moments(strip)
        if M['m00'] >= MIN_LINE_PIXELS:
            self.get_logger().info('Main line reacquired → FOLLOWING')
            self._checkpoint_stack.pop()
            self._last_fork_ns = self.get_clock().now().nanoseconds
            self._state = State.FOLLOWING
            self._reset_pid()
        else:
            # Creep forward slowly until the line appears
            self._publish(RETURN_LINEAR_SPEED * 0.5, 0.0)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _pid(self, blue: np.ndarray, w: int, h: int) -> float:
        """Compute angular.z from centroid error in the bottom strip."""
        strip_top = int(h * CENTROID_STRIP_TOP_FRAC)
        strip_bot = int(h * CENTROID_STRIP_BOTTOM_FRAC)
        strip = blue[strip_top:strip_bot, :]
        M = cv2.moments(strip)
        if M['m00'] < MIN_LINE_PIXELS:
            # Line lost: hold last correction (P-only fallback)
            return float(-KP * self._pid_error_prev * 80)

        cx    = M['m10'] / M['m00']
        error = cx - w / 2.0

        now = self.get_clock().now()
        dt  = max((now - self._pid_time_prev).nanoseconds * 1e-9, 1e-4)
        self._pid_integral  += error * dt
        derivative           = (error - self._pid_error_prev) / dt
        self._pid_error_prev = error
        self._pid_time_prev  = now

        raw = -(KP * error + KI * self._pid_integral + KD * derivative)
        return float(np.clip(raw, -1.5, 1.5))

    def _detect_fork(self, blue: np.ndarray, h: int, w: int) \
            -> Optional[List[Tuple[float, float]]]:
        """
        Cast 3 scanlines and check for exactly 2 blue segments on each.
        Returns [(left_cx, right_cx), ...] for each scanline if fork confirmed,
        else None.
        """
        results: List[Tuple[float, float]] = []
        for frac in SCANLINE_FRACS:
            row = blue[int(h * frac), :].astype(bool)
            segs = _segments_in_row(row, SEGMENT_MIN_WIDTH_PX)
            if len(segs) != 2:
                return None
            left_cx  = (segs[0][0] + segs[0][1]) / 2.0
            right_cx = (segs[1][0] + segs[1][1]) / 2.0
            results.append((left_cx, right_cx))
        return results

    def _handle_fork(self, fork_segs: List[Tuple[float, float]], w: int):
        """Save checkpoint and begin branch traversal on the left branch."""
        left_cx, right_cx = fork_segs[0]   # topmost scanline gives best direction
        cp = Checkpoint(
            x=self._pos_x,
            y=self._pos_y,
            yaw=self._yaw,
            left_cx=left_cx,
            right_cx=right_cx,
            img_width=w,
        )
        self._checkpoint_stack.append(cp)
        self.get_logger().info(
            f'Fork → checkpoint ({cp.x:.2f}, {cp.y:.2f}) yaw={math.degrees(cp.yaw):.1f}°  '
            f'left_cx={left_cx:.0f}  right_cx={right_cx:.0f}')
        self._state = State.BRANCH_TRAVERSAL
        self._reset_pid()

    def _obstacle_ahead(self, depth: np.ndarray, h: int, w: int) -> bool:
        """True if median depth in the centre rectangle is below threshold."""
        y0 = int(h * (0.5 - DEPTH_STRIP_Y_FRAC / 2))
        y1 = int(h * (0.5 + DEPTH_STRIP_Y_FRAC / 2))
        x0 = int(w * (0.5 - DEPTH_STRIP_X_FRAC / 2))
        x1 = int(w * (0.5 + DEPTH_STRIP_X_FRAC / 2))
        region = depth[y0:y1, x0:x1]
        valid  = region[(region > DEPTH_MIN_VALID_MM) & (region < DEPTH_MAX_VALID_MM)]
        if valid.size == 0:
            return False
        median_m = float(np.median(valid)) / 1000.0
        return median_m < OBSTACLE_DIST_M

    def _start_turn(self, next_state: State):
        """Begin a 180-degree CCW spin."""
        self._turn_target_yaw = _wrap(self._yaw + math.pi)
        self._turn_next_state = next_state
        self._state = State.TURNING
        self._reset_pid()

    def _reset_pid(self):
        self._pid_error_prev = 0.0
        self._pid_integral   = 0.0
        self._pid_time_prev  = self.get_clock().now()

    def _publish(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _debug_display(self, bgr: np.ndarray, blue: np.ndarray, w: int, h: int):
        """Draw debug overlay and show in a window (disable if headless)."""
        try:
            vis = bgr.copy()

            # Draw centroid strip
            y0 = int(h * CENTROID_STRIP_TOP_FRAC)
            y1 = int(h * CENTROID_STRIP_BOTTOM_FRAC)
            cv2.rectangle(vis, (0, y0), (w, y1), (0, 255, 255), 1)

            # Draw scanlines
            for frac in SCANLINE_FRACS:
                ry = int(h * frac)
                cv2.line(vis, (0, ry), (w, ry), (255, 100, 0), 1)

            # State label
            cv2.putText(vis, self._state.name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Blue mask overlay (semi-transparent)
            blue_rgb = cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)
            blue_rgb[:, :, 0] = 0
            blue_rgb[:, :, 2] = 0
            vis = cv2.addWeighted(vis, 0.7, blue_rgb, 0.3, 0)

            cv2.imshow('LineFollower', vis)
            cv2.waitKey(1)
        except Exception:
            pass   # Suppress display errors in headless environments


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
