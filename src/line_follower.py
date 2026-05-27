#!/usr/bin/env python3
"""
Line-following node with fork exploration for TurtleBot3/4.

Approach
--------
Instead of dead-reckoning back to a saved checkpoint after exploring a branch,
the robot just keeps line-following continuously. At each fork we remember the
fork position and which branch we took. When we re-encounter a fork we've seen
before (matched by odometry within a tolerance ball), we bias the PID to the
unvisited branch. The line itself guides us through every transition; odometry
is only used as a coarse "have I been here before?" lookup.

Topics
------
  Sub  /gemini/color/image_raw   RGB image for line + fork detection
       /gemini/depth/image_raw   Depth image for obstacle detection
       /odom                     Odometry for fork-memory lookup
  Pub  /cmd_vel                  Twist velocity commands

State machine
-------------
  FOLLOWING  -> PID line follow; fork detection active
  TURNING    -> 180-degree spin-in-place (dead-end or mid-path obstacle)
  DONE       -> All branches at all known forks visited; stop
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from rins_robot.msg import FaceCoords
from rins_robot.srv import GenerateReport

# ---------------------------------------------------------------------------
# Camera / topic configuration
# ---------------------------------------------------------------------------

COLOR_TOPIC   = 'top_camera/rgb/preview/image_raw'
DEPTH_TOPIC   = 'oakd/rgb/preview/depth'
ODOM_TOPIC    = '/odom'
CMD_VEL_TOPIC = '/cmd_vel'

# ---------------------------------------------------------------------------
# Blue line HSV range (tune per lighting / line colour)
# ---------------------------------------------------------------------------
BLUE_HSV_LO = np.array([95,  80,  50],  dtype=np.uint8)
BLUE_HSV_HI = np.array([135, 255, 255], dtype=np.uint8)

# ---------------------------------------------------------------------------
# PID gains
# ---------------------------------------------------------------------------
KP = 0.01
KI = 0.0
KD = 0.003

# ---------------------------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------------------------
LINEAR_SPEED = 0.15   # m/s, forward speed
TURN_SPEED   = 0.8    # rad/s, 180-degree spin speed

# ---------------------------------------------------------------------------
# Image-region parameters (fractions of image height/width)
# ---------------------------------------------------------------------------
CENTROID_STRIP_TOP_FRAC    = 0.75
CENTROID_STRIP_BOTTOM_FRAC = 0.95

# Scanline heights for fork detection (fraction from top of image)
SCANLINE_FRACS = [0.45, 0.60, 0.72]

# Minimum contiguous blue pixels to count as one valid segment
SEGMENT_MIN_WIDTH_PX = 12

# Minimum blue pixels in centroid strip to consider the line found
MIN_LINE_PIXELS = 100

# ---------------------------------------------------------------------------
# Obstacle / depth parameters
# ---------------------------------------------------------------------------
OBSTACLE_DIST_M    = 0.4   # meters; triggers end-of-branch / mid-path turn
DEPTH_STRIP_Y_FRAC = 0.20
DEPTH_STRIP_X_FRAC = 0.30
DEPTH_MIN_VALID_M  = 0.05
DEPTH_MAX_VALID_M  = 2.0

# ---------------------------------------------------------------------------
# Turn tolerance
# ---------------------------------------------------------------------------
TURN_YAW_TOL = 0.06   # radians (~3.5°); done spinning

# ---------------------------------------------------------------------------
# Fork memory parameters
# ---------------------------------------------------------------------------
# Two fork detections within this distance are considered the same fork.
FORK_MATCH_RADIUS_M = 0.5

# After committing to a branch at a fork, suppress fork re-detection until:
#   (a) we have moved at least this far from the fork, AND
#   (b) the fork detector has returned None for this many consecutive frames.
FORK_CLEAR_DIST_M       = 0.5
FORK_CLEAR_FRAME_COUNT  = 5

# How long to bias the PID toward the chosen branch when committing.
# Acts as a safety net; the bias also drops when we leave the fork zone.
BRANCH_BIAS_DURATION_S = 1.5

# ---------------------------------------------------------------------------


class State(Enum):
    FOLLOWING = auto()
    TURNING   = auto()
    DONE      = auto()


@dataclass
class ForkMemory:
    """A fork we have encountered, with a count of how many times we've passed it."""
    x: float
    y: float
    visits: int = 0  # incremented each time we commit to a branch here

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


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
    return (a - b + math.pi) % (2 * math.pi) - math.pi


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
        self.create_subscription(
            FaceCoords, '/face_coords', self._face_coords_cb, 10)

        self._cmd_pub = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        self._report_client = self.create_client(GenerateReport, '/generate_report')

        # Fast timer for non-vision states (TURNING)
        self.create_timer(0.05, self._control_timer_cb)  # 20 Hz

        # State
        self._state: State = State.FOLLOWING
        self._new_face_detected = False
        self._shutdown_requested = False
        self._report_future = None

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

        # Fork memory: list of forks we've encountered
        self._forks: List[ForkMemory] = []

        # Active branch commitment (always biases to the left half of the image)
        self._branch_bias_active = False
        self._branch_bias_end_time = None
        self._active_fork_pos: Optional[Tuple[float, float]] = None

        # Fork suppression: don't re-detect a fork we're currently inside
        self._fork_clear_counter = 0
        self._suppress_fork_detection = False

        self.get_logger().info('LineFollower started — state: FOLLOWING')

    # -----------------------------------------------------------------------
    # Odometry
    # -----------------------------------------------------------------------

    def _face_coords_cb(self, msg: FaceCoords):
        if msg.ids:
            self._new_face_detected = True

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._pos_x = p.x
        self._pos_y = p.y
        self._yaw   = _yaw_from_quat(msg.pose.pose.orientation)

    def _control_timer_cb(self):
        """Runs at 20 Hz independently of camera. Handles non-vision states."""
        if self._shutdown_requested:
            if self._report_future is None:
                if self._report_client.service_is_ready():
                    req = GenerateReport.Request()
                    self._report_future = self._report_client.call_async(req)
                    self.get_logger().info('Calling /generate_report …')
                else:
                    self.get_logger().warn('/generate_report not available — skipping report')
                    raise SystemExit
            elif self._report_future.done():
                result = self._report_future.result()
                if result.success:
                    self.get_logger().info(f'Report saved: {result.pdf_path}')
                else:
                    self.get_logger().error(f'Report failed: {result.message}')
                raise SystemExit
            return
        if self._state == State.TURNING:
            self._do_turning()
        elif self._state == State.DONE:
            self._stop()

    # -----------------------------------------------------------------------
    # Main camera callback
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

        if self._state == State.FOLLOWING:
            self._do_following(blue, depth, w, h)

        self._debug_display(bgr, blue, depth, w, h)

    # -----------------------------------------------------------------------
    # State: FOLLOWING
    # -----------------------------------------------------------------------

    def _do_following(self, blue, depth, w, h):
        # Update branch-bias expiry & fork-suppression state
        self._update_branch_bias()
        self._update_fork_suppression(blue, h, w)

        # Obstacle ahead → check for face first, else 180° turn
        if self._obstacle_ahead(depth, h, w):
            if self._new_face_detected:
                self.get_logger().info('Face detected at end of segment → generating report and shutting down')
                self._stop()
                self._state = State.DONE
                self._shutdown_requested = True
                return
            self.get_logger().info('Obstacle / dead end → 180° turn')
            self._start_turn()
            return

        # Fork detection (only when not currently inside a fork)
        if not self._suppress_fork_detection:
            if self._detect_fork(blue, h, w):
                if self._handle_fork():
                    # _handle_fork returns True if we should stop (DONE)
                    return

        # PID line follow (uses branch bias if active)
        ang = self._pid(blue, w, h)
        self._publish(LINEAR_SPEED, ang)

    # -----------------------------------------------------------------------
    # Fork handling
    # -----------------------------------------------------------------------

    def _handle_fork(self) -> bool:
        """
        Commit to the left branch at the current fork (always).

        Rationale: at the first visit we go left, leaving the right branch
        unexplored. After traversing left + dead-end + 180° + returning, the
        previously-right branch is now on the robot's left side. So "always
        left" naturally alternates between branches across visits.

        Returns True if this fork has already been fully explored (third
        visit → all branches done, parent path also done → DONE), else False.
        """
        matched = self._find_nearby_fork(self._pos_x, self._pos_y)

        if matched is None:
            fork = ForkMemory(x=self._pos_x, y=self._pos_y, visits=1)
            self._forks.append(fork)
            self.get_logger().info(
                f'New fork #{len(self._forks)} at '
                f'({self._pos_x:.2f}, {self._pos_y:.2f}) → going LEFT (visit 1)')
        else:
            matched.visits += 1
            if matched.visits >= 3:
                # Visit 1: went left into branch A
                # Visit 2: returned from A, went left into branch B (was right)
                # Visit 3: returned from B — both branches done. Stop.
                self.get_logger().info(
                    f'Fork at ({matched.x:.2f}, {matched.y:.2f}) fully '
                    f'explored (visit {matched.visits}) → DONE')
                self._state = State.DONE
                self._stop()
                return True
            self.get_logger().info(
                f'Known fork at ({matched.x:.2f}, {matched.y:.2f}) → '
                f'going LEFT (visit {matched.visits})')

        # Commit: bias PID toward the left branch for a short window.
        self._branch_bias_active = True
        self._branch_bias_end_time = (
            self.get_clock().now()
            + rclpy.duration.Duration(seconds=BRANCH_BIAS_DURATION_S))
        self._active_fork_pos = (self._pos_x, self._pos_y)
        self._suppress_fork_detection = True
        self._fork_clear_counter = 0
        return False

    def _find_nearby_fork(self, x: float, y: float) -> Optional[ForkMemory]:
        best: Optional[ForkMemory] = None
        best_d = FORK_MATCH_RADIUS_M
        for f in self._forks:
            d = f.distance_to(x, y)
            if d <= best_d:
                best = f
                best_d = d
        return best

    def _update_branch_bias(self):
        """Drop the branch bias once the timer expires."""
        if not self._branch_bias_active:
            return
        if self.get_clock().now() >= self._branch_bias_end_time:
            self.get_logger().info('Branch bias expired')
            self._branch_bias_active = False
            self._branch_bias_end_time = None

    def _update_fork_suppression(self, blue, h, w):
        """
        Lift fork-detection suppression once:
          (a) we are far enough from the fork we just entered, AND
          (b) the fork detector has been quiet for several consecutive frames.
        """
        if not self._suppress_fork_detection:
            return

        # (a) Distance check
        if self._active_fork_pos is not None:
            dx = self._pos_x - self._active_fork_pos[0]
            dy = self._pos_y - self._active_fork_pos[1]
            if math.hypot(dx, dy) < FORK_CLEAR_DIST_M:
                self._fork_clear_counter = 0
                return

        # (b) Detector quiet check
        if not self._detect_fork(blue, h, w):
            self._fork_clear_counter += 1
        else:
            self._fork_clear_counter = 0

        if self._fork_clear_counter >= FORK_CLEAR_FRAME_COUNT:
            self._suppress_fork_detection = False
            self._active_fork_pos = None
            self._fork_clear_counter = 0

    def _detect_fork(self, blue: np.ndarray, h: int, w: int) -> bool:
        """
        Detect a fork/junction, direction-agnostic.

        Cast scanlines from top (far) to bottom (near). A fork is any pattern
        where segment count is non-decreasing top→bottom, with at least one
        scanline showing 1 segment and at least one showing 2. This catches:
          - Outbound (trunk → branches): top=1, bottom=2  (lines diverge)
          - Inbound  (branch → trunk):   top=1, bottom=2  (lines converge
            toward the robot, so still 2 nearby and 1 far away)

        Any scanline with 0 or >2 segments invalidates the detection.
        """
        counts: List[int] = []
        for frac in SCANLINE_FRACS:
            row = blue[int(h * frac), :].astype(bool)
            segs = _segments_in_row(row, SEGMENT_MIN_WIDTH_PX)
            n = len(segs)
            if n == 0 or n > 2:
                return False
            counts.append(n)

        # Must be monotonically non-decreasing top→bottom
        for i in range(1, len(counts)):
            if counts[i] < counts[i - 1]:
                return False

        # Must include both a 1-segment and a 2-segment row
        # (rules out a plain trunk [1,1,1] or being fully inside the fork [2,2,2])
        return 1 in counts and 2 in counts

    # -----------------------------------------------------------------------
    # PID (with optional branch bias)
    # -----------------------------------------------------------------------

    def _pid(self, blue: np.ndarray, w: int, h: int) -> float:
        """
        Compute angular.z from centroid error in the bottom strip.
        If branch bias is active, zero out the right half of the strip so the
        centroid latches onto the left branch through the fork.
        """
        strip_top = int(h * CENTROID_STRIP_TOP_FRAC)
        strip_bot = int(h * CENTROID_STRIP_BOTTOM_FRAC)
        strip = blue[strip_top:strip_bot, :].copy()

        if self._branch_bias_active:
            strip[:, w // 2:] = 0

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

    def _reset_pid(self):
        self._pid_error_prev = 0.0
        self._pid_integral   = 0.0
        self._pid_time_prev  = self.get_clock().now()

    # -----------------------------------------------------------------------
    # Obstacle detection
    # -----------------------------------------------------------------------

    def _obstacle_ahead(self, depth: np.ndarray, h: int, w: int) -> bool:
        y0 = int(h * (0.5 - DEPTH_STRIP_Y_FRAC / 2))
        y1 = int(h * (0.5 + DEPTH_STRIP_Y_FRAC / 2))
        x0 = int(w * (0.5 - DEPTH_STRIP_X_FRAC / 2))
        x1 = int(w * (0.5 + DEPTH_STRIP_X_FRAC / 2))
        region = depth[y0:y1, x0:x1]
        valid = region[(region > DEPTH_MIN_VALID_M) &
                       (region < DEPTH_MAX_VALID_M) &
                       np.isfinite(region)]
        if valid.size == 0:
            return False
        median_m = float(np.median(valid))
        self.get_logger().info(
            f'Depth median: {median_m:.2f}m', throttle_duration_sec=1.0)
        return median_m < OBSTACLE_DIST_M

    # -----------------------------------------------------------------------
    # State: TURNING (180° spin in place)
    # -----------------------------------------------------------------------

    def _start_turn(self):
        self._turn_target_yaw = _wrap(self._yaw + math.pi)
        self._state = State.TURNING
        self._reset_pid()

    def _do_turning(self):
        if self._turn_target_yaw is None:
            return
        diff = _angle_diff(self._turn_target_yaw, self._yaw)
        if abs(diff) < TURN_YAW_TOL:
            self.get_logger().info('Turn done → FOLLOWING')
            self._stop()
            self._state = State.FOLLOWING
            self._turn_target_yaw = None
            self._reset_pid()
        else:
            self._publish(0.0, math.copysign(TURN_SPEED, diff))

    # -----------------------------------------------------------------------
    # Publishing helpers
    # -----------------------------------------------------------------------

    def _publish(self, linear: float, angular: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x  = float(linear)
        msg.twist.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def _stop(self):
        self._publish(0.0, 0.0)

    # -----------------------------------------------------------------------
    # Debug visualisation
    # -----------------------------------------------------------------------

    def _debug_display(self, bgr: np.ndarray, blue: np.ndarray,
                       depth: np.ndarray, w: int, h: int):
        try:
            vis = bgr.copy()

            # Centroid strip
            y0 = int(h * CENTROID_STRIP_TOP_FRAC)
            y1 = int(h * CENTROID_STRIP_BOTTOM_FRAC)
            cv2.rectangle(vis, (0, y0), (w, y1), (0, 255, 255), 1)

            # If branch bias is active, shade the suppressed (right) half
            if self._branch_bias_active:
                cv2.rectangle(vis, (w // 2, y0), (w, y1), (50, 50, 50), -1)

            # Scanlines
            for frac in SCANLINE_FRACS:
                ry = int(h * frac)
                cv2.line(vis, (0, ry), (w, ry), (255, 100, 0), 1)

            # Obstacle detection region
            dy0 = int(h * (0.5 - DEPTH_STRIP_Y_FRAC / 2))
            dy1 = int(h * (0.5 + DEPTH_STRIP_Y_FRAC / 2))
            dx0 = int(w * (0.5 - DEPTH_STRIP_X_FRAC / 2))
            dx1 = int(w * (0.5 + DEPTH_STRIP_X_FRAC / 2))
            cv2.rectangle(vis, (dx0, dy0), (dx1, dy1), (0, 0, 255), 1)

            # State label + fork count + bias
            label = f'{self._state.name}  forks={len(self._forks)}'
            if self._branch_bias_active:
                label += '  bias=LEFT'
            if self._suppress_fork_detection:
                label += '  [fork-suppress]'
            cv2.putText(vis, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Blue mask overlay
            blue_rgb = cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)
            blue_rgb[:, :, 0] = 0
            blue_rgb[:, :, 2] = 0
            vis = cv2.addWeighted(vis, 0.7, blue_rgb, 0.3, 0)

            cv2.imshow('LineFollower', vis)

            # Depth window
            depth_vis = np.zeros_like(depth)
            valid_mask = (np.isfinite(depth) &
                          (depth > DEPTH_MIN_VALID_M) &
                          (depth < DEPTH_MAX_VALID_M))
            if valid_mask.any():
                d_min = float(depth[valid_mask].min())
                d_max = float(depth[valid_mask].max())
                if d_max > d_min:
                    depth_vis[valid_mask] = (
                        (depth[valid_mask] - d_min) / (d_max - d_min) * 255)
            depth_color = cv2.applyColorMap(
                depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.rectangle(depth_color, (dx0, dy0), (dx1, dy1), (255, 255, 255), 2)
            region = depth[dy0:dy1, dx0:dx1]
            valid_reg = region[np.isfinite(region) &
                               (region > DEPTH_MIN_VALID_M) &
                               (region < DEPTH_MAX_VALID_M)]
            if valid_reg.size > 0:
                med = float(np.median(valid_reg))
                triggered = med < OBSTACLE_DIST_M
                col = (0, 0, 255) if triggered else (0, 255, 0)
                cv2.putText(depth_color,
                            f'{med:.2f}m {"OBSTACLE" if triggered else "clear"}',
                            (dx0, max(dy0 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            cv2.imshow('Depth', depth_color)

            cv2.waitKey(1)
        except Exception:
            pass  # Headless environments


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