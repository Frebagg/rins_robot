#!/usr/bin/env python3
"""
ROS2 node: anomaly_detector

Loads a trained PatchCore model exported by anomalib and continuously
scans the camera feed.  When a tile appears anywhere in the frame and
stays stable for a few frames, the node automatically:
  1. de-rotates the tile to upright via minAreaRect + perspective warp,
  2. runs anomalib inference,
  3. forwards results + images to /report_anomaly_tile,
  4. waits until the tile leaves the frame before inspecting again.

The /inspect_tile service is still available for manual triggering.

Parameters (ROS2):
    camera_topic       (default: /top_camera/rgb/preview/image_raw)
    model_path         (default: <package_share>/models/anomaly_model.pt)
    show_debug         (default: True)
    auto_scan          (default: True)
    stable_frames      (default: 8)    frames a tile must be stable before
                                       auto-inspection fires
    stable_iou         (default: 0.85) IoU threshold for "same tile"
    reacquire_misses   (default: 6)    frames without a tile before the
                                       node is ready to inspect the next one
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from rins_robot.srv import InspectTile, ReportAnomalyTile


class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__('anomaly_detector')

        self.declare_parameters(namespace='', parameters=[
            ('camera_topic', '/top_camera/rgb/preview/image_raw'),
            ('model_path', ''),
            ('show_debug', True),
            ('auto_scan', True),
            ('stable_frames', 8),
            ('stable_iou', 0.85),
            ('reacquire_misses', 6),
        ])

        self.camera_topic     = self.get_parameter('camera_topic').value
        self.model_path       = str(self.get_parameter('model_path').value)
        self.show_debug       = bool(self.get_parameter('show_debug').value)
        self.auto_scan        = bool(self.get_parameter('auto_scan').value)
        self.stable_frames    = int(self.get_parameter('stable_frames').value)
        self.stable_iou       = float(self.get_parameter('stable_iou').value)
        self.reacquire_misses = int(self.get_parameter('reacquire_misses').value)

        if not self.model_path:
            self.model_path = self._default_model_path()

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.inferencer = None

        # ── auto-scan state ───────────────────────────────────────────────────
        # tracked_box: rotated rect of the currently-tracked tile, or None
        # ((cx,cy), (w,h), angle)
        self._tracked_box: Optional[tuple] = None
        self._tracked_aabb: Optional[tuple] = None  # axis-aligned for IoU
        self._stable_count = 0
        self._miss_count   = 0
        self._auto_tile_id = 0
        self._awaiting_clear = False  # True after inspect, until tile leaves

        self._load_model()

        self.create_subscription(Image, self.camera_topic, self._camera_cb,
                                 qos_profile_sensor_data)
        self.create_service(InspectTile, '/inspect_tile',
                            self._inspect_tile_cb)
        self._report_client = self.create_client(ReportAnomalyTile,
                                                 '/report_anomaly_tile')

        self.get_logger().info(
            f'Anomaly detector ready  camera={self.camera_topic}  '
            f'auto_scan={self.auto_scan}')

    # ── setup ──────────────────────────────────────────────────────────────────

    def _default_model_path(self) -> str:
        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('rins_robot')
            candidates.append(
                os.path.join(share, 'models', 'anomaly_model.pt'))
        except Exception:
            pass
        candidates.append(
            os.path.join(os.path.dirname(__file__),
                         '..', 'models', 'anomaly_model.pt'))
        for p in candidates:
            if os.path.exists(p):
                return os.path.abspath(p)
        return candidates[0] if candidates else ''

    def _load_model(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            self.get_logger().error(
                f'Model not found at {self.model_path}. '
                'Train first: python3 tools/train_anomaly_detector.py '
                '--dataset <path> --output models/anomaly_model.pt')
            return

        try:
            from anomalib.deploy import TorchInferencer
        except ImportError:
            self.get_logger().error(
                'anomalib not installed.  Run:  pip install anomalib')
            return

        self.inferencer = TorchInferencer(path=self.model_path)
        self.get_logger().info(f'PaDiM model loaded: {self.model_path}')

    # ── tile isolation ─────────────────────────────────────────────────────────

    def _detect_tile(self, frame: np.ndarray):
        """
        Find the most prominent tile-shaped region anywhere in the frame.
        Tile may be tilted.

        Returns:
            (rot_rect, aabb)  on success
                rot_rect : ((cx, cy), (w, h), angle)  in frame coords
                aabb     : (x, y, w, h)               in frame coords
            (None, None)      if no tile candidate found
        """
        H, W = frame.shape[:2]

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred, 30, 100)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges   = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = W * H * 0.03            # tile ≥3% of frame
        max_area = W * H * 0.95            # not the whole frame
        best = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            rect = cv2.minAreaRect(cnt)    # ((cx,cy),(w,h),angle)
            (_, _), (rw_, rh_), _ = rect
            if rw_ < 1 or rh_ < 1:
                continue

            aspect = max(rw_, rh_) / max(min(rw_, rh_), 1.0)
            if aspect > 3.0:               # tiles aren't ribbon-shaped
                continue

            # rectangularity: area / (w*h of minAreaRect)
            rect_area = rw_ * rh_
            if rect_area <= 0:
                continue
            rectangularity = area / rect_area
            if rectangularity < 0.65:      # too irregular
                continue

            if area > best_area:
                best_area = area
                best = rect

        if best is None:
            return None, None

        rot_rect = best

        # axis-aligned bbox (for tracking IoU & quick viz)
        box_pts = cv2.boxPoints(rot_rect).astype(np.int32)
        x, y, w, h = cv2.boundingRect(box_pts)
        x = max(0, x);  y = max(0, y)
        w = min(W - x, w);  h = min(H - y, h)
        aabb = (x, y, w, h)

        return rot_rect, aabb

    def _warp_tile_upright(self, frame: np.ndarray,
                           rot_rect: tuple) -> np.ndarray:
        """De-rotate the tile to an upright rectangle via perspective warp."""
        box = cv2.boxPoints(rot_rect)              # 4 corners, float32
        (_, _), (rw_, rh_), _ = rot_rect
        # ensure portrait/landscape consistent: longer side = width
        tw = int(round(max(rw_, rh_)))
        th = int(round(min(rw_, rh_)))
        tw = max(tw, 16);  th = max(th, 16)

        # order box points: tl, tr, br, bl
        pts = box.astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0, 0], [tw - 1, 0],
                        [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (tw, th),
                                   flags=cv2.INTER_LINEAR)

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """IoU of two (x,y,w,h) axis-aligned boxes."""
        ax1, ay1, aw, ah = a;  ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b;  bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _to_numpy(x):
        """Coerce torch tensors / tv_tensors / numpy / None to a plain ndarray."""
        if x is None:
            return None
        # torch tensor / tv_tensors.Mask both expose .detach().cpu().numpy()
        if hasattr(x, 'detach'):
            try:
                x = x.detach().cpu().numpy()
            except Exception:
                pass
        if hasattr(x, 'numpy') and not isinstance(x, np.ndarray):
            try:
                x = x.numpy()
            except Exception:
                pass
        arr = np.asarray(x)
        # squeeze leading singleton dims: (1,1,H,W) -> (H,W)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def _mask_to_uint8(self, pred) -> np.ndarray:
        """Return an H×W uint8 mask (0/255) from an anomalib prediction."""
        mask = self._to_numpy(getattr(pred, 'pred_mask', None))
        if mask is not None and mask.size > 0:
            if mask.dtype == np.bool_:
                return (mask.astype(np.uint8)) * 255
            # numeric mask: threshold at 0.5 (works for 0/1 or 0..1 floats)
            return ((mask > 0.5).astype(np.uint8)) * 255

        amap = self._to_numpy(getattr(pred, 'anomaly_map', None))
        if amap is None or amap.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        thresh = float(amap.mean() + 2 * amap.std())
        return ((amap > thresh).astype(np.uint8)) * 255

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _camera_cb(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Camera conversion error: {e}')
            return

        if self.inferencer is None:
            if self.show_debug:
                self._show_live(self.latest_frame, None, None, 'no model')
            return

        # Detect once per frame; reuse result for tracking + preview.
        rot_rect, aabb = self._detect_tile(self.latest_frame)

        status = 'searching'
        if self.auto_scan:
            status = self._tick_auto_scan(self.latest_frame, rot_rect, aabb)

        if self.show_debug:
            self._show_live(self.latest_frame, rot_rect, aabb, status)

    # ── auto-scan FSM ──────────────────────────────────────────────────────────

    def _tick_auto_scan(self, frame: np.ndarray,
                        rot_rect: Optional[tuple],
                        aabb: Optional[tuple]) -> str:
        """
        Advance the auto-scan state machine for one frame.

        Returns a short status string for the live overlay.
        """
        # No tile in view -------------------------------------------------------
        if rot_rect is None:
            self._miss_count += 1
            if self._miss_count >= self.reacquire_misses:
                if self._awaiting_clear:
                    self.get_logger().info(
                        'Tile cleared - ready for next inspection')
                self._tracked_box  = None
                self._tracked_aabb = None
                self._stable_count = 0
                self._awaiting_clear = False
            return 'no tile' if not self._awaiting_clear else 'waiting clear'

        # A tile is visible -----------------------------------------------------
        self._miss_count = 0

        # Still waiting for the previous tile to be removed - do nothing.
        if self._awaiting_clear:
            return 'already inspected (remove tile)'

        # First sighting, or different tile -> reset tracker.
        if self._tracked_aabb is None or \
           self._iou(aabb, self._tracked_aabb) < self.stable_iou:
            self._tracked_box  = rot_rect
            self._tracked_aabb = aabb
            self._stable_count = 1
            return f'tracking 1/{self.stable_frames}'

        # Same tile as last frame -> increment stability.
        self._tracked_box  = rot_rect       # keep latest pose
        self._tracked_aabb = aabb
        self._stable_count += 1

        if self._stable_count < self.stable_frames:
            return f'tracking {self._stable_count}/{self.stable_frames}'

        # Stable long enough -> fire inspection.
        self._auto_tile_id += 1
        self.get_logger().info(
            f'Auto-inspect: tile_id={self._auto_tile_id} stable for '
            f'{self._stable_count} frames')
        try:
            self._inspect_with_rect(self._auto_tile_id, frame, rot_rect, aabb)
        except Exception as e:
            self.get_logger().error(
                f'Auto-inspect failed: {e}')

        # Block re-inspection until the tile leaves the frame.
        self._awaiting_clear = True
        self._stable_count   = 0
        return f'inspected #{self._auto_tile_id}'

    # ── live preview ───────────────────────────────────────────────────────────

    def _show_live(self, frame: np.ndarray,
                   rot_rect: Optional[tuple],
                   aabb: Optional[tuple],
                   status: str) -> None:
        """Show the camera feed with the (rotated) candidate tile overlay."""
        try:
            preview = frame.copy()

            if rot_rect is not None:
                box = cv2.boxPoints(rot_rect).astype(np.int32)
                cv2.drawContours(preview, [box], 0, (255, 200, 0), 2)
            if aabb is not None:
                x, y, bw, bh = aabb
                cv2.rectangle(preview, (x, y), (x + bw, y + bh),
                              (120, 120, 120), 1)

            mode = 'AUTO' if self.auto_scan else 'MANUAL'
            cv2.putText(preview, f'{mode}  {status}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 200, 0), 2, cv2.LINE_AA)
            cv2.imshow('Anomaly detector - live', preview)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'Live preview error: {e}')

    # ── /inspect_tile service ──────────────────────────────────────────────────

    def _inspect_tile_cb(self, request: InspectTile.Request,
                         response: InspectTile.Response) -> InspectTile.Response:
        if self.latest_frame is None:
            self.get_logger().warn('No camera frame received yet')
            response.success = False
            response.anomaly_detected = False
            response.defect_score = 0.0
            return response

        if self.inferencer is None:
            self.get_logger().error('Model not loaded')
            response.success = False
            response.anomaly_detected = False
            response.defect_score = 0.0
            return response

        frame   = self.latest_frame.copy()
        tile_id = request.tile_id
        rot_rect, aabb = self._detect_tile(frame)

        if rot_rect is None:
            self.get_logger().warn(
                f'Tile {tile_id}: no tile detected in current frame')
            response.success = False
            response.anomaly_detected = False
            response.defect_score = 0.0
            return response

        try:
            result = self._inspect_with_rect(tile_id, frame, rot_rect, aabb)
            response.success          = True
            response.anomaly_detected = result['anomaly_detected']
            response.defect_score     = result['defect_score']
        except Exception as e:
            self.get_logger().error(
                f'Inference failed for tile {tile_id}: {e}')
            response.success          = False
            response.anomaly_detected = False
            response.defect_score     = 0.0

        return response

    # ── shared inference path ──────────────────────────────────────────────────

    def _inspect_with_rect(self, tile_id: int, frame: np.ndarray,
                           rot_rect: tuple, aabb: tuple) -> dict:
        """
        Warp the tile upright, run anomalib, report, and (optionally) show
        the inspection popup.  Returns a dict with anomaly_detected and
        defect_score.  Raises on inference failure.
        """
        tile_crop = self._warp_tile_upright(frame, rot_rect)
        self.get_logger().info(
            f'Tile {tile_id}: warped to '
            f'{tile_crop.shape[1]}x{tile_crop.shape[0]}  '
            f'angle={rot_rect[2]:.1f} deg')

        pred = self.inferencer.predict(image=tile_crop)
        anomaly_detected = bool(pred.pred_label)
        defect_score     = float(pred.pred_score)

        mask_u8 = self._mask_to_uint8(pred)

        self.get_logger().info(
            f'Tile {tile_id}: {"NOK" if anomaly_detected else "OK"}  '
            f'score={defect_score:.4f}')

        self._send_to_report(tile_id, tile_crop, mask_u8, anomaly_detected)

        if self.show_debug:
            self._show_debug(tile_id, frame, rot_rect, aabb, tile_crop,
                             pred.anomaly_map, mask_u8, anomaly_detected)

        return {
            'anomaly_detected': anomaly_detected,
            'defect_score':     defect_score,
        }

    # ── reporting ──────────────────────────────────────────────────────────────

    def _send_to_report(self, tile_id: int, bgr: np.ndarray,
                        mask_u8: np.ndarray, anomaly_detected: bool) -> None:
        if not self._report_client.service_is_ready():
            return
        req = ReportAnomalyTile.Request()
        req.tile_id          = tile_id
        req.anomaly_detected = anomaly_detected
        req.tile_image       = self.bridge.cv2_to_imgmsg(bgr,     encoding='bgr8')
        if anomaly_detected:
            mask_bgr     = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            req.mask_image = self.bridge.cv2_to_imgmsg(mask_bgr, encoding='bgr8')
        else:
            from sensor_msgs.msg import Image as RosImage
            req.mask_image = RosImage()
        self._report_client.call_async(req)

    # ── debug viz ──────────────────────────────────────────────────────────────

    def _show_debug(self, tile_id: int, full_frame: np.ndarray,
                    rot_rect: tuple, aabb: tuple, tile_crop: np.ndarray,
                    anomaly_map: np.ndarray, mask_u8: np.ndarray,
                    anomaly_detected: bool) -> None:
        color = (0, 0, 255) if anomaly_detected else (0, 255, 0)
        label = f'Tile {tile_id}: {"NOK" if anomaly_detected else "OK"}'

        # ── left panel: full frame with oriented detection box ───────────────
        left = full_frame.copy()
        box = cv2.boxPoints(rot_rect).astype(np.int32)
        cv2.drawContours(left, [box], 0, color, 3)
        if aabb is not None:
            x, y, bw, bh = aabb
            cv2.rectangle(left, (x, y), (x + bw, y + bh),
                          (120, 120, 120), 1)
        cv2.putText(left, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # ── right panel: tile crop with heatmap overlay ───────────────────────
        right = tile_crop.copy()
        amap = self._to_numpy(anomaly_map)
        if amap is not None and amap.size > 0:
            amap = amap.astype(np.float32)
            e_min, e_max = amap.min(), amap.max()
            if e_max > e_min:
                heat = ((amap - e_min) / (e_max - e_min) * 255).astype(np.uint8)
            else:
                heat = np.zeros(amap.shape, dtype=np.uint8)
            if heat.shape[:2] != right.shape[:2]:
                heat = cv2.resize(heat, (right.shape[1], right.shape[0]))
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            right = cv2.addWeighted(right, 0.55, heat_color, 0.45, 0)

        if mask_u8 is not None:
            m = mask_u8
            if m.shape[:2] != right.shape[:2]:
                m = cv2.resize(m, (right.shape[1], right.shape[0]))
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(right, contours, -1, (0, 0, 255), 2)

        # ── combine side by side (scale right panel to match left height) ─────
        th = left.shape[0]
        scale = th / right.shape[0]
        tw    = int(right.shape[1] * scale)
        right = cv2.resize(right, (tw, th))

        combined = np.hstack([left, right])
        cv2.imshow('Anomaly detector', combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()