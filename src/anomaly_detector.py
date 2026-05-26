#!/usr/bin/env python3
"""
ROS2 node: anomaly_detector

Loads a trained PatchCore model exported by anomalib and provides
the /inspect_tile service.  On each call it grabs the latest camera
frame, runs inference, and forwards results + images to
/report_anomaly_tile.

Parameters (ROS2):
    camera_topic  (default: /top_camera/rgb/preview/image_raw)
    model_path    (default: <package_share>/models/anomaly_model.pt)
    show_debug    (default: True)

Services provided:
    /inspect_tile  (rins_robot/srv/InspectTile)

Services called:
    /report_anomaly_tile  (rins_robot/srv/ReportAnomalyTile)
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
        ])

        self.camera_topic = self.get_parameter('camera_topic').value
        self.model_path   = str(self.get_parameter('model_path').value)
        self.show_debug   = bool(self.get_parameter('show_debug').value)

        if not self.model_path:
            self.model_path = self._default_model_path()

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.inferencer = None

        self._load_model()

        self.create_subscription(Image, self.camera_topic, self._camera_cb,
                                 qos_profile_sensor_data)
        self.create_service(InspectTile, '/inspect_tile',
                            self._inspect_tile_cb)
        self._report_client = self.create_client(ReportAnomalyTile,
                                                 '/report_anomaly_tile')

        self.get_logger().info(
            f'Anomaly detector ready  camera={self.camera_topic}')

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

    def _isolate_tile(self, frame: np.ndarray):
        """
        Detect and crop the tile visible in the center of the frame.

        Strategy:
          1. Search the central 70% of the frame for a large rectangular
             contour using Canny edges.
          2. Among candidates, pick the one whose centroid is closest to
             the frame center.
          3. Fall back to a fixed center-crop if nothing is found.

        Returns (tile_bgr, (x, y, w, h)) in original frame coordinates.
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # ── search region: central 70% ────────────────────────────────────────
        mx, my = int(w * 0.15), int(h * 0.15)
        roi = frame[my:h-my, mx:w-mx]
        rh, rw = roi.shape[:2]

        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred, 30, 100)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges   = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = rw * rh * 0.06   # tile must cover ≥6% of search region
        best, best_dist = None, float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 3.0:          # skip very elongated shapes
                continue

            # distance of bbox centre from frame centre
            dist = abs((x + bw/2 + mx) - cx) + abs((y + bh/2 + my) - cy)
            if dist < best_dist:
                best_dist = dist
                best = (x + mx, y + my, bw, bh)

        if best is not None:
            x, y, bw, bh = best
            pad = 8
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad);  y2 = min(h, y + bh + pad)
            return frame[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

        # ── fallback: fixed centre crop (central 60%) ──────────────────────────
        px, py = int(w * 0.20), int(h * 0.20)
        return frame[py:h-py, px:w-px], (px, py, w-2*px, h-2*py)

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _camera_cb(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Camera conversion error: {e}')

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

        try:
            tile_crop, bbox = self._isolate_tile(frame)
            self.get_logger().info(
                f'Tile {tile_id}: isolated at {bbox}  '
                f'size={tile_crop.shape[1]}×{tile_crop.shape[0]}')

            pred = self.inferencer.predict(image=tile_crop)

            anomaly_detected = bool(pred.pred_label)
            defect_score     = float(pred.pred_score)

            # pred_mask is bool H×W; anomaly_map is float H×W
            if pred.pred_mask is not None:
                mask_u8 = (pred.pred_mask.astype(np.uint8)) * 255
            else:
                amap = pred.anomaly_map
                thresh = float(amap.mean() + 2 * amap.std())
                mask_u8 = ((amap > thresh).astype(np.uint8)) * 255

            response.anomaly_detected = anomaly_detected
            response.defect_score     = defect_score
            response.success          = True

            self.get_logger().info(
                f'Tile {tile_id}: {"NOK" if anomaly_detected else "OK"}  '
                f'score={defect_score:.4f}')

            self._send_to_report(tile_id, tile_crop, mask_u8, anomaly_detected)

            if self.show_debug:
                self._show_debug(tile_id, frame, bbox, tile_crop,
                                 pred.anomaly_map, mask_u8, anomaly_detected)

        except Exception as e:
            self.get_logger().error(
                f'Inference failed for tile {tile_id}: {e}')
            response.success          = False
            response.anomaly_detected = False
            response.defect_score     = 0.0

        return response

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
                    bbox: tuple, tile_crop: np.ndarray,
                    anomaly_map: np.ndarray, mask_u8: np.ndarray,
                    anomaly_detected: bool) -> None:
        color = (0, 0, 255) if anomaly_detected else (0, 255, 0)
        label = f'Tile {tile_id}: {"NOK" if anomaly_detected else "OK"}'

        # ── left panel: full frame with detection box ─────────────────────────
        left = full_frame.copy()
        x, y, bw, bh = bbox
        cv2.rectangle(left, (x, y), (x+bw, y+bh), color, 3)
        cv2.putText(left, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # ── right panel: tile crop with heatmap overlay ───────────────────────
        right = tile_crop.copy()
        if anomaly_map is not None:
            amap = anomaly_map.astype(np.float32)
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
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
