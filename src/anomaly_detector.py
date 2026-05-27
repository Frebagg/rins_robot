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
    auto_scan          (default: False)
    stable_frames      (default: 8)    frames a tile must be stable before
                                       auto-inspection fires
    stable_iou         (default: 0.85) IoU threshold for "same tile"
    reacquire_misses   (default: 6)    frames without a tile before the
                                       node is ready to inspect the next one
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from rins_robot.srv import InspectTile, ReportAnomalyTile
from std_srvs.srv import SetBool


class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__('anomaly_detector')

        self.declare_parameters(namespace='', parameters=[
            ('camera_topic', '/top_camera/rgb/preview/image_raw'),
            ('model_path', ''),
            ('show_debug', True),
            ('auto_scan', False),
            ('stable_frames', 8),
            ('stable_iou', 0.85),
            ('reacquire_misses', 6),
            # ── tile-must-overlap-center gate ────────────────────────────────
            # Inspect when any part of the tile's rotated rect overlaps a
            # thin vertical band at image center. Width of the band is
            # `center_band_ratio` * frame_width (default: very thin, so it
            # behaves like "any part of the rect crosses the center line").
            ('center_band_ratio',  0.02),
            # ── crop expansion (optional safety margins; default off) ────────
            # The warp now fits the tile's true 4-corner polygon directly,
            # so by default we want a pure tight hug. These knobs exist
            # only if you want to dial in a small safety margin later.
            ('edge_dilate_px',   0),
            ('crop_pad_ratio',   0.0),
            # ── model input geometry (must match training) ───────────────────
            ('target_size', 512),
            # ── output ───────────────────────────────────────────────────────
            ('save_dir', '/images'),
        ])

        self.camera_topic      = self.get_parameter('camera_topic').value
        self.model_path        = str(self.get_parameter('model_path').value)
        self.show_debug        = bool(self.get_parameter('show_debug').value)
        self.auto_scan         = bool(self.get_parameter('auto_scan').value)
        self.stable_frames     = int(self.get_parameter('stable_frames').value)
        self.stable_iou        = float(self.get_parameter('stable_iou').value)
        self.reacquire_misses  = int(self.get_parameter('reacquire_misses').value)
        self.center_band_ratio = float(self.get_parameter('center_band_ratio').value)
        self.edge_dilate_px    = int(self.get_parameter('edge_dilate_px').value)
        self.crop_pad_ratio    = float(self.get_parameter('crop_pad_ratio').value)
        self.target_size       = int(self.get_parameter('target_size').value)
        self.save_dir          = str(self.get_parameter('save_dir').value)

        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception as e:
            self.get_logger().warn(
                f'Could not create save_dir {self.save_dir}: {e}')

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
        # service to enable/disable auto_scan remotely
        self.create_service(SetBool, '/set_auto_scan', self._set_auto_scan_cb)
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
        Find the tile on a near-black conveyor.

        Tiles are grayish-brown, background is black — so a simple brightness
        threshold cleanly separates them. We use Otsu (which auto-picks the
        split between the two modes), then take the largest sufficiently-
        rectangular blob as the tile.

        Returns:
            (rot_rect, aabb, contour)  on success
                rot_rect : ((cx, cy), (w, h), angle)
                aabb     : (x, y, w, h)
                contour  : Nx1x2 int32 array of the tile's outline
            (None, None, None)         if no tile candidate found
        """
        H, W = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Otsu picks the threshold automatically between the two brightness
        # modes (black background ≈ 0–20, tile ≈ 80–180). Falls back
        # gracefully if the frame is all-black or all-tile.
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Safety net: if Otsu landed in a weird place (very dim or very bright
        # frame), force a low fixed threshold. Background is black, so any
        # pixel above ~10 is part of a tile (aggressive — catches dim edge
        # transition pixels too).
        if cv2.countNonZero(binary) < 0.005 * H * W:
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Clean up: small holes inside the tile (specular spots / defects),
        # tiny noise outside it. Open kills speckle, close fills holes.
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

        # Grow the mask outward so the contour reaches the true tile edge.
        # Otsu drops the dim transition pixels at the tile boundary, leaving
        # the contour ~1-3 px inside the actual edge; dilation reclaims that.
        if self.edge_dilate_px > 0:
            d = 2 * self.edge_dilate_px + 1
            dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
            binary = cv2.dilate(binary, dilate_k, iterations=1)

        # Pad by 1px so tiles touching the frame border still close into
        # contours.
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1,
                                    cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(
            padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c - 1 for c in contours]

        min_area = W * H * 0.01            # tile ≥1% of frame
        max_area = W * H * 0.95
        best_rect = None
        best_cnt  = None
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

            rect_area = rw_ * rh_
            if rect_area <= 0:
                continue
            rectangularity = area / rect_area
            if rectangularity < 0.75:      # solid tile vs irregular blob
                continue

            # Pick the largest valid tile.
            if area > best_area:
                best_area = area
                best_rect = rect
                best_cnt  = cnt

        if best_rect is None:
            return None, None, None

        rot_rect = best_rect

        # axis-aligned bbox (for tracking IoU & quick viz)
        box_pts = cv2.boxPoints(rot_rect).astype(np.int32)
        x, y, w, h = cv2.boundingRect(box_pts)
        x = max(0, x);  y = max(0, y)
        w = min(W - x, w);  h = min(H - y, h)
        aabb = (x, y, w, h)

        return rot_rect, aabb, best_cnt

    def _find_tile_quad(self, contour: np.ndarray,
                        rot_rect: tuple) -> np.ndarray:
        """
        Find 4 corner points that hug the tile's actual outline.

        Strategy:
          1. Try approxPolyDP with a few epsilon values — if the contour
             simplifies to exactly 4 points, use those (best fit for tiles
             that are genuine quadrilaterals, perspective-skewed or not).
          2. Otherwise, pick 4 extreme corners: from the rotated rect's 4
             corners, find the contour point closest to each. This gives
             a tight 4-point hug for tiles with rounded/chamfered corners
             where approxPolyDP would over-simplify.

        Returns a (4, 2) float32 array of corner points, ordered TL, TR,
        BR, BL.
        """
        # --- attempt 1: approxPolyDP ---
        # We accept a 4-point approximation only if it doesn't significantly
        # under-cover the contour. Rounded-corner tiles will simplify to 4
        # points but with corner area cut off; we want those to fall through
        # to the extreme-corners method instead.
        cnt_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        for eps_frac in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(contour, eps_frac * perimeter, closed=True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                approx_area = cv2.contourArea(approx)
                # Quad must cover ≥97% of the contour's area to be considered
                # a faithful fit. Less means we're cutting corner curvature.
                if cnt_area > 0 and approx_area / cnt_area >= 0.97:
                    quad = approx.reshape(4, 2).astype(np.float32)
                    return self._order_quad(quad)

        # --- fallback: extreme corners along the rect's diagonals ---
        # We want corners that CONTAIN the contour (so the quad's edges
        # don't cut through curved/rounded corners). For each direction
        # corresponding to a rect corner, find the contour point that is
        # FURTHEST in that direction. This naturally handles rounded
        # corners — the extreme point along the diagonal is past the
        # corner's curvature.
        rect_corners = cv2.boxPoints(rot_rect).astype(np.float32)  # (4, 2)
        center = rect_corners.mean(axis=0)
        cnt_pts = contour.reshape(-1, 2).astype(np.float32)         # (N, 2)

        chosen = []
        for rc in rect_corners:
            direction = rc - center
            direction /= (np.linalg.norm(direction) + 1e-9)
            # Project every contour point onto this direction; take the max.
            projections = (cnt_pts - center) @ direction
            chosen.append(cnt_pts[int(np.argmax(projections))])
        quad = np.array(chosen, dtype=np.float32)
        return self._order_quad(quad)

    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as TL, TR, BR, BL using sum/diff of coordinates.
        TL has smallest x+y, BR largest x+y; TR has smallest x-y, BL largest.
        """
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _warp_tile_upright(self, frame: np.ndarray,
                           rot_rect: tuple,
                           contour: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Warp the tile to an upright target_size × target_size square.

        Uses the contour's actual 4 corners (via _find_tile_quad), NOT the
        minAreaRect bounding rectangle. This gives a tight hug on tiles
        that aren't perfect rectangles (chamfered corners, perspective
        skew, slight bowing).

        Falls back to the rotated rect's corners only if no contour is given.
        """
        S = int(self.target_size)

        if contour is not None:
            src = self._find_tile_quad(contour, rot_rect)
        else:
            box = cv2.boxPoints(rot_rect).astype(np.float32)
            src = self._order_quad(box)

        # Optional outward expansion along the diagonals from the quad's
        # centroid. Default crop_pad_ratio=0 → pure tight hug.
        if self.crop_pad_ratio > 0:
            centroid = src.mean(axis=0)
            src = centroid + (src - centroid) * (1.0 + self.crop_pad_ratio)

        dst = np.array([[0, 0], [S - 1, 0],
                        [S - 1, S - 1], [0, S - 1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (S, S),
                                     flags=cv2.INTER_LINEAR)

        if contour is not None:
            # Mask anything outside the tile's true outline so the model
            # sees a clean tile-only image (no conveyor bleed at edges).
            H, W = frame.shape[:2]
            cnt_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [contour], -1, 255, thickness=cv2.FILLED)
            cnt_mask_warped = cv2.warpPerspective(cnt_mask, M, (S, S),
                                                  flags=cv2.INTER_NEAREST)
            warped = cv2.bitwise_and(warped, warped, mask=cnt_mask_warped)

        return warped

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

    def _overlaps_center(self, rot_rect: tuple, frame_width: int,
                         frame_height: int) -> bool:
        """
        True when the tile's rotated rect shares any area with the central
        vertical band of the frame.  The band is `center_band_ratio` wide
        (as a fraction of frame width), centered on x = frame_width / 2.
        """
        band_w = max(1.0, self.center_band_ratio * frame_width)
        center_band = (
            (frame_width * 0.5, frame_height * 0.5),
            (band_w, float(frame_height)),
            0.0,
        )
        retval, _ = cv2.rotatedRectangleIntersection(rot_rect, center_band)
        # retval is INTERSECT_NONE (0), INTERSECT_PARTIAL (1), or INTERSECT_FULL (2)
        return retval != cv2.INTERSECT_NONE

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
        rot_rect, aabb, contour = self._detect_tile(self.latest_frame)

        status = 'searching'
        if self.auto_scan:
            status = self._tick_auto_scan(
                self.latest_frame, rot_rect, aabb, contour)

        if self.show_debug:
            self._show_live(self.latest_frame, rot_rect, aabb, status)

    # ── auto-scan FSM ──────────────────────────────────────────────────────────

    def _tick_auto_scan(self, frame: np.ndarray,
                        rot_rect: Optional[tuple],
                        aabb: Optional[tuple],
                        contour: Optional[np.ndarray]) -> str:
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

        H, W = frame.shape[:2]
        centered = self._overlaps_center(rot_rect, W, H)

        # Re-arm path: we previously inspected a tile and have been waiting
        # for the conveyor to move it past the center. As soon as no tile is
        # overlapping the center band (even if the inspected tile is still
        # in the frame, just past center), we're ready for the next one.
        # We also drop the IoU tracker so the *next* tile entering doesn't
        # accidentally match the just-cleared one's position.
        if self._awaiting_clear:
            if not centered:
                self.get_logger().info(
                    'Center cleared - ready for next inspection')
                self._awaiting_clear = False
                self._tracked_box    = None
                self._tracked_aabb   = None
                self._stable_count   = 0
                # Fall through to normal handling below for THIS new tile,
                # if it happens to already be a different one centered.
                # (Usually it isn't — the just-inspected tile is past center.)
            else:
                return 'already inspected (remove tile)'

        # Centering gate: the tile's rotated rect must overlap the central
        # vertical band of the frame. We still track the tile (to know it's
        # the same one), but we don't accumulate "stable" frames until any
        # part of it has crossed into the band. (`centered` already computed
        # above.)

        # First sighting, or different tile -> reset tracker.
        if self._tracked_aabb is None or \
           self._iou(aabb, self._tracked_aabb) < self.stable_iou:
            self._tracked_box  = rot_rect
            self._tracked_aabb = aabb
            self._stable_count = 1 if centered else 0
            if not centered:
                return 'waiting for center'
            return f'tracking 1/{self.stable_frames}'

        # Same tile as last frame.
        self._tracked_box  = rot_rect       # keep latest pose
        self._tracked_aabb = aabb

        if not centered:
            # Hold stability counter at 0 while the tile drifts through
            # the frame — fire only once it's actually in the window.
            self._stable_count = 0
            return 'waiting for center'

        self._stable_count += 1

        if self._stable_count < self.stable_frames:
            return f'tracking {self._stable_count}/{self.stable_frames}'

        # Stable long enough -> fire inspection.
        self._auto_tile_id += 1
        self.get_logger().info(
            f'Auto-inspect: tile_id={self._auto_tile_id} stable for '
            f'{self._stable_count} frames')
        try:
            self._inspect_with_rect(
                self._auto_tile_id, frame, rot_rect, aabb, contour)
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
            H, W = preview.shape[:2]

            # Draw the central trigger band (any tile pixel inside this
            # vertical band fires the inspection).
            band_w = max(1.0, self.center_band_ratio * W)
            lo_x = int(W * 0.5 - band_w * 0.5)
            hi_x = int(W * 0.5 + band_w * 0.5)
            band = preview.copy()
            cv2.rectangle(band, (lo_x, 0), (hi_x, H), (0, 200, 200), -1)
            preview = cv2.addWeighted(preview, 0.85, band, 0.15, 0)
            cv2.line(preview, (lo_x, 0), (lo_x, H), (0, 200, 200), 1)
            cv2.line(preview, (hi_x, 0), (hi_x, H), (0, 200, 200), 1)

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
        rot_rect, aabb, contour = self._detect_tile(frame)

        if rot_rect is None:
            self.get_logger().warn(
                f'Tile {tile_id}: no tile detected in current frame')
            response.success = False
            response.anomaly_detected = False
            response.defect_score = 0.0
            return response

        try:
            result = self._inspect_with_rect(
                tile_id, frame, rot_rect, aabb, contour)
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

    def _set_auto_scan_cb(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        try:
            self.auto_scan = bool(request.data)
            response.success = True
            response.message = f"auto_scan set to {self.auto_scan}"
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"failed to set auto_scan: {e}"
            self.get_logger().error(response.message)
        return response

    # ── shared inference path ──────────────────────────────────────────────────

    def _inspect_with_rect(self, tile_id: int, frame: np.ndarray,
                           rot_rect: tuple, aabb: tuple,
                           contour: Optional[np.ndarray] = None) -> dict:
        """
        Warp the tile upright at target_size × target_size, run anomalib,
        report, save defectives, and (optionally) show the inspection popup.
        Returns a dict with anomaly_detected and defect_score.
        Raises on inference failure.
        """
        tile_crop = self._warp_tile_upright(frame, rot_rect, contour)
        self.get_logger().info(
            f'Tile {tile_id}: warped to '
            f'{tile_crop.shape[1]}x{tile_crop.shape[0]}  '
            f'angle={rot_rect[2]:.1f} deg')

        pred = self.inferencer.predict(image=tile_crop)
        anomaly_detected = bool(pred.pred_label)
        defect_score     = float(pred.pred_score)

        mask_u8 = self._mask_to_uint8(pred)
        # Anomalib may return the mask at the model's native input size; align
        # it to the saved tile so the two images overlay 1:1.
        if mask_u8.shape[:2] != tile_crop.shape[:2]:
            mask_u8 = cv2.resize(mask_u8,
                                 (tile_crop.shape[1], tile_crop.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        self.get_logger().info(
            f'Tile {tile_id}: {"NOK" if anomaly_detected else "OK"}  '
            f'score={defect_score:.4f}')

        self._send_to_report(tile_id, tile_crop, mask_u8, anomaly_detected)

        if anomaly_detected:
            self._save_defective(tile_id, tile_crop, mask_u8, defect_score)

        if self.show_debug:
            self._show_debug(tile_id, frame, rot_rect, aabb, tile_crop,
                             pred.anomaly_map, mask_u8, anomaly_detected)

        return {
            'anomaly_detected': anomaly_detected,
            'defect_score':     defect_score,
        }

    # ── saving defectives ──────────────────────────────────────────────────────

    def _save_defective(self, tile_id: int, tile_crop: np.ndarray,
                        mask_u8: np.ndarray, defect_score: float) -> None:
        """Write the warped tile and its binary defect mask to save_dir."""
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            stem = f'tile_{tile_id:04d}_{ts}_score{defect_score:.3f}'
            tile_path = os.path.join(self.save_dir, f'{stem}.png')
            mask_path = os.path.join(self.save_dir, f'{stem}_mask.png')

            ok_tile = cv2.imwrite(tile_path, tile_crop)
            ok_mask = cv2.imwrite(mask_path, mask_u8)

            if ok_tile and ok_mask:
                self.get_logger().info(
                    f'Saved defective tile + mask -> {tile_path}')
            else:
                self.get_logger().warn(
                    f'cv2.imwrite reported failure for {tile_path} / {mask_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save defective tile: {e}')

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