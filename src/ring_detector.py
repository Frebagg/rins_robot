#!/usr/bin/env python3


"""Ring detector node. See README.md for design, topics, and tuning."""

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import message_filters

from sensor_msgs.msg import CameraInfo, CompressedImage
from rins_robot.msg import RingCoords
from geometry_msgs.msg import PointStamped, Point

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import struct

import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

# Frame to publish ring coordinates in. The face detection node uses 'map'
# (requires SLAM running). 'odom' is always published by the base and is safe
# when SLAM is not guaranteed. Set to 'map' if your downstream consumer and the
# face node both work in the map frame.
TARGET_FRAME       = 'map'

DEPTH_MIN_M        = 0.30
DEPTH_MAX_M        = 1.5

BINARY_DEPTH_MIN_M = 0.25
BINARY_DEPTH_MAX_M = 1.5

MORPH_OPEN_K  = 3
MORPH_CLOSE_K = 5

MIN_CONTOUR_PTS  = 100
AXIS_RATIO_MAX   = 3.5
AXIS_MIN_PX      = 8
AXIS_MAX_PX      = 260

MIN_CIRCULARITY  = 0.25

MAX_ELLIPSE_RESIDUAL  = 0.15

INNER_SCALE           = 0.45

MIN_RING_DEPTH_PTS    = 8

CENTRE_PATCH_HALF     = 5

MIN_CENTRE_PATCH_PTS  = 2

HOLE_DEPTH_MARGIN_M   = 0.03

HOLE_INVALID_FRACTION = 0.60

BACKPROJ_PATCH_HALF  = 8

PENDING_MINHITS       = 1
PENDING_KEEPTIME_NS   = int(4e9)

PENDING_XY_THR  = 0.55
PENDING_Z_THR   = 0.55
MATCH_XY_THR    = 0.65
MATCH_Z_THR     = 0.65
MERGE_XY_THR    = 0.45
MERGE_Z_THR     = 0.45

PUBLISH_MIN_HITS = 1

SYNC_QUEUE  = 10
SYNC_SLOP_S = 0.08

RGB_TOPIC   = '/gemini/color/image_raw/compressed'
DEPTH_TOPIC = '/gemini/depth/image_raw/compressedDepth'

NEIGHBOURHOOD_DE_THR     = 15.0
NEIGHBOURHOOD_MIN_FRAC   = 0.60

NEIGHBOURHOOD_DZ_REL_THR = 0.03

NEIGHBOURHOOD_MIN_VALID  = 6

NEIGHBOURHOOD_DEPTH_BUCKETS = (
    (0.0, 0.5, 9),
    (0.5, 1.0, 5),
    (1.0, 99.0, 3),
)

BINARY_CLEANUP_K        = 10
BINARY_CLEANUP_MIN_FRAC = 0.50

HOLLOW_ENABLE   = False
HOLLOW_K        = 5
HOLLOW_MAX_FRAC = 0.85

COLOUR_SAT_THR = 45

COLOUR_VAL_THR = 30

COLOR_RANGES_HSV = {
    'red': [
        ((0,   110, 70),  (10,  255, 255)),
        ((170, 110, 70),  (179, 255, 255)),
    ],
    'green': [((40,  80,  50),  (85,  255, 255))],
    'blue':  [((95,  120, 50),  (130, 255, 255))],
    'black': [((0,   0,   0),   (179, 80,  60))],
}

COLOR_DOMINANCE_THR = 0.30

# ---------------------------------------------------------------------------
# Precomputed constants (perf): avoid re-allocating these on every frame /
# every candidate. Functionally identical to building them inline.
# ---------------------------------------------------------------------------
_COLOR_RANGES_NP = {
    color: [(np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            for lo, hi in ranges]
    for color, ranges in COLOR_RANGES_HSV.items()
}
_KERNEL_OPEN  = np.ones((MORPH_OPEN_K,  MORPH_OPEN_K),  dtype=np.uint8)
_KERNEL_CLOSE = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), dtype=np.uint8)


class RingDetector(Node):

    def __init__(self):
        super().__init__('ring_detector')

        self.fx = self.fy = self.cx_cam = self.cy_cam = None
        self.camera_frame = None

        self.bridge      = CvBridge()
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = qos_profile_sensor_data

        self.info_sub = self.create_subscription(
            CameraInfo, '/gemini/depth/camera_info',
            self._camera_info_cb, sensor_qos)

        self._rgb_sub   = message_filters.Subscriber(
            self, CompressedImage, RGB_TOPIC,  qos_profile=sensor_qos)
        self._depth_sub = message_filters.Subscriber(
            self, CompressedImage, DEPTH_TOPIC,  qos_profile=sensor_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=SYNC_QUEUE, slop=SYNC_SLOP_S)
        self._sync.registerCallback(self._rgbd_cb)

        self.pending   = []
        self.confirmed = []
        self.next_id   = 1

        self._rej_stats = {
            'frames': 0,
            'contours': 0,
            'candidates': 0,
            'accepted': 0,
            'rej_contour_pts': 0,
            'rej_circularity': 0,
            'rej_axis_invalid': 0,
            'rej_residual': 0,
            'rej_ratio': 0,
            'rej_size': 0,
            'rej_hole': 0,
            'rej_color': 0,
            'rej_depth': 0,
            'rej_tf': 0,
            'rej_nonfinite': 0,
        }
        self._last_rej_log_ns = self.get_clock().now().nanoseconds

        self.coord_pub = self.create_publisher(RingCoords, '/ring_coords', 10)
        self.create_timer(1.0 / 5.0, self._publish_cb)

        cv2.namedWindow('Ring detector – RGB',    cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ring detector – Binary', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ring detector – Depth',  cv2.WINDOW_NORMAL)

        self.get_logger().info('RingDetector initialised – waiting for camera_info…')

    def _camera_info_cb(self, msg: CameraInfo):
        if self.fx is not None:
            return
        k = msg.k
        self.fx, self.fy       = k[0], k[4]
        self.cx_cam, self.cy_cam = k[2], k[5]
        self.camera_frame        = msg.header.frame_id
        self.get_logger().info(
            f'Camera info: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx_cam:.1f} cy={self.cy_cam:.1f}  frame={self.camera_frame}')
        self.destroy_subscription(self.info_sub)

    def _rgbd_cb(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        if self.fx is None:
            self.get_logger().warn(
                'Waiting for camera_info…', throttle_duration_sec=5.0)
            return

        self._rej_stats['frames'] += 1

        try:
            rgb = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_raw = self._decode_compressed_depth(depth_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge: {e}')
            return
        except ValueError as e:
            self.get_logger().error(f'Compressed depth decode failed: {e}')
            return

        depth_raw = cv2.medianBlur(depth_raw, 5)

        depth_m = depth_raw.astype(np.float32) / 1000.0
        # Perf: integer depth sources (16UC1) can never produce NaN/inf after
        # the cast, so the isfinite scrub is only needed for float encodings.
        if not np.issubdtype(depth_raw.dtype, np.integer):
            depth_m[~np.isfinite(depth_m)] = 0.0

        depth_m[depth_m > DEPTH_MAX_M] = 0.0

        h_rgb, w_rgb = rgb.shape[:2]
        h_dep, w_dep = depth_m.shape[:2]

        depth_m[h_dep // 2:, :] = 0.0

        # Perf: single fused range test. Identical result to the previous
        # valid-mask & binary-range combination because the binary range is a
        # superset of (DEPTH_MIN_M, DEPTH_MAX_M) on both ends here, and depth_m
        # is already zeroed above DEPTH_MAX_M.
        lo = max(DEPTH_MIN_M, BINARY_DEPTH_MIN_M)
        hi = min(DEPTH_MAX_M, BINARY_DEPTH_MAX_M)
        binary = (((depth_m > lo) & (depth_m < hi))
                  .astype(np.uint8) * 255)

        rgb_at_dep_res = cv2.resize(rgb, (w_dep, h_dep),
                                    interpolation=cv2.INTER_AREA)
        cc_mask = self._neighbourhood_color_consistency(rgb_at_dep_res, depth_m)
        binary = cv2.bitwise_and(binary, cc_mask)

        ring_color_mask = self._ring_color_mask(rgb_at_dep_res)
        binary = cv2.bitwise_and(binary, ring_color_mask)

        binary = self._sparse_neighbourhood_cleanup(
            binary, BINARY_CLEANUP_K, BINARY_CLEANUP_MIN_FRAC)

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _KERNEL_OPEN)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _KERNEL_CLOSE)

        if HOLLOW_ENABLE:
            binary = self._hollow_out(binary, HOLLOW_K, HOLLOW_MAX_FRAC)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        raw_ellipses_rgb = []

        scale_x = w_rgb / w_dep
        scale_y = h_rgb / h_dep

        for idx, cnt in enumerate(contours):
            self._rej_stats['contours'] += 1
            if cnt.shape[0] < MIN_CONTOUR_PTS:
                self._rej_stats['rej_contour_pts'] += 1
                continue

            try:
                ellipse_dep = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            ax1, ax2 = ellipse_dep[1]
            if ax1 <= 0 or ax2 <= 0:
                self._rej_stats['rej_axis_invalid'] += 1
                continue

            residual = self._ellipse_fit_residual(cnt, ellipse_dep)
            if residual > MAX_ELLIPSE_RESIDUAL:
                self._rej_stats['rej_residual'] += 1
                continue

            # Perf: perimeter depths were previously computed twice per
            # candidate (once inside the hole check, once again later for the
            # depth estimate). Compute once here and reuse.
            perim_depths = self._perimeter_depths(depth_m, ellipse_dep)
            valid_pd = perim_depths[
                np.isfinite(perim_depths) &
                (perim_depths > DEPTH_MIN_M) &
                (perim_depths < DEPTH_MAX_M)]

            if len(valid_pd) < MIN_RING_DEPTH_PTS:
                self.get_logger().debug(
                    f'_ellipse_has_real_hole: insufficient perimeter depth samples ({len(valid_pd)})')
                self._rej_stats['rej_hole'] += 1
                continue
            ring_depth = float(np.median(valid_pd))

            if not self._ellipse_has_real_hole(depth_m, ellipse_dep, ring_depth):
                self._rej_stats['rej_hole'] += 1
                continue

            cx_d, cy_d = ellipse_dep[0]
            ax1_d, ax2_d = ellipse_dep[1]
            ellipse_rgb = (
                (cx_d * scale_x, cy_d * scale_y),
                (ax1_d * scale_x, ax2_d * scale_y),
                ellipse_dep[2]
            )

            candidates.append((ellipse_dep, ellipse_rgb, ring_depth))
            self._rej_stats['candidates'] += 1
            raw_ellipses_rgb.append(ellipse_dep)

        vis   = rgb.copy()
        depth_vis = self._depth_to_gray(depth_m)

        now   = self.get_clock().now()
        stamp = rgb_msg.header.stamp

        # Perf: one HSV conversion of the full-resolution RGB frame, shared by
        # all candidates (previously converted once per candidate). Computed
        # lazily so frames without candidates skip it entirely.
        hsv_full = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV) if candidates else None

        for ellipse_dep, ellipse_rgb, ring_depth in candidates:

            color_name = self._classify_color(hsv_full, ellipse_rgb)
            if color_name == 'unknown':
                self._rej_stats['rej_color'] += 1
                continue

            cx_d = int(round(ellipse_dep[0][0]))
            cy_d = int(round(ellipse_dep[0][1]))

            # Candidates only reach this point with enough valid perimeter
            # samples (checked above), so the ring depth is the perimeter
            # median computed in the first loop. The centre-patch fallback for
            # the insufficient-samples case is kept for safety.
            depth_val = ring_depth
            if depth_val is None:

                depth_val = self._sample_depth_patch(depth_m, cx_d, cy_d)
                if depth_val is None:
                    self._rej_stats['rej_depth'] += 1
                    continue

            point_cam = self._backproject(cx_d, cy_d, depth_val)
            point_map = self._to_map(point_cam, self.camera_frame, stamp)
            if point_map is None:
                self._rej_stats['rej_tf'] += 1
                continue

            mx, my, mz = point_map.point.x, point_map.point.y, point_map.point.z
            if not np.isfinite([mx, my, mz]).all():
                self._rej_stats['rej_nonfinite'] += 1
                continue

            self._update_tracking(mx, my, mz, color_name, now)
            self._rej_stats['accepted'] += 1

            try:
                cv2.ellipse(vis, ellipse_rgb, (0, 255, 0), 2)
            except Exception:
                pass
            cx_v = int(round(ellipse_rgb[0][0]))
            cy_v = int(round(ellipse_rgb[0][1]))
            cv2.circle(vis, (cx_v, cy_v), 4, (0, 0, 255), -1)
            cv2.putText(vis, f'{color_name} d={depth_val:.2f}m',
                        (cx_v + 8, cy_v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        self._remove_stale_pending(now)
        self._merge_confirmed()
        self._maybe_log_rejection_summary(now)

        n_pub = sum(1 for _, _, c, _, _ in self.confirmed if c >= PUBLISH_MIN_HITS)
        cv2.putText(vis,
                    f'Confirmed rings: {n_pub}  |  Pending: {len(self.pending)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis,
                    f'Confirmed rings: {n_pub}  |  Pending: {len(self.pending)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Ring detector – RGB',    vis)
        cv2.imshow('Ring detector – Binary', binary)
        cv2.imshow('Ring detector – Depth',  depth_vis)
        key = cv2.waitKey(1)
        if key == 27:
            rclpy.shutdown()

    @staticmethod
    def _decode_compressed_depth(msg: CompressedImage) -> np.ndarray:
        data = bytes(msg.data)
        if not data:
            raise ValueError('empty payload')

        png_signature = b'\x89PNG\r\n\x1a\n'
        png_start = data.find(png_signature)
        if png_start < 0:
            png_start = 0

        decoded = cv2.imdecode(
            np.frombuffer(data[png_start:], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if decoded is None:
            raise ValueError(f'OpenCV could not decode payload format={msg.format!r}')

        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]

        if '32FC1' in msg.format:
            if png_start < 12:
                raise ValueError('32FC1 compressedDepth payload missing config header')
            _, depth_quant_a, depth_quant_b = struct.unpack_from('<iff', data, 0)
            inv_depth = decoded.astype(np.float32)
            depth_m = np.zeros(inv_depth.shape, dtype=np.float32)
            valid = inv_depth > 0.0
            denom = inv_depth[valid] - depth_quant_b
            good = denom > 0.0
            depth_m_valid = np.zeros_like(denom, dtype=np.float32)
            depth_m_valid[good] = depth_quant_a / denom[good]
            depth_m[valid] = depth_m_valid
            depth_m[~np.isfinite(depth_m)] = 0.0
            return np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        if decoded.dtype == np.uint16:
            return decoded

        if np.issubdtype(decoded.dtype, np.floating):
            decoded = np.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(decoded * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        return decoded.astype(np.uint16)

    def _neighbourhood_color_consistency(
        self, bgr: np.ndarray, depth_m: np.ndarray
    ) -> np.ndarray:
        h, w = bgr.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)

        # depth_m has already been scrubbed of non-finite values upstream, so
        # validity reduces to a positivity test.
        valid = depth_m > 0.0
        if not valid.any():
            return out

        # Perf: LAB conversion done once per frame and cropped per bucket
        # (previously recomputed for every kernel size).
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int32)

        # Perf: the per-pixel result only matters where the pixel itself is
        # valid AND inside the bucket's depth band (output is forced to 0
        # everywhere else, exactly as before). So each kernel pass is run only
        # on the bounding box of its bucket's valid pixels, expanded by the
        # kernel radius so every neighbourhood window sees the same data it
        # would in a full-frame pass. With the bottom half of the depth image
        # zeroed upstream, this cuts the work dramatically — especially for
        # the expensive K=9 near-range bucket.
        for blo, bhi, K in NEIGHBOURHOOD_DEPTH_BUCKETS:
            m = (depth_m >= blo) & (depth_m < bhi) & valid
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                continue
            pad = K // 2
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(h, int(ys.max()) + 1 + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(w, int(xs.max()) + 1 + pad)

            res = self._neighbourhood_pass(
                lab[y0:y1, x0:x1],
                depth_m[y0:y1, x0:x1],
                valid[y0:y1, x0:x1],
                K)

            sub = m[y0:y1, x0:x1]
            roi = out[y0:y1, x0:x1]
            roi[sub] = res[sub]
        return out

    @staticmethod
    def _neighbourhood_pass(
        lab: np.ndarray, depth_m: np.ndarray, valid: np.ndarray, K: int
    ) -> np.ndarray:
        thr_color2 = float(NEIGHBOURHOOD_DE_THR * NEIGHBOURHOOD_DE_THR)

        h, w = lab.shape[:2]
        lab_l = lab[..., 0]
        lab_a = lab[..., 1]
        lab_b = lab[..., 2]

        valid_u8 = valid.astype(np.uint8)

        depth_thr = (NEIGHBOURHOOD_DZ_REL_THR * depth_m).astype(np.float32)

        pad = K // 2
        l_pad = cv2.copyMakeBorder(lab_l, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        a_pad = cv2.copyMakeBorder(lab_a, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        b_pad = cv2.copyMakeBorder(lab_b, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        depth_pad = cv2.copyMakeBorder(depth_m, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        valid_pad = cv2.copyMakeBorder(valid_u8, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=0)

        # Perf: the centre offset (dy == dx == pad) always agrees with itself
        # wherever the pixel is valid (zero colour distance below threshold,
        # zero depth difference below the strictly positive relative
        # threshold), so it is folded in here instead of running a full shift
        # iteration for it.
        agree_count = valid.astype(np.int32)
        valid_count = np.rint(
            cv2.boxFilter(valid_u8.astype(np.float32), -1, (K, K),
                          normalize=False, borderType=cv2.BORDER_CONSTANT)
        ).astype(np.int32)

        for dy in range(K):
            for dx in range(K):
                if dy == pad and dx == pad:
                    continue
                dl = l_pad[dy:dy + h, dx:dx + w] - lab_l
                da = a_pad[dy:dy + h, dx:dx + w] - lab_a
                db = b_pad[dy:dy + h, dx:dx + w] - lab_b
                # Perf: square in place to avoid extra temporaries.
                np.multiply(dl, dl, out=dl)
                np.multiply(da, da, out=da)
                np.multiply(db, db, out=db)
                dl += da
                dl += db
                d2_color = dl

                dz = np.abs(depth_pad[dy:dy + h, dx:dx + w] - depth_m)
                v = valid_pad[dy:dy + h, dx:dx + w]

                both = (v != 0) & (d2_color < thr_color2) & (dz < depth_thr)
                agree_count += both

        min_frac_ok = (
            agree_count.astype(np.float32) /
            np.maximum(valid_count, 1).astype(np.float32)
            >= NEIGHBOURHOOD_MIN_FRAC
        )
        enough_valid = valid_count >= NEIGHBOURHOOD_MIN_VALID
        keep = min_frac_ok & enough_valid & valid
        return (keep.astype(np.uint8)) * 255

    @staticmethod
    def _ring_color_mask(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        for _color, ranges in _COLOR_RANGES_NP.items():
            for lo, hi in ranges:
                out |= cv2.inRange(hsv, lo, hi)
        return out

    @staticmethod
    def _sparse_neighbourhood_cleanup(
        binary: np.ndarray, K: int, min_frac: float
    ) -> np.ndarray:

        mean = cv2.boxFilter(binary, ddepth=-1, ksize=(K, K),
                             normalize=True, borderType=cv2.BORDER_REPLICATE)
        thr = int(round(255 * min_frac))

        keep = (binary > 0) & (mean >= thr)
        return (keep.astype(np.uint8)) * 255

    @staticmethod
    def _hollow_out(binary: np.ndarray, K: int, max_frac: float) -> np.ndarray:
        mean = cv2.boxFilter(binary, ddepth=-1, ksize=(K, K),
                             normalize=True, borderType=cv2.BORDER_REPLICATE)
        thr = int(round(255 * max_frac))
        keep = (binary > 0) & (mean < thr)
        return (keep.astype(np.uint8)) * 255

    @staticmethod
    def _ellipse_fit_residual(contour: np.ndarray, ellipse) -> float:
        (cx, cy), (axis_a, axis_b), angle_deg = ellipse

        a = axis_a / 2.0
        b = axis_b / 2.0
        if a < 1e-3 or b < 1e-3:
            return float('inf')

        pts = contour.reshape(-1, 2).astype(np.float64)

        pts[:, 0] -= cx
        pts[:, 1] -= cy

        theta = -np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t],
                        [sin_t,  cos_t]], dtype=np.float64)
        pts = pts @ rot.T

        pts[:, 0] /= a
        pts[:, 1] /= b

        dists = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        return float(np.mean(np.abs(dists - 1.0)))

    @staticmethod
    def _ellipse_roi(shape, ellipse, margin: int = 2):
        """Integer bounding box (clamped to image) that fully contains the
        ellipse. Used to draw/sample ellipse masks on a small crop instead of
        allocating full-frame masks (perf). Returns None if the box is empty.
        """
        h, w = shape[:2]
        (ecx, ecy), (ea1, ea2), _ang = ellipse
        r = 0.5 * max(ea1, ea2) + margin
        x0 = max(0, int(np.floor(ecx - r)))
        x1 = min(w, int(np.ceil(ecx + r)) + 1)
        y0 = max(0, int(np.floor(ecy - r)))
        y1 = min(h, int(np.ceil(ecy + r)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _ellipse_has_real_hole(self, depth_m: np.ndarray, ellipse,
                               ring_depth: float) -> bool:
        # ring_depth: median of valid perimeter depths, computed by the caller
        # (which also enforces MIN_RING_DEPTH_PTS before calling).

        inner_ellipse = (
            ellipse[0],
            (ellipse[1][0] * INNER_SCALE, ellipse[1][1] * INNER_SCALE),
            ellipse[2]
        )

        # Perf: draw the inner-ellipse mask on a crop around the ellipse
        # instead of a full-frame buffer. Pixel-identical because the integer
        # ROI shift preserves cv2's centre/axis rounding.
        roi = self._ellipse_roi(depth_m.shape, ellipse)
        if roi is None:
            self.get_logger().debug('_ellipse_has_real_hole: inner ellipse produced no pixels')
            return False
        x0, y0, x1, y1 = roi
        inner_shifted = (
            (inner_ellipse[0][0] - x0, inner_ellipse[0][1] - y0),
            inner_ellipse[1],
            inner_ellipse[2]
        )
        mask_inner = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        try:
            cv2.ellipse(mask_inner, inner_shifted, 255, thickness=-1)
        except Exception:
            self.get_logger().debug('_ellipse_has_real_hole: failed to draw inner ellipse mask')
            return False

        hole_raw = depth_m[y0:y1, x0:x1][mask_inner > 0]
        if hole_raw.size == 0:
            self.get_logger().debug('_ellipse_has_real_hole: inner ellipse produced no pixels')
            return False

        n_total   = len(hole_raw)
        n_invalid = int(np.sum(
            (hole_raw == 0) |
            ~np.isfinite(hole_raw) |
            (hole_raw < DEPTH_MIN_M)
        ))

        invalid_fraction = n_invalid / n_total if n_total > 0 else 1.0

        if invalid_fraction >= HOLE_INVALID_FRACTION:
            return True

        valid_hole = hole_raw[
            np.isfinite(hole_raw) &
            (hole_raw > DEPTH_MIN_M) &
            (hole_raw < DEPTH_MAX_M)]

        if len(valid_hole) < MIN_CENTRE_PATCH_PTS:

            self.get_logger().debug(
                f'_ellipse_has_real_hole: insufficient hole depth samples ({len(valid_hole)}), invalid_fraction={invalid_fraction:.2f}')
            return False

        hole_depth = float(np.median(valid_hole))
        is_real = hole_depth > ring_depth + HOLE_DEPTH_MARGIN_M
        if not is_real:
            self.get_logger().debug(
                f'_ellipse_has_real_hole: rejected as fake-like (hole_depth={hole_depth:.2f}, ring_depth={ring_depth:.2f}, margin={HOLE_DEPTH_MARGIN_M:.2f})')
        return is_real

    def _perimeter_depths(self, depth_m: np.ndarray, ellipse,
                          delta_deg: int = 8) -> np.ndarray:
        h, w = depth_m.shape[:2]
        cx  = int(round(ellipse[0][0]))
        cy  = int(round(ellipse[0][1]))
        ax1 = max(1, int(round(ellipse[1][0] / 2.0)))
        ax2 = max(1, int(round(ellipse[1][1] / 2.0)))
        ang = int(round(ellipse[2]))

        pts = cv2.ellipse2Poly((cx, cy), (ax1, ax2), ang, 0, 360, delta_deg)
        if pts is None or len(pts) == 0:
            return np.array([], dtype=np.float32)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        return depth_m[ys, xs]

    def _sample_depth_patch(self, depth_m: np.ndarray,
                            cx: int, cy: int) -> float | None:
        h, w = depth_m.shape[:2]
        r = BACKPROJ_PATCH_HALF
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return None
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[(patch > DEPTH_MIN_M) & (patch < DEPTH_MAX_M) &
                      np.isfinite(patch)]
        if len(valid) < 4:
            return None
        return float(np.median(valid))

    def _classify_color(self, hsv: np.ndarray, ellipse) -> str:
        # hsv: full-resolution HSV frame, converted once per frame by the
        # caller and shared across candidates (perf).
        inner_ell = (
            ellipse[0],
            (ellipse[1][0] * INNER_SCALE, ellipse[1][1] * INNER_SCALE),
            ellipse[2]
        )

        # Perf: build the rim mask and run the colour counting on a crop
        # around the ellipse instead of full-frame buffers per candidate.
        roi = self._ellipse_roi(hsv.shape, ellipse)
        if roi is None:
            return 'unknown'
        x0, y0, x1, y1 = roi
        rh, rw = y1 - y0, x1 - x0
        shift = lambda e: ((e[0][0] - x0, e[0][1] - y0), e[1], e[2])

        mask_outer = np.zeros((rh, rw), dtype=np.uint8)
        mask_inner = np.zeros((rh, rw), dtype=np.uint8)
        try:
            cv2.ellipse(mask_outer, shift(ellipse),   255, thickness=-1)
            cv2.ellipse(mask_inner, shift(inner_ell), 255, thickness=-1)
        except Exception:
            return 'unknown'
        rim_mask = cv2.subtract(mask_outer, mask_inner)
        total = int(np.count_nonzero(rim_mask))
        if total < 20:

            return 'unknown'

        hsv_roi = hsv[y0:y1, x0:x1]

        best_color = None
        best_count = 0
        for color, ranges in _COLOR_RANGES_NP.items():
            color_mask = np.zeros((rh, rw), dtype=np.uint8)
            for lo, hi in ranges:
                color_mask |= cv2.inRange(hsv_roi, lo, hi)

            count = int(np.count_nonzero(cv2.bitwise_and(rim_mask, color_mask)))
            if count > best_count:
                best_count = count
                best_color = color

        if best_color is None:
            return 'unknown'

        dominance = best_count / total
        if dominance < COLOR_DOMINANCE_THR:
            self.get_logger().debug(
                f'Color rejection: best={best_color} dominance={dominance:.2f} '
                f'< {COLOR_DOMINANCE_THR:.2f}')
            return 'unknown'

        return best_color

    def _backproject(self, u: int, v: int, depth_m: float) -> np.ndarray:
        X = (u - self.cx_cam) * depth_m / self.fx
        Y = (v - self.cy_cam) * depth_m / self.fy
        return np.array([X, Y, depth_m], dtype=float)

    def _to_map(self, point_cam: np.ndarray,
                frame_id: str, stamp) -> PointStamped | None:
        ps = PointStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp    = stamp
        ps.point.x = float(point_cam[0])
        ps.point.y = float(point_cam[1])
        ps.point.z = float(point_cam[2])

        timeout     = Duration(seconds=0.15)
        source_time = rclpy.time.Time.from_msg(stamp)

        # Transform using the robot pose AT THE TIME THE IMAGE WAS CAPTURED,
        # not the latest pose. Using the latest transform (rclpy.time.Time())
        # places the point with the robot's *current* pose even though the
        # pixel was observed earlier, which smears detections while moving.
        try:
            tf = self.tf_buffer.lookup_transform(
                TARGET_FRAME, frame_id, source_time, timeout)
            return tfg.do_transform_point(ps, tf)
        except TransformException as te:
            err = str(te).lower()
            if ('extrapolation into the future' in err or
                    'lookup would require extrapolation' in err):
                # The buffer hasn't caught up to this stamp yet: fall back to
                # the most recent transform available.
                try:
                    tf = self.tf_buffer.lookup_transform(
                        TARGET_FRAME, frame_id, rclpy.time.Time(), timeout)
                    return tfg.do_transform_point(ps, tf)
                except TransformException as te2:
                    self.get_logger().debug(f'Fallback TF failed: {te2}')
                    return None
            self.get_logger().debug(f'TF lookup failed: {te}')
            return None

    @staticmethod
    def _xy_dist(ax, ay, bx, by) -> float:
        return float(np.hypot(ax - bx, ay - by))

    @staticmethod
    def _dom_color(votes) -> str:
        return max(votes, key=votes.get) if votes else 'unknown'

    def _confirmed_colors(self) -> set:
        """Dominant colours currently held by confirmed rings."""
        return {self._dom_color(v) for _id, _pt, _c, _t, v in self.confirmed}

    def _update_tracking(self, x, y, z, color, now):
        # One ring per colour. If a detection of this colour matches the
        # already-confirmed ring of that colour (by position), refine it.
        if self._hit_confirmed(x, y, z, color, now):
            return
        # If this colour is already confirmed but the detection didn't match
        # the existing ring's position, drop it: we never confirm a second
        # ring of an already-confirmed colour.
        if color in self._confirmed_colors():
            return
        self._hit_pending(x, y, z, color, now)

    def _hit_confirmed(self, x, y, z, color, now) -> bool:
        # Association now requires matching colour AND position. With one ring
        # per colour this uniquely identifies the ring and keeps a stray
        # differently-coloured detection from polluting another ring's votes.
        best_i, best_d = -1, float('inf')
        for i, (ring_id, pt, count, last_seen, votes) in enumerate(self.confirmed):
            if self._dom_color(votes) != color:
                continue
            d_xy = self._xy_dist(pt.x, pt.y, x, y)
            d_z  = abs(pt.z - z)
            if d_xy <= MATCH_XY_THR and d_z <= MATCH_Z_THR and d_xy < best_d:
                best_d = d_xy
                best_i = i
        if best_i < 0:
            return False

        ring_id, pt, count, _, votes = self.confirmed[best_i]

        w   = 1.0 / (count + 1)
        pt.x = pt.x * (1 - w) + x * w
        pt.y = pt.y * (1 - w) + y * w
        pt.z = pt.z * (1 - w) + z * w
        votes[color] = votes.get(color, 0) + 1
        self.confirmed[best_i] = (ring_id, pt, count + 1, now, votes)
        return True

    def _hit_pending(self, x, y, z, color, now):
        best_i, best_d = -1, float('inf')
        for i, (pt, count, last_seen, votes) in enumerate(self.pending):
            d_xy = self._xy_dist(pt.x, pt.y, x, y)
            d_z  = abs(pt.z - z)
            if d_xy <= PENDING_XY_THR and d_z <= PENDING_Z_THR and d_xy < best_d:
                best_d = d_xy
                best_i = i

        if best_i >= 0:
            # Update an existing pending track.
            pt, count, _, votes = self.pending[best_i]
            w   = 1.0 / (count + 1)
            pt.x = pt.x * (1 - w) + x * w
            pt.y = pt.y * (1 - w) + y * w
            pt.z = pt.z * (1 - w) + z * w
            votes[color] = votes.get(color, 0) + 1
            count += 1
            self.pending[best_i] = (pt, count, now, votes)
            idx = best_i
        else:
            # First sighting of this ring: create a new pending track.
            pt = Point()
            pt.x, pt.y, pt.z = float(x), float(y), float(z)
            count = 1
            votes = {color: 1}
            self.pending.append((pt, count, now, votes))
            idx = len(self.pending) - 1

        # Graduation check runs for BOTH paths, so with PENDING_MINHITS == 1 a
        # brand-new detection is confirmed on the same frame it first appears.
        if count >= PENDING_MINHITS:
            best_color = max(votes, key=votes.get)
            # One ring per colour: refuse to confirm if this colour already
            # exists among confirmed rings (e.g. two same-colour blobs in one
            # frame — the first confirms, the rest are dropped here).
            if best_color in self._confirmed_colors():
                del self.pending[idx]
                return
            self.get_logger().info(
                f'Ring confirmed! id={self.next_id} color={best_color} '
                f'{TARGET_FRAME}=({pt.x:.2f},{pt.y:.2f},{pt.z:.2f})')
            self.confirmed.append((self.next_id, pt, count, now, votes))
            self.next_id += 1
            del self.pending[idx]

    def _remove_stale_pending(self, now):
        self.pending = [
            (pt, c, t, v) for pt, c, t, v in self.pending
            if max(0, (now - t).nanoseconds) <= PENDING_KEEPTIME_NS
        ]

    def _merge_confirmed(self):
        if len(self.confirmed) < 2:
            return
        merged = []
        used   = set()
        for i in range(len(self.confirmed)):
            if i in used:
                continue
            id_i, pt_i, c_i, t_i, v_i = self.confirmed[i]
            for j in range(i + 1, len(self.confirmed)):
                if j in used:
                    continue
                id_j, pt_j, c_j, t_j, v_j = self.confirmed[j]
                if self._dom_color(v_i) != self._dom_color(v_j):
                    continue
                if (self._xy_dist(pt_i.x, pt_i.y, pt_j.x, pt_j.y) <= MERGE_XY_THR
                        and abs(pt_i.z - pt_j.z) <= MERGE_Z_THR):
                    total = c_i + c_j
                    pt_i.x = (pt_i.x * c_i + pt_j.x * c_j) / total
                    pt_i.y = (pt_i.y * c_i + pt_j.y * c_j) / total
                    pt_i.z = (pt_i.z * c_i + pt_j.z * c_j) / total
                    c_i = total
                    t_i = t_i if t_i > t_j else t_j
                    id_i = min(id_i, id_j)
                    for col, cnt in v_j.items():
                        v_i[col] = v_i.get(col, 0) + cnt
                    used.add(j)
                    self.get_logger().info(
                        f'Merged confirmed ring {id_j} → {id_i}')
            merged.append((id_i, pt_i, c_i, t_i, v_i))
        self.confirmed = merged

    def _maybe_log_rejection_summary(self, now):
        now_ns = now.nanoseconds
        if now_ns - self._last_rej_log_ns < int(1e9):
            return

        s = self._rej_stats
        self.get_logger().info(
            'Reject summary (1s): '
            f'frames={s["frames"]} contours={s["contours"]} candidates={s["candidates"]} accepted={s["accepted"]} '
            f'contour_pts={s["rej_contour_pts"]} circ={s["rej_circularity"]} axis={s["rej_axis_invalid"]} '
            f'resid={s["rej_residual"]} ratio={s["rej_ratio"]} '
            f'size={s["rej_size"]} hole={s["rej_hole"]} color={s["rej_color"]} depth={s["rej_depth"]} '
            f'tf={s["rej_tf"]} nonfinite={s["rej_nonfinite"]}'
        )

        for key in s:
            s[key] = 0
        self._last_rej_log_ns = now_ns

    def _publish_cb(self):
        msg = RingCoords()
        for ring_id, pt, count, _, votes in self.confirmed:
            if count < PUBLISH_MIN_HITS:
                continue
            if not np.isfinite([pt.x, pt.y, pt.z]).all():
                continue
            best_color = max(votes, key=votes.get)
            msg.ids.append(ring_id)
            msg.points.append(pt)
            msg.colors.append(best_color)
        self.coord_pub.publish(msg)

    def _depth_to_gray(self, depth_m: np.ndarray) -> np.ndarray:
        d = depth_m.copy()
        d[~np.isfinite(d)] = 0.0
        d[(d < DEPTH_MIN_M) | (d > DEPTH_MAX_M)] = 0.0
        out   = np.zeros(d.shape, dtype=np.uint8)
        valid = d[d > 0]
        if len(valid) == 0:
            return out
        mn, mx = np.min(valid), np.max(valid)
        if mx - mn < 1e-6:
            return out
        norm = (d - mn) / (mx - mn)
        norm[d == 0] = 1.0
        return (norm * 255).astype(np.uint8)

def main():
    rclpy.init(args=None)
    node = RingDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()