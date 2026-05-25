#!/usr/bin/env python3
"""YOLO segmentation based cylinder detector for the RINS TurtleBot4 project.

Expected custom model classes:
  0: cylinder_upright
  1: cylinder_lying
  2: liquid_puddle

Pipeline:
  RGB image -> YOLO instance segmentation -> color classification from mask
  -> leaking check for lying cylinders -> depth + camera_info 3D point
  -> TF to map -> normalized /cylinder_coords publication.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - only for non-ROS static checks
    get_package_share_directory = None

from rins_robot.msg import CylinderCoords


@dataclass
class Detection2D:
    kind: str                    # cylinder or puddle
    orientation: str             # upright / lying / unknown
    color: str
    confidence: float
    mask: np.ndarray             # uint8 mask, image-sized, 0/255
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]


@dataclass
class CylinderTrack:
    track_id: int
    point: Point
    color: str
    orientation: str
    leaking: bool
    count: int
    last_seen_sec: float
    color_votes: Dict[str, int]
    orientation_votes: Dict[str, int]


class YoloCylinderDetector(Node):
    def __init__(self):
        super().__init__('yolo_cylinder_detector')

        default_model = self._default_model_path()
        self.declare_parameters(
            namespace='',
            parameters=[
                ('image_topic', '/oakd/rgb/preview/image_raw'),
                ('depth_topic', '/oakd/rgb/preview/depth'),
                ('pointcloud_topic', '/oakd/rgb/preview/depth/points'),
                ('camera_info_topic', '/oakd/rgb/preview/camera_info'),
                ('model_path', default_model),
                ('map_frame', 'map'),
                ('confidence_threshold', 0.45),
                ('iou_threshold', 0.50),
                ('image_size', 640),
                ('device', ''),              # '', 'cpu', '0', ...
                ('max_fps', 5.0),
                ('show_debug', True),
                ('normalize_radius_m', 0.75),
                ('tight_duplicate_radius_m', 0.25),
                ('pending_min_hits', 3),
                ('publish_min_hits', 6),
                ('pending_keep_time_s', 4.0),
                ('ground_z_m', 0.0),
                ('min_depth_m', 0.20),
                ('max_depth_m', 5.00),
                ('max_detection_range_m', 2.50),
                ('min_depth_pixels', 15),
                ('min_cylinder_mask_pixels', 80),
                ('puddle_contact_px', 18),
                ('require_same_puddle_color', True),
            ]
        )

        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.conf_th = float(self.get_parameter('confidence_threshold').value)
        self.iou_th = float(self.get_parameter('iou_threshold').value)
        self.image_size = int(self.get_parameter('image_size').value)
        self.device = str(self.get_parameter('device').value)
        self.max_fps = float(self.get_parameter('max_fps').value)
        self.show_debug = bool(self.get_parameter('show_debug').value)
        self.normalize_radius_m = float(self.get_parameter('normalize_radius_m').value)
        self.tight_duplicate_radius_m = float(self.get_parameter('tight_duplicate_radius_m').value)
        self.pending_min_hits = int(self.get_parameter('pending_min_hits').value)
        self.publish_min_hits = int(self.get_parameter('publish_min_hits').value)
        self.pending_keep_time_s = float(self.get_parameter('pending_keep_time_s').value)
        self.ground_z_m = float(self.get_parameter('ground_z_m').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.max_detection_range_m = float(self.get_parameter('max_detection_range_m').value)
        self.min_depth_pixels = int(self.get_parameter('min_depth_pixels').value)
        self.min_cylinder_mask_pixels = int(self.get_parameter('min_cylinder_mask_pixels').value)
        self.puddle_contact_px = int(self.get_parameter('puddle_contact_px').value)
        self.require_same_puddle_color = bool(self.get_parameter('require_same_puddle_color').value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_depth: Optional[np.ndarray] = None
        self.last_depth_header = None
        self.last_cloud: Optional[np.ndarray] = None
        self.last_cloud_header = None
        self.camera_info: Optional[CameraInfo] = None
        self.fx = self.fy = self.cx = self.cy = None

        self.model = None
        self.model_names: Dict[int, str] = {}
        self._load_model()

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self.pointcloud_callback, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)

        self.coord_pub = self.create_publisher(CylinderCoords, '/cylinder_coords', 10)
        self.publish_timer = self.create_timer(0.5, self.publish_callback)

        self.tracks: List[CylinderTrack] = []
        self.pending_tracks: List[CylinderTrack] = []
        self.next_id = 1
        self.last_process_time = 0.0
        self.last_status = 'waiting'

        self.get_logger().info('YOLO cylinder detector initialized')
        self.get_logger().info(f'image={self.image_topic}')
        self.get_logger().info(f'depth={self.depth_topic}')
        self.get_logger().info(f'pointcloud={self.pointcloud_topic}')
        self.get_logger().info(f'camera_info={self.camera_info_topic}')
        self.get_logger().info(f'model_path={self.model_path}')

    def _default_model_path(self) -> str:
        candidates: List[str] = []
        if get_package_share_directory is not None:
            try:
                package_share = get_package_share_directory('rins_robot')
                candidates.extend([
                    os.path.join(package_share, 'models', 'cylinder_yolo_seg.pt'),
                    os.path.join(package_share, 'models', 'best.pt'),
                ])
            except Exception:
                pass
        candidates.extend([
            os.path.expanduser('~/ris/ros_ws/src/rins_robot/models/cylinder_yolo_seg.pt'),
            os.path.expanduser('~/ris/ros_ws/src/rins_robot/models/best.pt'),
        ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.get_logger().error(
                'Python package ultralytics is not installed. Install it with: '
                'python3 -m pip install ultralytics'
            )
            self.get_logger().error(str(exc))
            return

        if not os.path.exists(self.model_path):
            self.get_logger().error(
                'YOLO model file not found. Train the custom model and copy best.pt to this path, '
                'or pass -p model_path:=/path/to/best.pt'
            )
            return

        try:
            self.model = YOLO(self.model_path)
            names = getattr(self.model, 'names', {}) or {}
            self.model_names = {int(k): str(v) for k, v in dict(names).items()}
            self.get_logger().info(f'Loaded YOLO model with classes: {self.model_names}')
        except Exception as exc:
            self.get_logger().error(f'Failed to load YOLO model: {exc}')
            self.model = None

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            self.get_logger().warn(f'Depth conversion failed: {exc}')
            return

        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)

        # 16UC1 is usually millimeters. Some Gazebo streams may already be meters.
        finite = depth[np.isfinite(depth)]
        if depth.dtype == np.uint16 or (finite.size > 0 and float(np.nanmedian(finite)) > 20.0):
            depth = depth / 1000.0

        depth[~np.isfinite(depth)] = 0.0
        self.last_depth = depth
        self.last_depth_header = msg.header

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        if msg.height <= 0 or msg.width <= 0:
            return
        try:
            cloud = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
            cloud = cloud.reshape((msg.height, msg.width, 3)).astype(np.float32)
        except Exception as exc:
            self.get_logger().debug(f'Point cloud conversion failed: {exc}')
            return

        self.last_cloud = cloud
        self.last_cloud_header = msg.header

    def image_callback(self, msg: Image) -> None:
        now_wall = time.monotonic()
        if self.max_fps > 0 and now_wall - self.last_process_time < 1.0 / self.max_fps:
            return
        self.last_process_time = now_wall

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}')
            return

        debug = bgr.copy()

        if self.model is None:
            self._draw_status(debug, 'model missing')
            self._show_debug(debug, None)
            return

        try:
            results = self.model.predict(
                source=bgr,
                imgsz=self.image_size,
                conf=self.conf_th,
                iou=self.iou_th,
                device=(self.device if self.device else None),
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().warn(f'YOLO predict failed: {exc}')
            return

        if not results:
            self._draw_status(debug, 'no yolo results')
            self._show_debug(debug, None)
            return

        cylinders, puddles = self._parse_yolo_result(results[0], bgr)

        valid_3d = 0
        fail_reasons: List[str] = []
        for cyl in cylinders:
            leaking = self._detect_leaking(cyl, puddles)
            map_point = self._estimate_map_point(cyl.mask, msg.header)
            if map_point is None:
                fail_reasons.append('no_depth_or_tf')
                self._draw_detection(debug, cyl, leaking, accepted=False)
                continue

            valid_3d += 1
            self._update_tracks(map_point.point, cyl.color, cyl.orientation, leaking, now_wall)
            self._draw_detection(debug, cyl, leaking, accepted=True)

        for pud in puddles:
            self._draw_puddle(debug, pud)

        self._remove_stale_pending(now_wall)
        self._merge_duplicate_tracks()

        status = (
            f'yolo cylinders:{len(cylinders)} puddles:{len(puddles)} valid3d:{valid_3d} '
            f'confirmed:{len(self.tracks)} pending:{len(self.pending_tracks)}'
        )
        if fail_reasons:
            status += ' fail:' + ','.join(sorted(set(fail_reasons)))
        self.last_status = status
        self._draw_status(debug, status)
        self._show_debug(debug, self._build_mask_debug(bgr.shape, cylinders, puddles))

    def _parse_yolo_result(self, result, bgr: np.ndarray) -> Tuple[List[Detection2D], List[Detection2D]]:
        cylinders: List[Detection2D] = []
        puddles: List[Detection2D] = []
        h, w = bgr.shape[:2]

        boxes = getattr(result, 'boxes', None)
        if boxes is None or boxes.cls is None:
            return cylinders, puddles

        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy().astype(float) if boxes.conf is not None else np.ones_like(cls_ids, dtype=float)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int) if boxes.xyxy is not None else None

        for i, cls_id in enumerate(cls_ids):
            kind, orientation = self._class_to_kind(cls_id)
            if kind not in ('cylinder', 'puddle'):
                continue

            mask = self._mask_for_detection(result, i, (h, w), xyxy[i] if xyxy is not None else None)
            min_mask_pixels = self.min_cylinder_mask_pixels if kind == 'cylinder' else 20
            if mask is None or cv2.countNonZero(mask) < min_mask_pixels:
                continue

            x1, y1, x2, y2 = self._bbox_from_mask(mask)
            if x2 <= x1 or y2 <= y1:
                continue
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if kind == 'cylinder' and orientation == 'unknown':
                orientation = self._infer_orientation_from_bbox(x1, y1, x2, y2)

            color = self._classify_color(bgr, mask)
            det = Detection2D(
                kind=kind,
                orientation=orientation,
                color=color,
                confidence=float(confs[i]),
                mask=mask,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
            )
            if kind == 'cylinder':
                cylinders.append(det)
            else:
                puddles.append(det)

        return cylinders, puddles

    def _class_to_kind(self, cls_id: int) -> Tuple[str, str]:
        name = self.model_names.get(int(cls_id), str(cls_id)).lower().strip()
        name = name.replace(' ', '_').replace('-', '_')

        if any(word in name for word in ('puddle', 'liquid', 'leak', 'spill')):
            return 'puddle', 'unknown'
        if 'cylinder' in name or 'barrel' in name:
            if any(word in name for word in ('upright', 'standing', 'vertical')):
                return 'cylinder', 'upright'
            if any(word in name for word in ('lying', 'horizontal', 'sideways', 'fallen')):
                return 'cylinder', 'lying'
            return 'cylinder', 'unknown'
        return 'unknown', 'unknown'

    def _mask_for_detection(self, result, idx: int, shape_hw: Tuple[int, int], bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)

        masks = getattr(result, 'masks', None)
        if masks is not None:
            polygons = getattr(masks, 'xy', None)
            if polygons is not None and idx < len(polygons) and len(polygons[idx]) >= 3:
                pts = np.asarray(polygons[idx], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                return mask

            data = getattr(masks, 'data', None)
            if data is not None and idx < len(data):
                m = data[idx].detach().cpu().numpy().astype(np.float32)
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask[m > 0.5] = 255
                return mask

        # Fallback for detection-only models. Not ideal, but useful for smoke tests.
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
                return mask
        return None

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return 0, 0, 0, 0
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _infer_orientation_from_bbox(x1: int, y1: int, x2: int, y2: int) -> str:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return 'upright' if height >= width * 1.05 else 'lying'

    def _classify_color(self, bgr: np.ndarray, mask: np.ndarray) -> str:
        pixels = bgr[mask > 0]
        if pixels.size == 0:
            return 'unknown'

        hsv_pixels = cv2.cvtColor(pixels.reshape((-1, 1, 3)), cv2.COLOR_BGR2HSV).reshape((-1, 3))
        h = hsv_pixels[:, 0].astype(np.float32)
        s = hsv_pixels[:, 1].astype(np.float32)
        v = hsv_pixels[:, 2].astype(np.float32)

        med_s = float(np.median(s))
        med_v = float(np.median(v))
        if med_v < 55:
            return 'black'
        if med_s < 45:
            return 'gray'

        colored = hsv_pixels[(s > 50) & (v > 45)]
        if colored.shape[0] < 10:
            return 'gray'
        med_h = float(np.median(colored[:, 0]))

        if med_h < 10 or med_h >= 170:
            return 'red'
        if 10 <= med_h < 22:
            return 'orange'
        if 22 <= med_h < 35:
            return 'yellow'
        if 35 <= med_h < 85:
            return 'green'
        if 85 <= med_h < 130:
            return 'blue'
        if 130 <= med_h < 170:
            return 'purple'
        return 'unknown'

    def _detect_leaking(self, cyl: Detection2D, puddles: List[Detection2D]) -> bool:
        if cyl.orientation != 'lying':
            return False
        if not puddles:
            return False

        kernel_size = max(3, int(self.puddle_contact_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_cyl = cv2.dilate(cyl.mask, kernel, iterations=1)

        for puddle in puddles:
            if self.require_same_puddle_color:
                if puddle.color != cyl.color:
                    continue
            overlap = cv2.countNonZero(cv2.bitwise_and(dilated_cyl, puddle.mask))
            if overlap > 0:
                return True

            if self._bbox_distance(cyl.bbox, puddle.bbox) <= self.puddle_contact_px:
                return True
        return False

    @staticmethod
    def _bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        dx = max(bx1 - ax2, ax1 - bx2, 0)
        dy = max(by1 - ay2, ay1 - by2, 0)
        return float(math.hypot(dx, dy))

    def _estimate_map_point(self, mask: np.ndarray, image_header) -> Optional[PointStamped]:
        cloud_point = self._estimate_map_point_from_cloud(mask)
        if cloud_point is not None:
            return cloud_point

        if self.last_depth is None or self.fx is None or self.fy is None:
            return None

        depth = self.last_depth
        if depth.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        valid_mask = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(valid_mask) < 10:
            valid_mask = mask

        ys, xs = np.where(valid_mask > 0)
        if xs.size == 0:
            return None

        depths = depth[ys, xs]
        valid = np.isfinite(depths) & (depths > self.min_depth_m) & (depths < self.max_depth_m)
        if int(np.count_nonzero(valid)) < self.min_depth_pixels:
            return None

        xs_valid = xs[valid]
        ys_valid = ys[valid]
        d_valid = depths[valid]

        # Use a front-biased percentile so background pixels at mask borders have less influence.
        z = float(np.percentile(d_valid, 35))
        close = np.abs(d_valid - z) < 0.20
        if int(np.count_nonzero(close)) >= 5:
            u = float(np.median(xs_valid[close]))
            v = float(np.median(ys_valid[close]))
        else:
            u = float(np.median(xs_valid))
            v = float(np.median(ys_valid))

        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        z_cam = z
        detection_range = float(math.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam))
        if detection_range > self.max_detection_range_m:
            return None

        frame_id = image_header.frame_id
        stamp = image_header.stamp
        if self.last_depth_header is not None and self.last_depth_header.frame_id:
            frame_id = self.last_depth_header.frame_id
            stamp = self.last_depth_header.stamp
        elif self.camera_info is not None and self.camera_info.header.frame_id:
            frame_id = self.camera_info.header.frame_id

        ps = PointStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.point.x = float(x_cam)
        ps.point.y = float(y_cam)
        ps.point.z = float(z_cam)

        return self._transform_point_to_map(ps)

    def _estimate_map_point_from_cloud(self, mask: np.ndarray) -> Optional[PointStamped]:
        if self.last_cloud is None or self.last_cloud_header is None:
            return None

        cloud = self.last_cloud
        if cloud.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (cloud.shape[1], cloud.shape[0]), interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        valid_mask = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(valid_mask) < 10:
            valid_mask = mask

        ys, xs = np.where(valid_mask > 0)
        if xs.size == 0:
            return None

        points = cloud[ys, xs, :]
        ranges = np.linalg.norm(points, axis=1)
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(ranges)
            & (ranges > self.min_depth_m)
            & (ranges < min(self.max_depth_m, self.max_detection_range_m))
        )
        if int(np.count_nonzero(valid)) < self.min_depth_pixels:
            return None

        valid_points = points[valid]
        valid_ranges = ranges[valid]
        front_range = float(np.percentile(valid_ranges, 35))
        close = np.abs(valid_ranges - front_range) < 0.20
        if int(np.count_nonzero(close)) >= 5:
            point_cam = np.median(valid_points[close], axis=0)
        else:
            point_cam = np.median(valid_points, axis=0)

        ps = PointStamped()
        ps.header.frame_id = self.last_cloud_header.frame_id
        ps.header.stamp = self.last_cloud_header.stamp
        ps.point.x = float(point_cam[0])
        ps.point.y = float(point_cam[1])
        ps.point.z = float(point_cam[2])
        return self._transform_point_to_map(ps)

    def _transform_point_to_map(self, ps: PointStamped) -> Optional[PointStamped]:
        timeout = Duration(seconds=0.05)
        frame_id = ps.header.frame_id
        stamp = ps.header.stamp
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                frame_id,
                rclpy.time.Time.from_msg(stamp),
                timeout,
            )
            return tfg.do_transform_point(ps, tf)
        except TransformException as exc:
            # Fallback to latest transform. This is useful in simulation when timestamps jump.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    frame_id,
                    rclpy.time.Time(),
                    timeout,
                )
                return tfg.do_transform_point(ps, tf)
            except TransformException:
                self.get_logger().debug(f'TF failed for cylinder point: {exc}')
                return None

    def _update_tracks(self, point: Point, color: str, orientation: str, leaking: bool, now_sec: float) -> None:
        point = self._track_point(point)
        if self._hit_confirmed(point, color, orientation, leaking, now_sec):
            return
        self._hit_pending(point, color, orientation, leaking, now_sec)

    def _track_point(self, point: Point) -> Point:
        tracked = Point()
        tracked.x = float(point.x)
        tracked.y = float(point.y)
        tracked.z = self.ground_z_m
        return tracked

    def _hit_confirmed(self, point: Point, color: str, orientation: str, leaking: bool, now_sec: float) -> bool:
        best_idx = -1
        best_dist = 999.0
        for i, tr in enumerate(self.tracks):
            dist = math.hypot(tr.point.x - point.x, tr.point.y - point.y)
            if not self._track_matches_detection(tr, color, orientation, dist):
                continue
            if dist <= self.normalize_radius_m and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx < 0:
            return False

        self._update_track(self.tracks[best_idx], point, color, orientation, leaking, now_sec)
        self._merge_duplicate_tracks()
        return True

    def _hit_pending(self, point: Point, color: str, orientation: str, leaking: bool, now_sec: float) -> None:
        best_idx = -1
        best_dist = 999.0
        for i, tr in enumerate(self.pending_tracks):
            dist = math.hypot(tr.point.x - point.x, tr.point.y - point.y)
            if not self._track_matches_detection(tr, color, orientation, dist):
                continue
            if dist <= self.normalize_radius_m and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            tr = self.pending_tracks[best_idx]
            self._update_track(tr, point, color, orientation, leaking, now_sec)
            if tr.count >= self.pending_min_hits:
                tr.track_id = self.next_id
                self.next_id += 1
                self.tracks.append(tr)
                del self.pending_tracks[best_idx]
                self.get_logger().info(
                    f'Cylinder confirmed id={tr.track_id} color={tr.color} '
                    f'orientation={tr.orientation} map=({tr.point.x:.2f},{tr.point.y:.2f}) '
                    f'hits={tr.count}'
                )
                self._merge_duplicate_tracks()
            return

        self.pending_tracks.append(self._new_track(0, point, color, orientation, leaking, now_sec))

    def _new_track(
        self,
        track_id: int,
        point: Point,
        color: str,
        orientation: str,
        leaking: bool,
        now_sec: float,
    ) -> CylinderTrack:
        new_point = Point()
        new_point.x = float(point.x)
        new_point.y = float(point.y)
        new_point.z = float(point.z)
        return CylinderTrack(
            track_id=track_id,
            point=new_point,
            color=color,
            orientation=orientation,
            leaking=leaking,
            count=1,
            last_seen_sec=now_sec,
            color_votes={color: 1},
            orientation_votes={orientation: 1},
        )

    def _update_track(
        self,
        tr: CylinderTrack,
        point: Point,
        color: str,
        orientation: str,
        leaking: bool,
        now_sec: float,
    ) -> None:
        n = float(tr.count)
        tr.point.x = (tr.point.x * n + point.x) / (n + 1.0)
        tr.point.y = (tr.point.y * n + point.y) / (n + 1.0)
        tr.point.z = self.ground_z_m
        tr.count += 1
        tr.last_seen_sec = now_sec
        tr.color_votes[color] = tr.color_votes.get(color, 0) + 1
        tr.orientation_votes[orientation] = tr.orientation_votes.get(orientation, 0) + 1
        tr.color = self._best_vote(tr.color_votes, tr.color)
        tr.orientation = self._best_orientation_vote(tr.orientation_votes, tr.orientation)
        tr.leaking = tr.leaking or leaking

    @staticmethod
    def _same_orientation_for_tracking(old: str, new: str) -> bool:
        if old == new:
            return True
        return old == 'unknown' or new == 'unknown'

    def _track_matches_detection(
        self,
        tr: CylinderTrack,
        color: str,
        orientation: str,
        dist: float,
    ) -> bool:
        if dist <= self.tight_duplicate_radius_m:
            return True
        return tr.color == color and self._same_orientation_for_tracking(tr.orientation, orientation)

    @staticmethod
    def _best_vote(votes: Dict[str, int], fallback: str) -> str:
        if not votes:
            return fallback
        return max(votes.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _best_orientation_vote(votes: Dict[str, int], fallback: str) -> str:
        known_votes = {k: v for k, v in votes.items() if k != 'unknown'}
        if known_votes:
            return max(known_votes.items(), key=lambda item: item[1])[0]
        return fallback

    def _remove_stale_pending(self, now_sec: float) -> None:
        keep = []
        for tr in self.pending_tracks:
            if now_sec - tr.last_seen_sec <= self.pending_keep_time_s:
                keep.append(tr)
        self.pending_tracks = keep

    def _merge_duplicate_tracks(self) -> None:
        merged: List[CylinderTrack] = []

        for tr in self.tracks:
            match: Optional[CylinderTrack] = None
            for existing in merged:
                dist = math.hypot(existing.point.x - tr.point.x, existing.point.y - tr.point.y)
                if not self._track_matches_detection(existing, tr.color, tr.orientation, dist):
                    continue
                if dist <= self.normalize_radius_m:
                    match = existing
                    break

            if match is None:
                merged.append(tr)
                continue

            total_count = match.count + tr.count
            if total_count <= 0:
                total_count = 1
            match.point.x = (match.point.x * match.count + tr.point.x * tr.count) / total_count
            match.point.y = (match.point.y * match.count + tr.point.y * tr.count) / total_count
            match.point.z = self.ground_z_m
            match.count = total_count
            match.last_seen_sec = max(match.last_seen_sec, tr.last_seen_sec)
            match.leaking = match.leaking or tr.leaking
            for col, count in tr.color_votes.items():
                match.color_votes[col] = match.color_votes.get(col, 0) + count
            for orient, count in tr.orientation_votes.items():
                match.orientation_votes[orient] = match.orientation_votes.get(orient, 0) + count
            match.color = self._best_vote(match.color_votes, match.color)
            match.orientation = self._best_orientation_vote(match.orientation_votes, match.orientation)

        self.tracks = merged

    def publish_callback(self) -> None:
        msg = CylinderCoords()
        for tr in self.tracks:
            if tr.count < self.publish_min_hits:
                continue
            msg.points.append(tr.point)
            msg.ids.append(int(tr.track_id))
            msg.colors.append(str(tr.color))
            msg.orientations.append(str(tr.orientation))
            msg.leaking.append(bool(tr.leaking))
        self.coord_pub.publish(msg)

    def _draw_detection(self, img: np.ndarray, det: Detection2D, leaking: bool, accepted: bool) -> None:
        x1, y1, x2, y2 = det.bbox
        color_bgr = (0, 255, 0) if accepted else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        label = f'{det.color} {det.orientation} leak:{int(leaking)} {det.confidence:.2f}'
        cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

    def _draw_puddle(self, img: np.ndarray, det: Detection2D) -> None:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(img, f'puddle {det.color}', (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

    def _draw_status(self, img: np.ndarray, text: str) -> None:
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    def _build_mask_debug(self, shape, cylinders: List[Detection2D], puddles: List[Detection2D]) -> np.ndarray:
        h, w = shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for det in cylinders:
            bgr = self._color_name_to_bgr(det.color)
            out[det.mask > 0] = bgr
        for det in puddles:
            out[det.mask > 0] = (255, 0, 255)
        return out

    @staticmethod
    def _color_name_to_bgr(name: str) -> Tuple[int, int, int]:
        return {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'black': (20, 20, 20),
            'purple': (180, 0, 180),
            'gray': (130, 130, 130),
            'unknown': (255, 255, 255),
        }.get(name, (255, 255, 255))

    def _show_debug(self, debug: np.ndarray, mask_debug: Optional[np.ndarray]) -> None:
        if not self.show_debug:
            return
        cv2.imshow('YOLO cylinder detector', debug)
        if mask_debug is not None:
            cv2.imshow('YOLO cylinder masks', mask_debug)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloCylinderDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
