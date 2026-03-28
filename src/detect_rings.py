#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)


class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # ellipse thresholds
        self.ecc_thr = 100
        self.ratio_thr = 1.5
        self.min_contour_points = 20

        # depth thresholds
        self.latest_depth = None
        self.depth_thr = 0.10
        self.min_valid_depth = 0.05
        self.max_valid_depth = 5.0
        self.min_ring_depth_points = 12
        self.min_center_depth_points = 8

        # inner ellipse scale for "hole" check
        self.inner_scale = 0.45

        # threshold for depth gray image
        self.depth_binary_threshold = 60

        # bridge
        self.bridge = CvBridge()

        # subscribers
        self.image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1
        )
        self.depth_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/depth", self.depth_callback, 1
        )

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

    def is_valid_depth(self, d):
        return np.isfinite(d) and self.min_valid_depth < d < self.max_valid_depth

    def depth_to_gray(self, depth):
        d = depth.copy().astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[(d < self.min_valid_depth) | (d > self.max_valid_depth)] = 0.0

        out = np.zeros(d.shape, dtype=np.uint8)
        valid = d[d > 0]

        if len(valid) == 0:
            return out

        mn = np.min(valid)
        mx = np.max(valid)

        if mx - mn < 1e-6:
            return out

        norm = (d - mn) / (mx - mn)
        norm[d == 0] = 1.0
        out = (norm * 255).astype(np.uint8)
        return out

    def scale_ellipse(self, ellipse, scale):
        (cx, cy), (ax1, ax2), angle = ellipse
        return ((cx, cy), (ax1 * scale, ax2 * scale), angle)

    def make_filled_ellipse_mask(self, shape, ellipse):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, thickness=-1)
        return mask

    def make_ring_mask_from_one_ellipse(self, shape, ellipse, inner_scale):
        outer_mask = self.make_filled_ellipse_mask(shape, ellipse)
        inner_ellipse = self.scale_ellipse(ellipse, inner_scale)
        inner_mask = self.make_filled_ellipse_mask(shape, inner_ellipse)
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        return ring_mask, inner_mask, inner_ellipse

    def classify_ring_color(self, bgr_image, ellipse):
        ring_mask, _, _ = self.make_ring_mask_from_one_ellipse(
            bgr_image.shape, ellipse, self.inner_scale
        )

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        ring_pixels = hsv[ring_mask > 0]

        if len(ring_pixels) < 20:
            return "unknown"

        s = ring_pixels[:, 1]
        v = ring_pixels[:, 2]
        colored = ring_pixels[s > 60]

        if len(colored) < 10:
            median_v = float(np.median(v))
            if median_v > 180:
                return "white"
            elif median_v < 60:
                return "black"
            else:
                return "gray"

        hue_vals = colored[:, 0]
        median_h = float(np.median(hue_vals))

        if median_h < 10 or median_h >= 170:
            return "red"
        elif median_h < 25:
            return "orange"
        elif median_h < 35:
            return "yellow"
        elif median_h < 85:
            return "green"
        elif median_h < 130:
            return "blue"
        elif median_h < 170:
            return "purple"

        return "unknown"

    def ellipse_is_ring(self, depth, ellipse):
        ring_mask, inner_mask, inner_ellipse = self.make_ring_mask_from_one_ellipse(
            depth.shape, ellipse, self.inner_scale
        )

        ring_depths = depth[ring_mask > 0]
        ring_depths = ring_depths[np.isfinite(ring_depths)]
        ring_depths = ring_depths[
            (ring_depths > self.min_valid_depth) & (ring_depths < self.max_valid_depth)
        ]

        if len(ring_depths) < self.min_ring_depth_points:
            return False, inner_ellipse

        ring_depth = float(np.median(ring_depths))

        center_depths = depth[inner_mask > 0]
        center_depths = center_depths[np.isfinite(center_depths)]
        center_depths = center_depths[
            (center_depths > self.min_valid_depth) & (center_depths < self.max_valid_depth)
        ]

        if len(center_depths) < self.min_center_depth_points:
            return True, inner_ellipse

        center_depth = float(np.median(center_depths))

        if center_depth > ring_depth + self.depth_thr:
            return True, inner_ellipse

        return False, inner_ellipse

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if self.latest_depth is None:
            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)
            return

        depth = self.latest_depth.copy()

        if cv_image.shape[:2] != depth.shape[:2]:
            print("RGB in depth nimata enake velikosti")
            return

        # DEPTH -> grayscale
        depth_gray = self.depth_to_gray(depth)

        # navaden threshold na depth sliki
        _, thresh = cv2.threshold(
            depth_gray,
            self.depth_binary_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # če želiš adaptive threshold, lahko uporabiš to vrstico namesto zgornjih dveh:
        # thresh = cv2.adaptiveThreshold(depth_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Binary Image", thresh)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_vis = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, contours, -1, (255, 0, 0), 1)
        cv2.imshow("Detected contours", contour_vis)

        elps = []
        for cnt in contours:
            if cnt.shape[0] < self.min_contour_points:
                continue

            ellipse = cv2.fitEllipse(cnt)

            e = ellipse[1]
            ecc1 = e[0]
            ecc2 = e[1]

            if ecc1 <= 0 or ecc2 <= 0:
                continue

            ratio = ecc1 / ecc2 if ecc1 > ecc2 else ecc2 / ecc1

            if ratio <= self.ratio_thr and ecc1 < self.ecc_thr and ecc2 < self.ecc_thr:
                elps.append(ellipse)

        vis = cv_image.copy()
        candidates = []

        for ellipse in elps:
            is_ring, inner_ellipse = self.ellipse_is_ring(depth, ellipse)

            cv2.ellipse(vis, ellipse, (255, 255, 0), 1)
            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])
            cv2.circle(vis, (cx, cy), 1, (255, 255, 0), -1)

            if not is_ring:
                continue

            candidates.append((ellipse, inner_ellipse))

        print("Processing is done! found", len(candidates), "candidates for rings")

        for ellipse, inner_ellipse in candidates:
            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
            cv2.ellipse(vis, inner_ellipse, (0, 255, 0), 1)

            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])

            color_name = self.classify_ring_color(cv_image, ellipse)

            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                vis,
                color_name,
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Detected rings", vis)
        cv2.waitKey(1)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        depth_image = depth_image.astype(np.float32)
        depth_image[depth_image == np.inf] = 0
        depth_image[~np.isfinite(depth_image)] = 0

        self.latest_depth = depth_image.copy()

        image_viz = self.depth_to_gray(depth_image)
        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()