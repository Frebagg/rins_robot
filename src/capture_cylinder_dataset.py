#!/usr/bin/env python3

import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CylinderDatasetCapture(Node):
    def __init__(self):
        super().__init__('cylinder_dataset_capture')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('image_topic', '/oakd/rgb/preview/image_raw'),
                ('output_dir', os.path.expanduser('~/ris/ros_ws/src/rins_robot/datasets/cylinder_yolo/raw_images')),
                ('prefix', 'cylinder'),
                ('save_every_sec', 0.0),
                ('show_preview', True),
            ],
        )
        self.image_topic = self.get_parameter('image_topic').value
        self.output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))
        self.prefix = str(self.get_parameter('prefix').value)
        self.save_every_sec = float(self.get_parameter('save_every_sec').value)
        self.show_preview = bool(self.get_parameter('show_preview').value)

        os.makedirs(self.output_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.count = len([f for f in os.listdir(self.output_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.last_save_time = 0.0

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.get_logger().info(f'Saving images to: {self.output_dir}')
        self.get_logger().info(f'image_topic={self.image_topic}')
        if self.save_every_sec > 0:
            self.get_logger().info(f'Autosave every {self.save_every_sec:.2f}s')
        else:
            self.get_logger().info('Manual save: focus preview window and press s')

    def image_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}')
            return

        now = time.monotonic()
        should_save = False

        preview = bgr.copy()
        cv2.putText(preview, f'saved: {self.count} | press s to save, q to close preview',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, f'saved: {self.count} | press s to save, q to close preview',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if self.show_preview:
            cv2.imshow('Cylinder dataset capture', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                should_save = True
            elif key == ord('q'):
                cv2.destroyWindow('Cylinder dataset capture')
                self.show_preview = False

        if self.save_every_sec > 0 and now - self.last_save_time >= self.save_every_sec:
            should_save = True

        if should_save:
            self._save_image(bgr)
            self.last_save_time = now

    def _save_image(self, bgr):
        filename = f'{self.prefix}_{self.count:06d}.jpg'
        path = os.path.join(self.output_dir, filename)
        ok = cv2.imwrite(path, bgr)
        if ok:
            self.count += 1
            self.get_logger().info(f'Saved {path}')
        else:
            self.get_logger().error(f'Failed to save {path}')


def main(args=None):
    rclpy.init(args=args)
    node = CylinderDatasetCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
