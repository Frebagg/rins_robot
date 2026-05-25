#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from visualization_msgs.msg import Marker
from rins_robot.msg import FaceCoords, RingCoords, CylinderCoords


class visualizeMarkers(Node):

    def __init__(self):
        super().__init__('visualizeMarkers')

        marker_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.faceCoordClient = self.create_subscription(
            FaceCoords, '/face_coords', self.manageFaceMarkers_callback, 10)
        self.ringCoordClient = self.create_subscription(
            RingCoords, '/ring_coords', self.manageRingMarkers_callback, 10)
        self.cylinderCoordClient = self.create_subscription(
            CylinderCoords, '/cylinder_coords', self.manageCylinderMarkers_callback, 10)

        self.faceMarkerPublisher = self.create_publisher(Marker, '/face_marker', marker_qos)
        self.ringMarkerPublisher = self.create_publisher(Marker, '/ring_marker', marker_qos)
        self.cylinderMarkerPublisher = self.create_publisher(Marker, '/cylinder_marker', marker_qos)

        self.faceMarkerIds = []
        self.ringMarkerIds = []

        self.ringOffset = 100
        self.cylinderOffset = 300
        self.get_logger().info('Visualisation node initialized!')

    def manageFaceMarkers_callback(self, msg):
        if len(msg.points) != len(msg.ids):
            self.get_logger().info('Length mismatch in face_coords!')
            return

        for face, face_id in zip(msg.points, msg.ids):
            if face_id in self.faceMarkerIds:
                continue

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'faces'
            marker.id = int(face_id)
            marker.type = Marker.SPHERE
            marker.scale.x = marker.scale.y = marker.scale.z = 0.25
            marker.pose.position = face
            marker.pose.orientation.w = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.action = Marker.ADD

            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = marker.header.stamp
            text.ns = 'faces_label'
            text.id = int(face_id)
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = face.x
            text.pose.position.y = face.y
            text.pose.position.z = face.z + 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.5
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f'Face {face_id}'
            text.action = Marker.ADD

            self.faceMarkerPublisher.publish(marker)
            self.faceMarkerPublisher.publish(text)
            self.faceMarkerIds.append(face_id)
            self.get_logger().info(f'Published face {face_id}!')

    def manageRingMarkers_callback(self, msg):
        if len(msg.points) != len(msg.ids) or len(msg.points) != len(msg.colors):
            self.get_logger().info('Length mismatch in ring_coords!')
            return

        for ring, ring_id, color in zip(msg.points, msg.ids, msg.colors):
            if ring_id in self.ringMarkerIds:
                continue

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'rings'
            marker.id = int(ring_id) + self.ringOffset
            marker.type = Marker.CUBE
            marker.scale.x = marker.scale.y = marker.scale.z = 0.25
            marker.pose.position = ring
            marker.pose.orientation.w = 1.0
            r, g, b = self.color_to_rgb(color)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            marker.action = Marker.ADD

            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = marker.header.stamp
            text.ns = 'rings_label'
            text.id = int(ring_id) + self.ringOffset
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = ring.x
            text.pose.position.y = ring.y
            text.pose.position.z = ring.z + 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            text.text = f'Ring {ring_id} ({color})'
            text.action = Marker.ADD

            self.ringMarkerPublisher.publish(marker)
            self.ringMarkerPublisher.publish(text)
            self.ringMarkerIds.append(ring_id)
            self.get_logger().info(f'Published ring {ring_id}!')

    def manageCylinderMarkers_callback(self, msg):
        if not (len(msg.points) == len(msg.ids) == len(msg.colors) == len(msg.orientations) == len(msg.leaking)):
            self.get_logger().info('Length mismatch in cylinder_coords!')
            return

        for point, cyl_id, color, orientation, leaking in zip(
            msg.points, msg.ids, msg.colors, msg.orientations, msg.leaking
        ):
            r, g, b = self.color_to_rgb(color)
            marker_id = int(cyl_id) + self.cylinderOffset
            stamp = self.get_clock().now().to_msg()

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'cylinders'
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            if orientation == 'lying':
                marker_height = 0.45
                marker.scale.x = 0.25
                marker.scale.y = 0.25
            else:
                marker_height = 0.55
                marker.scale.x = 0.25
                marker.scale.y = 0.25
            marker.scale.z = marker_height
            marker.pose.position.x = point.x
            marker.pose.position.y = point.y
            marker.pose.position.z = point.z + marker_height / 2.0
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = stamp
            text.ns = 'cylinders_label'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = point.x
            text.pose.position.y = point.y
            text.pose.position.z = point.z + marker_height + 0.25
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            leak_text = 'leaking' if leaking else 'not leaking'
            text.text = f'Cylinder {cyl_id}\n{color}, {orientation}\n{leak_text}'

            self.cylinderMarkerPublisher.publish(marker)
            self.cylinderMarkerPublisher.publish(text)

    def color_to_rgb(self, color_name):
        color_name = str(color_name).lower()
        if color_name == 'red':
            return (1.0, 0.0, 0.0)
        if color_name == 'green':
            return (0.0, 1.0, 0.0)
        if color_name == 'blue':
            return (0.0, 0.0, 1.0)
        if color_name == 'yellow':
            return (1.0, 1.0, 0.0)
        if color_name == 'black':
            return (0.05, 0.05, 0.05)
        if color_name == 'white':
            return (1.0, 1.0, 1.0)
        if color_name == 'gray':
            return (0.5, 0.5, 0.5)
        if color_name == 'orange':
            return (1.0, 0.55, 0.0)
        if color_name == 'purple':
            return (0.5, 0.0, 0.8)
        return (0.5, 0.5, 0.5)


def main():
    print('Visualisation Node starting.')
    rclpy.init(args=None)
    node = visualizeMarkers()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
