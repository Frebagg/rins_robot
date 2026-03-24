#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from RINS_robot.msg import FaceCoords
from geometry_msgs.msg import Point
import time


from ultralytics import YOLO

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
		
		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.yolo_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.checkFace_callback, qos_profile_sensor_data)
		self.model = YOLO("yolov8n.pt")

		self.coordPublisher = self.create_publisher(FaceCoords, "/face_coords", 10)

		self.faces = []
		self.coords = [] #tukaj notri sta Point() in stevilo detekcij zanj
		self.COUNTTHRESHOLD = 7 #za spreminjat,treba testirat
		self.lastSeen = []
		self.counter= 0 #za določanje kdaj se sprozi cleanFaceList

		self.get_logger().info(f"Face detection node initialized!")

	def yolo_callback(self, data):
		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			# iterate over results
			for x in res:
				boxes = x.boxes
				if boxes.xyxy.nelement() == 0:
					continue
				bestIx = boxes.conf.argmax() #vzame bbox z najbolj samozavestno detekcijo
				bbox = boxes.xyxy[bestIx]

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

				self.faces.append((cx,cy))

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def checkFace_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width

		# get 3-channel representation of the point cloud in numpy format once per callback
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))

		# iterate over face coordinates
		for cx,cy in self.faces:
			if cx < 0 or cy < 0 or cx >= width or cy >= height:
				continue

			# read center coordinates
			d = a[cy,cx,:]
			if np.isnan(d).any(): #ce kaksen NaN continue
				continue
			detection = self.baseLink2Map(d) #map koordinate
				
			if not (detection is None):
				x = detection.point.x
				y = detection.point.y
				z = detection.point.z
				newFace = True
				for i,(face,count) in enumerate(self.coords):
					xx = face.x
					yy = face.y
					zz = face.z
					if ( np.abs(xx-x) <0.5 and np.abs(yy-y)<0.5 and np.abs(zz-z) < 1 ):
						newFace = False
						face.x = (x+xx)/2 #povpreci lokacijo ponovno zaznanih
						face.y = (y+yy)/2
						face.z = (z+zz)/2
						self.coords[i] = (face, count+1)
						self.lastSeen[i] = self.get_clock().now()
				if newFace:
					self.coords.append((detection.point,1))
					self.lastSeen.append(self.get_clock().now())
			
		if self.counter >= 30:
			self.counter = 0
			self.cleanFaceList()
		self.publishFaces()
		self.counter += 1
		

	def baseLink2Map(self, data):
		detection = PointStamped()
		detection.header.frame_id = "base_link"
		detection.header.stamp = self.get_clock().now().to_msg()
		detection.point.x = float(data[0])
		detection.point.y = float(data[1])
		detection.point.z = float(data[2])
		time_now = rclpy.time.Time()
		timeout = Duration(seconds=0.1)

		try:
			trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
			map_detection = tfg.do_transform_point(detection, trans)
		except TransformException as te:
			self.get_logger().info(f"Could not get the transform: {te}")
			return None
		return map_detection
		
	def cleanFaceList(self):
		if len(self.coords) != len(self.lastSeen):
			return
		ixList = []
		now = self.get_clock().now()
		for i in range(len(self.lastSeen)):
			if (now - self.lastSeen[i]).nanoseconds > 5000000000:  # 5s
				ixList.append(i)
		offset = 0
		for idx in ixList:
			del self.coords[idx - offset]
			del self.lastSeen[idx - offset]
			offset += 1


	def publishFaces(self):
		pub = FaceCoords()
		for face,count in self.coords:
			if count >= self.COUNTTHRESHOLD:
				pub.points.append(face)
		self.coordPublisher.publish(pub)


def main():
	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()