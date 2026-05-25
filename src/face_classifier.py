#!/usr/bin/env python3

import json
import os
import tempfile
import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

from rins_robot.srv import FaceRecognition


class FaceClassifier(Node):

    def __init__(self):
        super().__init__("FaceClassifier")

        self.match_threshold = 0.45
        self.face_db = self.load_face_database()

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_app.prepare(ctx_id=ctx_id)

        # keep last camera frame so we can crop it when a bbox request arrives
        self.bridge = CvBridge()
        self.latest_cv_image = None
        self.image_sub = self.create_subscription(Image,
                                                 "/oakd/rgb/preview/image_raw",
                                                 self.image_callback,
                                                 qos_profile_sensor_data)

        self.classificator = self.create_service(FaceRecognition, "/classify_face", self.classify)
        self.get_logger().info("Face classifier service node initialized!")

    def load_face_database(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "face_db.json")

        if not os.path.exists(db_path):
            self.get_logger().warn(f"Face database not found at {db_path}")
            return []

        try:
            with open(db_path, "r", encoding="utf-8") as f:
                database = json.load(f)
        except Exception as e:
            self.get_logger().error(f"Could not load face database from {db_path}: {e}")
            return []

        normalized_db = []
        for entry in database:
            if "name" not in entry or "gender" not in entry or "embedding" not in entry:
                continue

            embedding = np.asarray(entry["embedding"], dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            normalized_db.append({
                "name": entry["name"],
                "gender": entry["gender"],
                "embedding": embedding,
            })

        self.get_logger().info(f"Loaded {len(normalized_db)} face embeddings from {db_path}")
        return normalized_db

    def recognize_face(self, crop):
        if crop is None or crop.size == 0:
            return "NONE", "NONE"

        faces = self.face_app.get(crop)
        if len(faces) == 0:
            return "UNKNOWN", "UNKNOWN"

        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_match = None
        best_score = -1.0

        for entry in self.face_db:
            score = float(np.dot(embedding, entry["embedding"]))
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match is None or best_score < self.match_threshold:
            return "UNKNOWN", "UNKNOWN"

        return best_match["name"], best_match["gender"]

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_cv_image = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error converting image: {e}")

    def classify(self, req, res):
        # tolerant access to bbox field
        bbox = req.bbox
        if bbox is None:
            self.get_logger().warn("Received classify request without bbox")
            res.person = "NONE"
            res.gender = "NONE"
            return res

        if self.latest_cv_image is None:
            self.get_logger().warn("No recent image available to crop")
            res.person = "NONE"
            res.gender = "NONE"
            return res

        # Work on a snapshot so the subscription callback cannot modify the image mid-request.
        frame = self.latest_cv_image.copy()

        """# attempt to coerce bbox to sequence of four numbers
        coords = None
        try:
            coords = list(bbox)
        except Exception:
            # if bbox is not iterable, try attributes commonly used
            try:
                coords = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            except Exception:
                coords = None

        if coords is None or len(coords) < 4:
            self.get_logger().warn("BBox in request is not a 4-value sequence")
            res.person = "NONE"
            res.gender = "NONE"
            return res"""

        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            self.get_logger().warn(f"Invalid bbox coords after clipping: {(x1,y1,x2,y2)}")
            res.person = "NONE"
            res.gender = "NONE"
            return res

        crop = frame[y1:y2, x1:x2]

        # For now: persist crop to temp dir for debugging and return placeholder
        try:
            tmp = tempfile.gettempdir()
            fname = os.path.join(tmp, f"face_crop_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cv2.imwrite(fname, crop)
            self.get_logger().info(f"Saved face crop to {fname}")
        except Exception as e:
            self.get_logger().warn(f"Could not save crop: {e}")

        person, gender = self.recognize_face(crop)
        res.person = person
        res.gender = gender
        return res


def main(args=None):
    rclpy.init(args=args)
    node = FaceClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()