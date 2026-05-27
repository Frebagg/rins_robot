#!/usr/bin/env python3

import json
import os
import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
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

        # keep bridge for decoding request images and optional debug saves
        self.bridge = CvBridge()
        self.crop_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tmp", "face_crops")
        os.makedirs(self.crop_output_dir, exist_ok=True)
        self.get_logger().info(f"Face crops will be saved to {self.crop_output_dir}")

        self.classificator = self.create_service(FaceRecognition, "/classify_face", self.classify)
        self.get_logger().info("Face classifier service node initialized!")

    def load_face_database(self):
        package_share_dir = get_package_share_directory("rins_robot")
        candidate_paths = [
            os.path.join(package_share_dir, "face_db.json"),
            os.path.join(os.path.dirname(package_share_dir), "face_db.json"),
            os.path.join(os.path.dirname(os.path.dirname(package_share_dir)), "face_db.json"),
        ]

        db_path = None
        for candidate_path in candidate_paths:
            self.get_logger().debug(f"Checking face database path: {candidate_path}")
            if os.path.exists(candidate_path):
                db_path = candidate_path
                break

        if db_path is None:
            self.get_logger().error(
                "Face database not found. Looked in: " + ", ".join(candidate_paths)
            )
            return []

        try:
            with open(db_path, "r", encoding="utf-8") as f:
                database = json.load(f)
            self.get_logger().info(f"Loaded face database from {db_path} with {len(database)} raw entries")
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
            else:
                self.get_logger().warn(f"Entry {entry['name']} has zero-norm embedding")

            normalized_db.append({
                "name": entry["name"],
                "gender": entry["gender"],
                "embedding": embedding,
            })

        self.get_logger().info(f"Normalized and loaded {len(normalized_db)} face embeddings")
        return normalized_db

    def recognize_face(self, crop):
        if crop is None or crop.size == 0:
            self.get_logger().info("Recognition result: person=NONE, gender=NONE, score=0.0000")
            return "NONE", "NONE"

        faces = self.face_app.get(crop)
        if len(faces) == 0:
            self.get_logger().info("Recognition result: person=UNKNOWN, gender=UNKNOWN, score=0.0000")
            return "UNKNOWN", "UNKNOWN"

        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        best_match = None
        best_score = -1.0

        self.get_logger().debug(f"Comparing against {len(self.face_db)} database entries")

        for entry in self.face_db:
            score = float(np.dot(embedding, entry["embedding"]))
            self.get_logger().debug(f"Candidate {entry['name']} ({entry['gender']}): score={score:.4f}")
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match is None or best_score < self.match_threshold:
            self.get_logger().info(f"No good match! person=UNKNOWN, gender=UNKNOWN, score={best_score:.4f}, candidates={len(self.face_db)}")
            return "UNKNOWN", "UNKNOWN"

        self.get_logger().info(f"Recognition result: person={best_match['name']}, gender={best_match['gender']}, score={best_score:.4f}")
        return best_match["name"], best_match["gender"]

    def classify(self, req, res):
        # tolerant access to bbox field and request image
        bbox = req.bbox
        if bbox is None:
            self.get_logger().warn("Received classify request without bbox")
            res.person = "NONE"
            res.gender = "NONE"
            return res

        if getattr(req, "image", None) is None:
            self.get_logger().warn("Received classify request without image")
            res.person = "NONE"
            res.gender = "NONE"
            return res

        try:
            frame = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error converting request image: {e}")
            res.person = "NONE"
            res.gender = "NONE"
            return res

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
            fname = os.path.join(self.crop_output_dir, f"face_crop_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cv2.imwrite(fname, crop)
            self.get_logger().info(f"Saved face crop to {fname}")
        except Exception as e:
            self.get_logger().warn(f"Could not save crop: {e}")

        person, gender = self.recognize_face(crop)
        res.person = person
        res.gender = gender
        self.get_logger().info(f"Returning response: person={person}, gender={gender}")
        return res


def main(args=None):
    rclpy.init(args=args)
    node = FaceClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()