# RINS-robot-

## Face Detection Service

### face_detection.py
`face_detection.py` runs YOLO object detection on camera frames to locate faces in real time and publishes their coordinates.

**Topics:**
- Publishes face coordinates to **"/face_coords"** (message type: `FaceCoords`)
  - Contains: `geometry_msgs/Point[] points` and `int32[] ids`

**Services:**
- Provides **"/bounding_box_service"** (service type: `BoundingBox`)
  - Request: `bool data`
  - Response: `string person`, `string gender`
  - Used by the face classifier to recognize detected faces via embedding matching

### face_classifier.py
`face_classifier.py` provides face recognition using InsightFace embeddings.

**Services:**
- Provides **"/classify_face"** (service type: `FaceRecognition`)
  - Request: `int32[4] bbox` (bounding box coordinates)
  - Response: `string person`, `string gender`
  - Matches detected faces against a pre-computed database (`face_db.json`) using normalized embeddings

**Database:**
- Loads face embeddings from `face_db.json` (installed to package share directory)
- Crops faces from the current camera frame and compares embeddings to previously recorded faces

**Important:** InsightFace requires NumPy < 2.0 due to ABI incompatibility with newer NumPy versions. The package is configured to use `numpy<2` (1.26.4) in the Python virtual environment to ensure cv_bridge and other dependent modules work correctly.

### face_dialogue_servicer.py
`face_dialogue_servicer.py` provides a voice-based dialogue service for task assignment.

**Services:**
- Provides **"/face_dialogue_service"** (service type: `FaceDialogue`)
  - Request: `string name`, `string gender`
  - Response: `string task`, `string cell`, `bool success`
  - Engages in dialogue to parse task commands from speech

**Topics:**
- Publishes task history to **"/face_dialogue_task_history"** (message type: `std_msgs/String`)
  - Payload: JSON list of completed tasks with ID, name, task type, and cell

**Audio Processing:**
- Records 5-second audio clips at 16 kHz mono using `sounddevice`
- Transcribes speech using `faster-whisper` (Whisper ASR model)
- Automatically falls back from CUDA float16 to float32 or CPU int8 if compute type is unsupported

**Task Parsing:**
- Recognizes commands: "inspect barrels", "count rings", "detect anomalies in the red cell", "detect anomalies in the green cell", "nothing"
- If no keywords are found, responds with "I didn't understand. Please repeat." and allows retry
- Maintains a numbered task history (starting from ID 1) and publishes on topic for external logging

## detect_rings.py
detect_rings.py publisha koordinate ringov na map gridu na topic **"/ring_coords"**. Tip sporocila je definiran v **msg/RingCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids** ter **string[] colors**.

## Speech_servicer.py
Nudi 2 servica:
- **"/greet_service"** in **"/sayColor_service"**
- uporabljata tip sporocila **Speech.srv** je iz **string data ||| bool success**, v data napies kar hoces da rece



run z:
1. ros2 run rmw_zenoh_cpp rmw_zenohd
2. ros2 launch rins_robot sim_turtlebot_nav.launch.py

ros2 launch rins_robot cylinder_detection_yolo.launch.py device:=0 confidence_threshold:=0.85 max_fps:=8.0


ros2 run rins_robot arm_mover_actions.py
-spremembe z ros2 topic pub --once /arm_command std_msgs/msg/String "{data: look_at_belt_right}"
-izbira med:
    -look_at_belt_right
    -look_at_belt_left
    -look_for_qr
    -garage
    -up
    (-manual)
ros2 run rqt_image_view rqt_image_view /top_camera/rgb/preview/image_raw
-kamera na roki