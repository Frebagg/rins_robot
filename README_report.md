# Inspection Report Generation

The `report_generator` node (`src/report_generator.py`) collects data from all other nodes during the mission and generates a PDF inspection report on demand.

The generated PDF matches the format from the task specification (slide 14):
- Header with date and robot name
- Ring counting section
- Barrel inspection section with table and leak images
- Anomaly detection section with tile table and NOK tile images

---

## Running the node

```bash
ros2 run rins_robot report_generator.py
```

---

## Triggering report generation

Call the `/generate_report` service at any point (typically after the robot has finished all tasks and reached the CTO):

```bash
ros2 service call /generate_report rins_robot/srv/GenerateReport \
  "{robot_name: 'R2D2', output_path: '/home/user/report.pdf'}"
```

Both fields are optional. Defaults: robot name = `R2D2`, path = `~/inspection_report.pdf`.

---

## Data sources

### 1. Ring data — automatic (no changes needed)

The node subscribes to `/ring_coords` (`rins_robot/msg/RingCoords`).
`ring_detector.py` already publishes to this topic.  No changes needed.

The report reads `ids` and `colors` from the latest message on that topic.

---

### 2. Barrel/cylinder data — automatic (no changes needed)

The node subscribes to `/cylinder_coords` (`rins_robot/msg/CylinderCoords`).
`cylinder_detection.py` already publishes to this topic.  No changes needed.

Fields used: `ids`, `colors`, `orientations`, `leaking`.

**Barrel images are captured automatically.**
`cylinder_detection.py` now calls `/report_barrel_image` automatically the first time each leaking barrel track is confirmed. A cropped image of the barrel (with 20 px padding around the bounding box) is sent. No changes needed in any other node.

---

### 3. Task assignment — requires change in `face_dialogue_servicer.py` / `robot_commander.py`

After the robot completes a dialogue and determines which task to perform, it must call `/report_task_assignment` so the report can record who requested what.

**Service:** `/report_task_assignment` (`rins_robot/srv/ReportTaskAssignment`)

```
Request:
  string person_name   # name of the person who gave the task (from face recognition)
  string task          # one of: "rings", "barrels", "anomaly"
  string cell          # "red", "green", or "none" (only relevant for anomaly task)

Response:
  bool success
```

**Where to call it:** In `robot_commander.py` (or wherever the dialogue result is handled), after `/face_dialogue_service` returns a confirmed task:

```python
from rins_robot.srv import ReportTaskAssignment

self.report_task_client = self.create_client(ReportTaskAssignment, '/report_task_assignment')

# After dialogue completes and task is confirmed:
req = ReportTaskAssignment.Request()
req.person_name = recognised_person_name   # from /classify_face or face_db
req.task = dialogue_result.task            # "rings", "barrels", or "anomaly"
req.cell = dialogue_result.cell            # "red", "green", or "none"
self.report_task_client.call_async(req)
```

The `task` string must exactly match one of: `"rings"`, `"barrels"`, `"anomaly"`.

---

### 4. Anomaly detection results — handled by `anomaly_detector.py`

`src/anomaly_detector.py` automatically calls `/report_anomaly_tile` for every tile.
The mission controller calls `/inspect_tile` once per tile position.

**Service:** `/report_anomaly_tile` (`rins_robot/srv/ReportAnomalyTile`)

```
Request:
  int32 tile_id              # sequential tile number starting from 1
  bool anomaly_detected      # True = defect found (NOK), False = clean (OK)
  sensor_msgs/Image tile_image   # captured image of the tile (bgr8 encoding)
  sensor_msgs/Image mask_image   # segmentation mask (white=defect, black=background)
                                 # leave empty (Image()) if no segmentation

Response:
  bool success
```

**Example call from the anomaly detector:**

```python
from rins_robot.srv import ReportAnomalyTile
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

bridge = CvBridge()
client = node.create_client(ReportAnomalyTile, '/report_anomaly_tile')

req = ReportAnomalyTile.Request()
req.tile_id = tile_counter           # 1, 2, 3, ...
req.anomaly_detected = bool(is_nok)
req.tile_image = bridge.cv2_to_imgmsg(tile_bgr, encoding='bgr8')

if segmentation_mask is not None:
    req.mask_image = bridge.cv2_to_imgmsg(mask_bgr, encoding='bgr8')
else:
    req.mask_image = RosImage()      # empty — report will skip the mask image

client.call_async(req)
tile_counter += 1
```

Call this once per tile, in order, as the robot scans along the working cell.

---

## Summary of required changes per node

| Node | Change required |
|------|----------------|
| `ring_detector.py` | None — publishes `/ring_coords` already |
| `cylinder_detection.py` | Automatically calls `/report_barrel_image` on first leak detection (already done) |
| `face_dialogue_servicer.py` / `robot_commander.py` | Call `/report_task_assignment` after task is confirmed via dialogue |
| `anomaly_detector.py` | Automatically calls `/report_anomaly_tile` per tile (already done) |
| robot mission controller | Call `/inspect_tile` per tile, then `/generate_report` at the end |

---

## Dependencies

Install `reportlab` if not already available:

```bash
pip install reportlab
```

The node also requires `cv_bridge` and `opencv-python`, both of which are already used by other nodes in this package.
