
# rins_robot + dis_tutorial3 merged

This archive merges the original `rins_robot` package with the local assets from `dis_tutorial3`:
- `launch/`, `config/`, `maps/`, `urdf/`, `worlds/`
- helper scripts from `dis_tutorial3/scripts/`
- patched launch files so they resolve the local package as `rins_robot`

Notes:
- External Turtlebot4 / Gazebo / Nav2 / iRobot dependencies from the original tutorial are still required on the system.
- `launch/dis_sim.launch.py` was patched to include `sim.launch.py` and `turtlebot4_spawn.launch.py`, and its default world was set to `2025/dis`.
- The package CMake was extended to install the imported launch assets and helper scripts.

# RINS-robot-

## Face_detection.py
Face_detection.py publisha koordinate obrazov na map gridu na topic **"/face_coords"**. Tip sporocila je definiran v **msg/FaceCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids**.

## Speech_servicer.py
Nudi 2 servica:
- **"/greet_service"** in **"/sayColor_service"**
- uporabljata tip sporocila **Speech.srv** je iz **string data ||| bool success**, v data napies kar hoces da rece
