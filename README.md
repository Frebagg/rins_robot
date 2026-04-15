# RINS-robot-

## Face_detection.py
Face_detection.py publisha koordinate obrazov na map gridu na topic **"/face_coords"**. Tip sporocila je definiran v **msg/FaceCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids**.

## detect_rings.py
detect_rings.py publisha koordinate ringov na map gridu na topic **"/ring_coords"**. Tip sporocila je definiran v **msg/RingCoords.msg** in vsebuje **geometry_msgs/Point[] points** ter **int32[] ids** ter **string[] colors**.

## Speech_servicer.py
Nudi 2 servica:
- **"/greet_service"** in **"/sayColor_service"**
- uporabljata tip sporocila **Speech.srv** je iz **string data ||| bool success**, v data napies kar hoces da rece


ne vem ce so cist usi te koraki na enkrat obvezni ampak runna se z vsemi na enkrat:
1. ros2 launch turtlebot4_navigation localization.launch.py map:=/home/firstmagician/ris/ros_ws/src/rins_robot/maps/izpit.yaml
2. ros2 launch turtlebot4_navigation nav2.launch.py
3. ros2 launch turtlebot4_viz view_navigation.launch.py

za rocno upravljanje:
4. ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true

