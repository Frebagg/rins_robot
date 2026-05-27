cmake_minimum_required(VERSION 3.8)
project(rins_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceCoords.msg"
  "msg/RingCoords.msg"
  "msg/CylinderCoords.msg"
  "srv/Speech.srv"
  "srv/FaceDialogue.srv"
  "srv/FaceRecognition.srv"
  "srv/BoundingBox.srv"
  "srv/GenerateReport.srv"
  "srv/ReportTaskAssignment.srv"
  "srv/ReportAnomalyTile.srv"
  "srv/ReportBarrelImage.srv"
  "srv/InspectTile.srv"
  DEPENDENCIES geometry_msgs sensor_msgs
)

install(DIRECTORY
  launch
  config
  worlds
  maps
  urdf
  models
  DESTINATION share/${PROJECT_NAME}/
)

install(FILES
  face_db.json
  DESTINATION share/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  install(DIRECTORY
    docs
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

install(PROGRAMS
  src/face_detection.py
  src/face_dialogue_servicer.py
  src/face_classifier.py
  src/ring_detector.py
  src/arm_mover_actions.py
  src/cylinder_detection.py
  src/capture_cylinder_dataset.py
  src/visualiser.py
  src/speech_servicer.py
  src/robot_commander.py
  src/line_follower.py
  src/report_generator.py
  src/anomaly_detector.py
  tools/train_cylinder_yolo.py
  tools/train_anomaly_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
