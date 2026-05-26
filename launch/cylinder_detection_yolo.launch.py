import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _default_model_path():
    package_share = get_package_share_directory('rins_robot')
    source_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    candidates = [
        os.path.join(package_share, 'models', 'cylinder_yolo_seg.pt'),
        os.path.join(package_share, 'models', 'best.pt'),
        os.path.join(source_root, 'models', 'cylinder_yolo_seg.pt'),
        os.path.join(source_root, 'models', 'best.pt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return PathJoinSubstitution([
        FindPackageShare('rins_robot'),
        'models',
        'cylinder_yolo_seg.pt',
    ])


def generate_launch_description():
    default_model = _default_model_path()

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument('show_debug', default_value='true'),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument('max_fps', default_value='5.0'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.45'),
        DeclareLaunchArgument('normalize_radius_m', default_value='0.75'),
        DeclareLaunchArgument('tight_duplicate_radius_m', default_value='0.25'),
        DeclareLaunchArgument('pending_min_hits', default_value='3'),
        DeclareLaunchArgument('publish_min_hits', default_value='6'),
        DeclareLaunchArgument('pending_keep_time_s', default_value='4.0'),
        DeclareLaunchArgument('ground_z_m', default_value='0.0'),
        DeclareLaunchArgument('max_detection_range_m', default_value='2.50'),
        DeclareLaunchArgument('min_cylinder_mask_pixels', default_value='80'),

        Node(
            package='rins_robot',
            executable='cylinder_detection.py',
            name='yolo_cylinder_detector',
            output='screen',
            parameters=[{
                'model_path': ParameterValue(LaunchConfiguration('model_path'), value_type=str),
                'show_debug': ParameterValue(LaunchConfiguration('show_debug'), value_type=bool),
                'device': ParameterValue(LaunchConfiguration('device'), value_type=str),
                'max_fps': ParameterValue(LaunchConfiguration('max_fps'), value_type=float),
                'confidence_threshold': ParameterValue(
                    LaunchConfiguration('confidence_threshold'),
                    value_type=float,
                ),
                'normalize_radius_m': ParameterValue(
                    LaunchConfiguration('normalize_radius_m'),
                    value_type=float,
                ),
                'tight_duplicate_radius_m': ParameterValue(
                    LaunchConfiguration('tight_duplicate_radius_m'),
                    value_type=float,
                ),
                'pending_min_hits': ParameterValue(
                    LaunchConfiguration('pending_min_hits'),
                    value_type=int,
                ),
                'publish_min_hits': ParameterValue(
                    LaunchConfiguration('publish_min_hits'),
                    value_type=int,
                ),
                'pending_keep_time_s': ParameterValue(
                    LaunchConfiguration('pending_keep_time_s'),
                    value_type=float,
                ),
                'ground_z_m': ParameterValue(LaunchConfiguration('ground_z_m'), value_type=float),
                'max_detection_range_m': ParameterValue(
                    LaunchConfiguration('max_detection_range_m'),
                    value_type=float,
                ),
                'min_cylinder_mask_pixels': ParameterValue(
                    LaunchConfiguration('min_cylinder_mask_pixels'),
                    value_type=int,
                ),
            }],
        ),
    ])
