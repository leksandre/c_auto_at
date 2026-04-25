import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
  #获取参数
  config = os.path.join(
    get_package_share_directory('los_guidance'),
   'config',
   'params.yaml')
  #创建节点
  node = Node(
    package='los_guidance',
    name='los_guidance_node',
    executable='los_guidance_node',
    output = 'screen',
    parameters=[config]
  )
  ld = LaunchDescription()
  ld.add_action(node)
  return ld