import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Получаем путь к директории share вашего нового пакета
    pkg_share = get_package_share_directory('los_guidance')
    
    # Путь к файлу параметров
    config_file = os.path.join(pkg_share, 'config', 'params.yaml')

    # Создаем ноду
    los_node = Node(
        package='los_guidance',
        executable='los_guidance_node',  # УКАЖИТЕ ТОЧНОЕ ИМЯ ИСПОЛНЯЕМОГО ФАЙЛА из CMakeLists.txt
        name='los_guidance_node',
        output='screen',
        parameters=[config_file],
        # Если нужно переопределить параметры прямо здесь, раскомментируйте:
        # parameters=[
        #     config_file,
        #     {"kp_yaw": 3.0, "max_xy_accel": 8.0}
        # ]
    )

    ld = LaunchDescription()
    ld.add_action(los_node)
    return ld
