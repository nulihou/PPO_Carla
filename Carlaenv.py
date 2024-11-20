import carla
import numpy as np
import random
import time
from sensors import CameraSensor, CameraSensorEnv, CollisionSensor# 导入传感器相关类
class CarlaEnv:
    def __init__(self,town,continuous_action=True,checkpoint_frequency=100)-> None:
        self.continuous_action_space = True
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world("Town03")
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = None  # Initialize self.vehicle as None
        self.spectator = self.world.get_spectator()
        self.stuck_counter = 0  # Counter to track how long the vehicle has been stuck
        self.max_stuck_count = 5  # Number of steps to consider the vehicle stuck
        self.stuck_threshold = 0.5  # Velocity threshold to consider the vehicle stuck
        self.map = self.world.get_map()
        self.town = town

        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action  # 动作空间是否为连续
        self.display_on = True  # 是否开启视觉显示
        self.current_waypoint_index = 0  # 当前路点索引
        self.checkpoint_waypoint_index = 0  # 检查点路点索引
        self.fresh_start = True  # 是否是全新的开始
        self.checkpoint_frequency = checkpoint_frequency  # 检查点频率
        self.route_waypoints = None  # 路线上的路点列表

        # 需要保持活动的对象
        self.camera_obj = None  # 前置摄像头对象
        self.env_camera_obj = None  # 环境视角摄像头对象
        self.collision_obj = None  # 碰撞检测对象
        self.lane_invasion_obj = None  # 车道入侵检测对象

        # 用于跟踪actor和它们观察结果的重要列表
        self.sensor_list = []  # 传感器列表
        self.actor_list = []  # actor列表

    def reset(self):
        if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.sensor_list.clear()
            self.actor_list.clear()

        # Spawn a new vehicle at the spawn point
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)

        # Set spectator view to follow the vehicle from above
        transform = self.vehicle.get_transform()
        self.total_distance = 750
        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                     carla.Rotation(pitch=-90)))
        self.actor_list.append(self.vehicle)
        start_point = self.spawn_point
        self.end_point = random.choice(self.world.get_map().get_spawn_points())

        # 添加前置摄像头
        self.camera_obj = CameraSensor(self.vehicle)
        while not self.camera_obj.front_camera:  # 等待直到有图像数据
            time.sleep(0.0001)
        self.image_obs = self.camera_obj.front_camera.pop(-1)  # 获取最新图像
        self.sensor_list.append(self.camera_obj.sensor)

        # 如果启用了显示功能，则添加环境视角摄像头
        if self.display_on:
            self.env_camera_obj = CameraSensorEnv(self.vehicle)
            self.sensor_list.append(self.env_camera_obj.sensor)

        # 添加碰撞传感器
        self.collision_obj = CollisionSensor(self.vehicle)
        self.collision_history = self.collision_obj.collision_data
        self.sensor_list.append(self.collision_obj.sensor)

        # 初始化各种参数
        self.timesteps = 0
        self.rotation = self.vehicle.get_transform().rotation.yaw
        self.previous_location = self.vehicle.get_location()
        self.distance_traveled = 0.0
        self.target_speed = 22  # 目标速度 (km/h)
        self.max_speed = 25.0
        self.min_speed = 15.0
        self.max_distance_from_center = 3
        self.throttle = 0.0
        self.previous_steer = 0.0
        self.velocity = 0.0
        self.distance_from_center = 0.0
        self.angle = 0.0
        self.center_lane_deviation = 0.0
        self.distance_covered = 0.0

        if self.fresh_start:
            self.current_waypoint_index = 0
            # Waypoint nearby angle and distance from it
            self.route_waypoints = list()
            self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True,
                                                  lane_type=(carla.LaneType.Driving))
            current_waypoint = self.waypoint
            self.route_waypoints.append(current_waypoint)
            for x in range(self.total_distance):
                if self.town == "Town03":
                    if x < 650:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[-1]
                elif self.town == "Town02":
                    if x < 650:
                        next_waypoint = current_waypoint.next(1.0)[-1]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                else:
                    next_waypoint = current_waypoint.next(1.0)[0]
                self.route_waypoints.append(next_waypoint)
                current_waypoint = next_waypoint
        else:
            # Teleport vehicle to last checkpoint
            waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
            transform = waypoint.transform
            self.vehicle.set_transform(transform)
            self.current_waypoint_index = self.checkpoint_waypoint_index

        self.navigation_obs = np.array(
            [self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])

        time.sleep(0.5)
        self.collision_history.clear()

        self.episode_start_time = time.time()
        return [self.image_obs, self.navigation_obs]

    def step(self, action_idx):
        # 更新时间步数并设置非新鲜开始
        self.timesteps += 1
        self.fresh_start = False
        # 获取车辆速度
        velocity = self.vehicle.get_velocity()
        self.velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6  # 将速度转换为km/h
        # 根据动作空间控制车辆
        if self.continous_action_space:
            # 连续动作空间
            steer = float(action_idx[0])
            steer = max(min(steer, 1.0), -1.0)  # 限制转向值在-1到1之间
            throttle = float((action_idx[1] + 1.0) / 2)
            throttle = max(min(throttle, 1.0), 0.0)  # 限制油门值在0到1之间
            # 应用平滑控制（使用前一时刻的值）
            self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1,
                                                            throttle=self.throttle * 0.9 + throttle * 0.1))
            self.previous_steer = steer
            self.throttle = throttle
        else:
            # 离散动作空间
            steer = self.action_space[action_idx]
            if self.velocity < 20.0:  # 如果速度低于20km/h，则全油门
                self.vehicle.apply_control(
                    carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1, throttle=1.0))
            else:
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1))
            self.previous_steer = steer
            self.throttle = 1.0

        # 处理交通灯
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)  # 将红灯变为绿灯
        # 更新碰撞历史
        self.collision_history = self.collision_obj.collision_data
        # 获取车辆旋转角度
        self.rotation = self.vehicle.get_transform().rotation.yaw
        # 获取车辆位置
        self.location = self.vehicle.get_location()
        # 跟踪最近的路点
        waypoint_index = self.current_waypoint_index
        if self.route_waypoints is None:
            raise ValueError("Route waypoints not initialized")

        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = (waypoint_index + 1) % len(self.route_waypoints)
            wp = self.route_waypoints[next_waypoint_index]
            dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                         self.vector(self.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break

        self.current_waypoint_index = waypoint_index
        # 计算车道中心偏差
        self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
        self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
        self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),
                                                          self.vector(self.next_waypoint.transform.location),
                                                          self.vector(self.location))
        self.center_lane_deviation += self.distance_from_center

        # 计算车辆前进方向与最近路点之间的角度差
        fwd = self.vector(self.vehicle.get_velocity())
        wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
        self.angle = self.angle_diff(fwd, wp_fwd)

        # 更新检查点
        if not self.fresh_start and self.checkpoint_frequency is not None:
            self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency
        # 计算奖励和终止条件
        done = False
        reward = 0
        # 检查是否发生碰撞
        if len(self.collision_history) != 0:
            done = True
            reward = -10
        # 检查是否偏离车道过远
        elif self.distance_from_center > self.max_distance_from_center:
            done = True
            reward = -10
        # 检查是否停滞不前
        elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
            reward = -10
            done = True
        # 检查是否超速
        elif self.velocity > self.max_speed:
            reward = -10
            done = True
        # 中心保持因子
        centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
        # 角度对齐因子
        angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

        if not done:
            if self.continuous_action_space:
                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * centering_factor * angle_factor
                elif self.velocity > self.target_speed:
                    reward = (1.0 - (self.velocity - self.target_speed) / (
                            self.max_speed - self.target_speed)) * centering_factor * angle_factor
                else:
                    reward = 1.0 * centering_factor * angle_factor
            else:
                reward = 1.0 * centering_factor * angle_factor
        # 检查是否达到最大时间步数或完成路线
        if self.timesteps >= 7500:
            done = True
        elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
            done = True
            self.fresh_start = True
            if self.checkpoint_frequency is not None:
                if self.checkpoint_frequency < self.total_distance // 2:
                    self.checkpoint_frequency += 2
                else:
                    self.checkpoint_frequency = None
                    self.checkpoint_waypoint_index = 0
        # 等待直到有新的图像数据
        while len(self.camera_obj.front_camera) == 0:
            time.sleep(0.0001)
        # 获取最新的图像观测
        self.image_obs = self.camera_obj.front_camera.pop(-1)
        # 归一化速度、距离和角度
        normalized_velocity = self.velocity / self.target_speed
        normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
        normalized_angle = abs(self.angle / np.deg2rad(20))
        # 更新导航观测
        self.navigation_obs = np.array(
            [self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center,
             normalized_angle])
        # 如果episode结束，清理资源
        if done:
            self.center_lane_deviation = self.center_lane_deviation / self.timesteps
            self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)

            for sensor in self.sensor_list:
                sensor.destroy()
            self.remove_sensors()
            for actor in self.actor_list:
                actor.destroy()

        return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]
    def get_state(self):
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        return np.array([location.x, location.y, velocity.x, velocity.y])

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()

    def get_world(self) -> object:
            return self.world
        # 用于获取模拟器当前所在的世界
    def get_world(self) -> object:
            return self.world
     # 用于获取模拟器的蓝图库
    def get_blueprint_library(self) -> object:
            return self.world.get_blueprint_library()
    def angle_diff(self, v0, v1):
            angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle <= -np.pi:
                angle += 2 * np.pi
            return angle
        # 计算点到线段的距离
    def distance_to_line(self, A, B, p):
            num = np.linalg.norm(np.cross(B - A, A - p))
            denom = np.linalg.norm(B - A)
            if np.isclose(denom, 0):
                return np.linalg.norm(p - A)
            return num / denom
        # 将CARLA的Location或Vector3D对象转换为NumPy数组
    def vector(self, v):
            if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
                return np.array([v.x, v.y, v.z])
            elif isinstance(v, carla.Rotation):
                return np.array([v.pitch, v.yaw, v.roll])
        # 获取指定名称的车辆蓝图，并随机设置颜色
    def get_vehicle(self, vehicle_name):
            blueprint = self.blueprint_library.filter(vehicle_name)[0]
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            return blueprint
        # 在环境中生成车辆
    def set_vehicle(self, vehicle_bp, spawn_points):# 从给定的生成点列表中随机选择一个生成点，如果没有提供则默认为空变换
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

    def remove_sensors(self):
            self.camera_obj = None
            self.collision_obj = None
            self.lane_invasion_obj = None
            self.env_camera_obj = None
            self.front_camera = None
            self.collision_history = None
            self.wrong_maneuver = None
    def get_discrete_action_space(self):
        action_space = \
            np.array([
                -0.50,
                -0.30,
                -0.10,
                0.0,
                0.10,
                0.30,
                0.50
            ])
        return action_space


