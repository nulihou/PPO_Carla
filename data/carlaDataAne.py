import random
from torch.utils.data import Dataset, DataLoader
import carla
import os
import cv2
import numpy as np
import pandas as pd
from queue import Queue
from PIL import Image
import torch
from torchvision import transforms

def collect_data_fixed_interval(client, output_dir, num_frames=1000,map_name='', interval=1):
    """
    使用 Carla 模拟器的自动驾驶功能和语义分割相机采集数据，包括图像和方向盘角度。
    :param client: Carla 客户端对象
    :param output_dir: 数据保存的目标目录
    :param num_frames: 采集的帧数
    :param interval: 采样时间间隔（秒）
    """

    print(f"\n正在采集地图 {map_name} 的数据...")
    # 连接到 Carla 世界
    world = client.load_world(map_name)
    blueprint_library = world.get_blueprint_library()

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True  # 开启同步模式
    settings.fixed_delta_seconds = 0.1  # 每帧时间间隔为 0.05 秒
    world.apply_settings(settings)

    # 生成车辆和语义分割相机
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())

    if spawn_point is None:
        spawn_point = random.choice(world.get_map().get_spawn_points())
    print(f"生成车辆的起始位置：{spawn_point.location}")
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 启用自动驾驶模式
    # 设置traffic manager
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    # 是否忽略红绿灯
    # tm.ignore_lights_percentage(ego_vehicle, 100)
    # 如果限速30km/h -> 30*(1-10%)=27km/h
    tm.global_percentage_speed_difference(10.0)
    #ego_vehicle.set_autopilot(True, tm.get_port())
    vehicle.set_autopilot(True,tm.get_port())
    print("车辆已启用自动驾驶模式")
    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 摄像头安装在车辆前方
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 设置目录
    map_output_dir = os.path.join(output_dir, map_name)
    os.makedirs(map_output_dir, exist_ok=True)
    image_dir = os.path.join(map_output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    angles_path = os.path.join(map_output_dir, "steering_angles.csv")

    # 定义队列用于同步图像和控制数据
    data_queue = Queue()

    # 定义回调函数
    def image_callback(image):
        # 将语义分割图像转换为 CityScapes 颜色编码
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
        # 降低图像分辨率，减少内存占用
        array_resized = cv2.resize(array, (640, 320))  # 调整分辨率为 640x320
        data_queue.put(array_resized)

    camera.listen(image_callback)

    # 数据采集
    steering_angles = []
    frame_id = 0
    while frame_id < num_frames:
        # Tick 世界以推进模拟
        world.tick()

        # 确保数据队列非空
        while data_queue.empty():
            pass  # 等待新图像数据

        # 获取车辆控制信息
        control = vehicle.get_control()
        print(f"当前控制信息：油门: {control.throttle}, 刹车: {control.brake}, 转向: {control.steer}")
        steering_angle = control.steer

        # 从队列中取出图像数据
        data = data_queue.get()
        if data is not None:
            if frame_id % save_interval == 0:
                # 保存图像
                img_path = os.path.join(image_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(img_path, data)

                # 保存方向盘角度
                steering_angles.append((img_path, steering_angle))
            frame_id += 1

            if frame_id % 100 == 0:
                print(f"已采集 {frame_id}/{num_frames} 帧")

        # 通过帧计数控制采样间隔
        for _ in range(int(interval / settings.fixed_delta_seconds)):
            world.tick()

    # 保存方向盘角度到 CSV 文件
    pd.DataFrame(steering_angles, columns=["image_path", "steering_angle"]).to_csv(angles_path, index=False)
    print(f"数据集已保存到 {output_dir}")

    # 停止传感器并销毁
    if camera is not None and camera.is_alive:
        camera.stop()
        camera.destroy()
        print("摄像头已销毁")

    if vehicle is not None and vehicle.is_alive:
        vehicle.destroy()
        print("车辆已销毁")

    # 关闭同步模式
    settings.synchronous_mode = False
    world.apply_settings(settings)


class SteeringAngleDataset(Dataset):
    """
    自定义 PyTorch 数据集，用于加载 Carla 数据集，包括图像和方向盘角度。
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        初始化数据集。
        :param csv_file: 包含图像路径和方向盘角度的 CSV 文件路径
        :param root_dir: 图像文件的根目录
        :param transform: 图像的预处理变换
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  # 根目录
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()  # 将图片转换为 Tensor 格式
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取索引为 idx 的数据项。
        """
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])  # 拼接路径
        steering_angle = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([steering_angle], dtype=torch.float32)


if __name__ == "__main__":
    # Carla 客户端配置
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # 采集数据
    output_dir = "E:/pythonCarlaPPO/data/output_data"
    num_frames = 2000  # 总帧数
    interval = 1     # 固定时间间隔（秒）
    warmup_time = 5.0  # 自动驾驶模式启动后的预热时间（秒）
    save_interval = 5

    #maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']  # 采集的地图
    #'Town01'com, 'Town02'com, 'Town03'com, 'Town04', 'Town05'com
    tm_port=8000
    map_name='Town04'
    collect_data_fixed_interval(client, output_dir, num_frames, map_name ,interval)

    # 加载并测试 SteeringAngleDataset
    csv_file = os.path.join(output_dir, map_name, "steering_angles.csv")
    image_dir = os.path.join(output_dir, map_name, "images")
    dataset = SteeringAngleDataset(csv_file, image_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("\n测试加载数据集:")
    for batch_idx, (images, angles) in enumerate(dataloader):
        print(f"批次 {batch_idx + 1}:")
        print(f"- 图像张量大小: {images.shape}")
        print(f"- 方向盘角度: {angles.flatten().tolist()}")
        if batch_idx == 2:  # 测试前3批次
            break