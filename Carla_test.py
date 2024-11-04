import time
import carla
import random
# 创建一个空列表来存储所有的actor对象，方便后续销毁
actor_list = []

try:
    client= carla.Client('localhost', 2000) # 创建CARLA客户端，连接到主机为localhost，端口为2000的CARLA服务器
    client.set_timeout(3.0) # 设置客户端的超时时间为3秒，防止连接超时

    world = client.load_world("Town03")# 如果需要加载特定的地图，可以取消注释这一行
    #world = client.get_world()# 获取当前世界的引用，而不是加载新的世界
    # 获取当前世界的蓝图库，包含所有可用的车辆、行人等模型
    blueprint_library = world.get_blueprint_library()
    # 从蓝图库中选择一个特斯拉Model 3的蓝图
    v_bp = blueprint_library.filter("model3")[0]
    # 随机选择一个车辆的起始位置
    spawn_point = random.choice(world.get_map().get_spawn_points())
    # 在选定的起始位置生成一辆车
    vehicle = world.spawn_actor(v_bp, spawn_point)

    # 将生成的车辆添加到actor_list中，以便后续销毁
    actor_list.append(vehicle)

    # 应用控制命令，使车辆全速前进，不转向
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                            carla.Rotation(pitch=-90)))


    # 暂停30秒，让车辆有足够的时间行驶
    time.sleep(30)
finally:
    for actor in actor_list:
        actor.destroy()
    print("结束")