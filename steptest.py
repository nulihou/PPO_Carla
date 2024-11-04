import carla
import time
from Carlaenv import CarlaEnv

def test_step_function():
    # 创建 CarlaEnv 实例
    env = CarlaEnv()

    # 重置环境
    state = env.reset()
    print("Initial State:", state)

    # 假设你已经定义了一些动作（可以是离散的或连续的）
    # 这里我们使用一个简单的示例：连续动作（steer, throttle）
    action = [0.0, 0.5]  # 例如，0.0 代表不转向，0.5 代表油门

    # 进行若干个步骤
    for _ in range(10):
        next_state, reward, done, info = env.step(action)
        print("Next State:", next_state)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

        # 如果已经结束，重置环境
        if done:
            state = env.reset()
            print("Environment reset. New State:", state)

        # 等待一小段时间以便观察效果
        time.sleep(0.5)

    # 关闭环境
    env.close()


if __name__ == "__main__":
    test_step_function()
