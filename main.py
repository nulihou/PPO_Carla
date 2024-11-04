import logging
import os
import pickle
import time
import random
import torch
import sys
import numpy as np
from Carlaenv import CarlaEnv
from encoder_init import EncodeState
from datetime import datetime
from PPO_agent import PPOAgent
import argparse
from parameters import *
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='实验名称')
    parser.add_argument('--env-name', type=str, default='carla', help='仿真环境名称，默认为carla')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='优化器的学习率')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='实验总步数')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='初始探索噪声')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='测试模型时的步数')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='每集的最大步数')
    parser.add_argument('--train', default=True, type=boolean_string, help='是否进行训练？')
    parser.add_argument('--town', type=str, default="Town03", help='选择哪个城镇')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='是否加载检查点继续训练？')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,help='如果启用，则`torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,help='如果启用，则默认使用CUDA')
    args = parser.parse_args()
    return args


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('不是一个有效的布尔字符串')
    return s == 'True'

def runner():
    # ========================================================================
    #                           基本参数和日志设置
    # ========================================================================

    # 解析命令行参数
    args = parse_args()
    exp_name = args.exp_name  # 实验名称
    train = args.train  # 是否训练
    town = args.town  # 选择的城镇
    checkpoint_load = args.load_checkpoint  # 是否加载检查点
    total_timesteps = args.total_timesteps  # 总步数
    action_std_init = args.action_std_init  # 初始动作标准差

    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """

            这里可以扩展到不同的算法。

            """
            sys.exit()
    except Exception as e:
        print(e.message)
        sys.exit()
    # 根据是否训练来设置TensorBoard的日志路径

    if train == True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))
    # 记录超参数

    # 设置随机种子以重现结果
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    action_std_decay_rate = 0.05
    min_action_std = 0.05
    action_std_decay_freq = 5e5
    timestep = 0  # 当前时间步
    episode = 0  # 当前集数
    cumulative_score = 0  # 累积分数
    episodic_length = list()  # 每集长度列表
    scores = list()  # 分数列表
    deviation_from_center = 0  # 偏离中心的距离
    distance_covered = 0  # 覆盖的距离

    # ========================================================================
    #                           创建仿真
    # ========================================================================


    # 根据是否训练来初始化环境
    if train:
        env = CarlaEnv(town,continuous_action=True)
    else:
        env = CarlaEnv(town,checkpoint_frequency=None)
    encode = EncodeState(LATENT_DIM)

    # 检查并创建目录
    checkpoint_dir = f'preTrained_models/ppo/{town}'
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)  # 创建目录

    # 检查目录是否存在
    if os.path.exists(checkpoint_dir):
        # 获取目录下的文件列表
        try:
            files = next(os.walk(checkpoint_dir))[2]  # 获取文件列表
            chkt_file_nums = len(files)  # 计算文件数量
        except StopIteration:
            chkt_file_nums = 0  # 如果目录为空，计数为0
    else:
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        chkt_file_nums = 0  # 如果目录不存在，计数为0

    print(f"Number of checkpoint files: {chkt_file_nums}")
    # ========================================================================
    #                           ALGORITHM
    # ========================================================================
    try:
        time.sleep(0.5)
        if checkpoint_load:  # 如果需要加载检查点
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1  # 获取最新的检查点文件编号
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_{chkt_file_nums}.pickle'  # 构建检查点文件路径
            with open(chkpt_file, 'rb') as f:  # 打开并读取检查点文件
                data = pickle.load(f)
                episode = data['episode']  # 加载集数
                timestep = data['timestep']  # 加载时间步
                cumulative_score = data['cumulative_score']  # 加载累积分数
                action_std_init = data['action_std_init']  # 加载动作标准差
            agent = PPOAgent(town, action_std_init)  # 初始化PPO代理
            agent.load()  # 加载代理模型
        else:
            if train == False:  # 如果不进行训练（即测试）
                agent = PPOAgent(town, action_std_init)  # 初始化PPO代理
                agent.load()  # 加载代理模型
                for params in agent.old_policy.actor.parameters():  # 冻结策略网络参数
                    params.requires_grad = False
            else:
                agent = PPOAgent(town, action_std_init)  # 初始化PPO代理

        if train:  # 训练模式

        #encode = EncodeState(LATENT_DIM)
            while timestep < total_timesteps:  # 直到达到总时间步
                observation = env.reset()  # 重置环境
                observation = encode.process(observation)  # 处理观察

                current_ep_reward = 0  # 当前集奖励
                t1 = datetime.now()  # 记录开始时间

                for t in range(args.episode_length):  # 每集的最大步数
                    # 选择动作
                    action = agent.get_action(observation, train=True)

                    observation, reward, done, info = env.step(action)  # 执行动作

                    if observation is None:
                            break  # 如果观察为空，则跳出循环
                    observation = encode.process(observation)  # 处理新的观察

                    agent.memory.rewards.append(reward)  # 存储奖励
                    agent.memory.dones.append(done)  # 存储是否结束标志

                    timestep += 1  # 更新时间步
                    current_ep_reward += reward  # 累加奖励
                    if timestep % action_std_decay_freq == 0:  # 动作标准差衰减
                        action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep == total_timesteps - 1:  # 如果是最后一个时间步
                        agent.chkpt_save()  # 保存检查点

                    if done:  # 如果集结束
                        episode += 1  # 更新集数
                        t2 = datetime.now()  # 记录结束时间
                        t3 = t2 - t1  # 计算集长度
                        episodic_length.append(abs(t3.total_seconds()))  # 存储集长度
                        break  # 跳出循环
                deviation_from_center += info[1]  # 累加偏离中心的距离
                distance_covered += info[0]  # 累加覆盖的距离

                scores.append(current_ep_reward)  # 存储当前集奖励

                if checkpoint_load:  # 如果加载了检查点
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)  # 更新累积分数
                else:
                    cumulative_score = np.mean(scores)  # 计算平均分数

                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep),
                      ', Reward:  {:.2f}'.format(current_ep_reward),
                      ', Average Reward:  {:.2f}'.format(cumulative_score))

                if episode % 10 == 0:  # 每10集学习一次
                    agent.learn()  # 学习
                    agent.chkpt_save()  # 保存检查点
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -= 1
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_{chkt_file_nums}.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep,
                                'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)

                if episode % 5 == 0:  # 每5集记录一次TensorBoard数据
                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5:]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5:]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center / 5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered / 5, timestep)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:  # 每100集保存一次模型
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_{chkt_file_nums}.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep,
                                'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)

            print("Terminating the run.")
            sys.exit()
        else:
            #Testing
            while timestep < args.test_timesteps:  # 直到达到测试时间步
                observation = env.reset()  # 重置环境
                observation = encode.process(observation)  # 处理观察

                current_ep_reward = 0  # 当前集奖励
                t1 = datetime.now()  # 记录开始时间
                for t in range(args.episode_length):  # 每集的最大步数
                    # 选择动作
                    action = agent.get_action(observation, train=False)

                    observation, reward, done, info = env.step(action)  # 执行动作
                    if observation is None:
                        break  # 如果观察为空，则跳出循环
                    observation = encode.process(observation)  # 处理新的观察

                    timestep += 1  # 更新时间步
                    current_ep_reward += reward  # 累加奖励

                    if done:  # 如果集结束
                        episode += 1  # 更新集数
                        t2 = datetime.now()  # 记录结束时间
                        t3 = t2 - t1  # 计算集长度
                        episodic_length.append(abs(t3.total_seconds()))  # 存储集长度
                        break  # 跳出循环

                deviation_from_center += info[1]  # 累加偏离中心的距离
                distance_covered += info[0]  # 累加覆盖的距离

                scores.append(current_ep_reward)  # 存储当前集奖励
                cumulative_score = np.mean(scores)  # 计算平均分数

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))

                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()

    finally:
        sys.exit()


if __name__ == "__main__":
    try:
         runner()
    except KeyboardInterrupt:
        print('Training interrupted.')
    finally:
        print('\nExit')