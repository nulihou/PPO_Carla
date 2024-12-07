import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch


class ImageDataset(Dataset):
    """
    自定义 PyTorch 数据集，用于加载 Carla 数据集，包括图像（不包括方向盘角度）。
    """

    def __init__(self, csv_file, transform=None):
        """
        初始化数据集。
        :param csv_file: 包含图像路径的 CSV 文件路径
        :param transform: 图像的预处理变换
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()  # 将图片转换为 Tensor 格式
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取索引为 idx 的数据项。
        """
        img_path = self.data.iloc[idx, 0]  # 获取图像的完整路径
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def merge_datasets(output_dir, maps, output_csv):
    """
    合并多个地图的数据集为一个完整的 CSV 文件（只包含图像路径）。
    :param output_dir: 数据存储的根目录
    :param maps: 包含地图名称的列表
    :param output_csv: 合并后的输出 CSV 文件路径
    """
    all_image_paths = []  # 用于存储所有图像路径的列表

    # 遍历每个地图
    for map_name in maps:
        map_output_dir = os.path.join(output_dir, map_name)  # 当前地图的文件夹路径
        angles_path = os.path.join(map_output_dir, "steering_angles.csv")  # CSV 文件路径

        # 检查文件是否存在
        if os.path.exists(angles_path):
            print(f"正在处理地图 {map_name} 的数据...")

            # 读取该地图的 steering_angles.csv 文件
            steering_angles = pd.read_csv(angles_path)

            # 生成图像的完整路径，拼接成 "地图文件夹/images/图像文件名"
            steering_angles['image_path'] = steering_angles['image_path'].apply(
                lambda x: os.path.join(map_output_dir, 'images', os.path.basename(x))
            )

            # 只保留图像路径列
            image_paths = steering_angles[['image_path']]
            all_image_paths.append(image_paths)
        else:
            print(f"警告: 地图 {map_name} 的数据文件不存在：{angles_path}")

    # 合并所有地图的数据
    if all_image_paths:
        merged_data = pd.concat(all_image_paths, ignore_index=True)
        merged_data.to_csv(output_csv, index=False)
        print(f"所有数据已合并并保存到 {output_csv}")
    else:
        print("没有有效的数据可以合并。")


if __name__ == "__main__":
    output_dir = "E:/pythonCarlaPPO/data/output_data"  # 数据存储的根目录
    maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']  # 地图名称列表
    output_csv = "E:/pythonCarlaPPO/data/merged_images.csv"  # 合并后的输出 CSV 文件路径

    # 合并数据
    merge_datasets(output_dir, maps, output_csv)

    # 加载并测试合并后的数据集
    dataset = ImageDataset(output_csv)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("\n测试加载数据集: ")
    for batch_idx, images in enumerate(dataloader):
        print(f"批次 {batch_idx + 1}:")
        print(f"- 图像张量大小: {images.shape}")
        if batch_idx == 2:  # 测试前3批次
            break
