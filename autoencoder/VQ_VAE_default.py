import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NUM_EPOCHS = 50  # 训练轮数
BATCH_SIZE = 64  # 批次大小
LEARNING_RATE = 1e-5  # 学习率
LATENT_SPACE = 64  #潜在空间维度设置为 95 可能过大，导致模型倾向于记忆而不是泛化特征。我们可以尝试将其减少到 32 或 64。
KL_ANNEALING_RATE = 0.001  # KL损失权重增加的速度
BETA = 0.1 # VQ损失权重
PRED_LOSS_WEIGHT = 0.1  # 预测损失权重
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 预测器，用于生成方向盘角度预测
class Predictor(nn.Module):
    def __init__(self, latent_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出方向盘角度
        )

    def forward(self, z):
        return self.fc(z)

# 定义离散代码簇嵌入层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 初始化嵌入矩阵
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # 输入: (batch, feature_dim)
        flat_x = x.view(-1, self.embedding_dim)

        # 计算每个编码器输出到所有嵌入向量的距离
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # 查找最近的嵌入向量
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 提取嵌入向量并重塑为输入形状
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 返回量化后的向量和损失
        quantized = x + (quantized - x).detach()  # 提供梯度给编码器
        return quantized, loss


# 编码器（Encoder）结构
class VQEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VQEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 79x39
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40x20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 19x9
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2),  # 9x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(9 * 4 * 256, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.linear(x)


# 解码器（Decoder）结构
class VQDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(VQDecoder, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 9 * 4 * 256),
            nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 9))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        return self.decoder(x)


# VQ-VAE 模型
class VQVAE(nn.Module):
    def __init__(self,latent_dims, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.encoder = VQEncoder(latent_dims)
        self.decoder = VQDecoder(latent_dims)
        self.predictor = Predictor(latent_dims)  # 新增的预测器
        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dims)
        self.model_file = r'E:\pythonCarlaPPO\autoencoder\model\VQ_VAE_Model\vqvae_model.pth'
        self.encoder_model_file=r'E:\pythonCarlaPPO\autoencoder\model\VQ_VAE_Model\VQ_encoder.pth'
        self.decoder_model_file=r'E:\pythonCarlaPPO\autoencoder\model\VQ_VAE_Model\VQ_decoder.pth'
    def forward(self, x):
        # 编码数据并量化
        encoded = self.encoder(x)
        quantized, vq_loss = self.vq_layer(encoded)
        # 解码数据
        decoded = self.decoder(quantized)
        # 预测方向盘角度
        predicted_angle = self.predictor(encoded)
        return decoded, vq_loss,predicted_angle
    """
    提取潜变量特征供 PPO 使用
    """
    def extract_features(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            quantized, _ = self.vq_layer(encoded)
        return quantized

    def save(self):
        model_dir = os.path.dirname(self.model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), self.model_file)
        torch.save(self.encoder.state_dict(), self.encoder_model_file)
        torch.save(self.decoder.state_dict(), self.decoder_model_file)
        print(f"Saving model to: {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        # Load encoder and decoder weights individually
        self.encoder.load_state_dict(torch.load(self.encoder_model_file))
        self.decoder.load_state_dict(torch.load(self.decoder_model_file))
        print("Model weights loaded successfully.")


# 定义损失函数，结合重建误差、KL 散度和预测损失
def calculate_loss(x, x_hat, vq_loss, predicted_angle, true_angle):
    # 重建误差，衡量解码图像和原始输入图像的差异
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL散度损失，强制潜变量接近高斯先验分布
    kl_loss = torch.mean(0.5 * torch.sum(predicted_angle ** 2, dim=1))
    # 预测损失，用于优化方向盘角度预测的准确性
    pred_loss = F.mse_loss(predicted_angle, true_angle)
    # 总损失，结合各部分权重
    total_loss = recon_loss + BETA * vq_loss + KL_ANNEALING_RATE * kl_loss + PRED_LOSS_WEIGHT * pred_loss
    return total_loss
#训练过程
def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for x, y in trainloader:  # x: 图像, y: 方向盘角度
        x, y = x.to(device), y.to(device)
        x_hat, vq_loss, _, predicted_angle = model(x)
        loss = calculate_loss(x, x_hat, vq_loss, predicted_angle, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(trainloader.dataset)


# 验证过程
def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            x_hat, vq_loss, _, predicted_angle = model(x)
            loss = calculate_loss(x, x_hat, vq_loss, predicted_angle, y)
            val_loss += loss.item()
    return val_loss / len(testloader.dataset)

def main():
    data_dir = r"E:\pythonCarlaPPO\autoencoder\dataset"
    writer = SummaryWriter(f"runs/" + "vqvae-visualization")

    # Applying Transformation
    train_transforms = transforms.Compose(
        [transforms.RandomRotation(30), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)

    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    # Data Loading
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = VQVAE(latent_dims=LATENT_SPACE, num_embeddings=512, embedding_dim=64).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print(f'Selected device : {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optim)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        val_loss = test(model, validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)
        print(
            '\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS, train_loss, val_loss))

    model.save()
    print("VQ-VAE model saved successfully.")

    # 提供潜变量提取接口
    return model

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
# import pandas as pd
# from PIL import Image
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from torch.utils.data import DataLoader, random_split
# from torch import optim
# from torchvision import transforms
#
# # 设置训练的超参数
# NUM_EPOCHS = 50  # 训练轮数
# BATCH_SIZE = 64  # 批次大小
# LEARNING_RATE = 1e-5  # 学习率
# LATENT_SPACE = 64  # 潜在空间维度设置为 64
# KL_ANNEALING_RATE = 0.001  # KL损失权重增加的速度
# BETA = 0.1  # VQ损失权重
# PRED_LOSS_WEIGHT = 0.1  # 预测损失权重
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置设备为 GPU 或 CPU
#
#
# # 自定义数据集类，用于加载合并后的 CSV 数据
# class SteeringAngleDataset(torch.utils.data.Dataset):
#     def __init__(self, csv_file, transform=None):
#         """
#         初始化数据集。
#         :param csv_file: 包含图像路径和方向盘角度的 CSV 文件路径
#         :param transform: 图像的预处理变换
#         """
#         self.data = pd.read_csv(csv_file)  # 读取 CSV 文件
#         self.transform = transform if transform else transforms.Compose([
#             transforms.ToTensor()  # 默认转换：将图片转换为 Tensor 格式
#         ])
#
#     def __len__(self):
#         return len(self.data)  # 返回数据集大小
#
#     def __getitem__(self, idx):
#         """
#         获取索引为 idx 的数据项。
#         """
#         img_path = self.data.iloc[idx, 0]  # 获取图像的完整路径
#         steering_angle = self.data.iloc[idx, 1]  # 获取方向盘角度
#         print(f"Image Path: {img_path}, Steering Angle: {steering_angle}")  # 检查数据
#         image = Image.open(img_path).convert('RGB')  # 加载图片并转为 RGB 格式
#         if self.transform:
#             image = self.transform(image)  # 对图像进行预处理
#         return image, torch.tensor([steering_angle], dtype=torch.float32)  # 返回图像和方向盘角度
#
#
# # 另一个数据集类，用于加载图像目录和角度标签
# class CarDataset(Dataset):
#     def __init__(self, image_dir, angle_file, transform=None):
#         self.image_dir = image_dir
#         self.angle_file = angle_file
#         self.transform = transform
#
#         # 加载方向盘角度标签
#         with open(angle_file, 'r') as f:
#             self.angles = [float(line.strip()) for line in f.readlines()]
#
#         # 获取图像路径
#         self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
#
#     def __len__(self):
#         return len(self.image_paths)  # 返回图像数量
#
#     def __getitem__(self, idx):
#         # 加载图像
#         img = Image.open(self.image_paths[idx]).convert('RGB')
#
#         # 加载方向盘角度标签
#         angle = self.angles[idx]
#
#         if self.transform:
#             img = self.transform(img)
#
#         return img, torch.tensor(angle, dtype=torch.float32)  # 返回图像和角度
#
#
# # 预测器，用于生成方向盘角度预测
# class Predictor(nn.Module):
#     def __init__(self, latent_dim):
#         super(Predictor, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 128),  # 输入是潜在空间的维度，输出是 128 维
#             nn.ReLU(),
#             nn.Linear(128, 1)  # 输出方向盘角度
#         )
#
#     def forward(self, z):
#         return self.fc(z)  # 计算预测的方向盘角度
#
#
# # 定义离散代码簇嵌入层，用于 VQ-VAE 的量化操作
# class VectorQuantizer(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
#         super(VectorQuantizer, self).__init__()
#         self.embedding_dim = embedding_dim  # 嵌入空间的维度
#         self.num_embeddings = num_embeddings  # 嵌入的数量
#         self.commitment_cost = commitment_cost  # 承诺损失系数
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)  # 嵌入层
#         self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)  # 初始化嵌入权重
#
#     def forward(self, x):
#         flat_x = x.view(-1, self.embedding_dim)  # 将输入展平
#         distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
#                      + torch.sum(self.embedding.weight ** 2, dim=1)
#                      - 2 * torch.matmul(flat_x, self.embedding.weight.t()))  # 计算输入与嵌入之间的距离
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # 找到距离最近的嵌入
#         encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
#         encodings.scatter_(1, encoding_indices, 1)  # one-hot 编码
#         quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)  # 获取量化后的结果
#         e_latent_loss = F.mse_loss(quantized.detach(), x)  # 计算嵌入向量和输入之间的重建误差
#         q_latent_loss = F.mse_loss(quantized, x.detach())  # 计算量化误差
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss  # 总损失
#         quantized = x + (quantized - x).detach()  # 提供梯度给编码器
#         return quantized, loss  # 返回量化结果和损失
#
#
# # 编码器（Encoder）结构
# class VQEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(VQEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
#             nn.Conv2d(64, 128, 4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(128, 256, 3, stride=2),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(),
#         )
#         self.flatten = nn.Flatten()
#
#     def forward(self, x):
#         x = self.encoder(x)
#         print(f"Shape after encoder: {x.shape}")  # 调试输出
#         x = self.flatten(x)
#         print("Shape after flattening:", x.shape)
#         return x
#
#
# # 解码器（Decoder）结构
# class VQDecoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(VQDecoder, self).__init__()
#         self.decoder_linear = nn.Sequential(
#             nn.Linear(latent_dim, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, 256 * 40 * 80),  # 这里调整输出尺寸的大小
#             nn.LeakyReLU()
#         )
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 40, 80))  # 解码时恢复图像大小
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.decoder_linear(x)
#         x = self.unflatten(x)  # 恢复为图片大小
#         x_hat = self.decoder(x)  # 通过解码器输出重建的图像
#
#         # 如果尺寸不匹配，使用插值调整大小
#         if x_hat.size(2) != 320 or x_hat.size(3) != 640:
#             x_hat = F.interpolate(x_hat, size=(320, 640), mode='bilinear', align_corners=False)
#         return x_hat
#
#
# # VQ-VAE 模型
# class VQVAE(nn.Module):
#     def __init__(self, latent_dims, num_embeddings, embedding_dim):
#         super(VQVAE, self).__init__()
#         self.encoder = VQEncoder(latent_dims)
#         self.decoder = VQDecoder(latent_dims)
#         self.predictor = Predictor(latent_dims)  # 新增的预测器
#         self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dims)
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         quantized, vq_loss = self.vq_layer(encoded)
#         decoded = self.decoder(quantized)
#         predicted_angle = self.predictor(encoded)
#         return decoded, vq_loss, predicted_angle
#
#     """
#     #     提取潜变量特征供 PPO 使用
#     #     """
#     def extract_features(self, x):
#         with torch.no_grad():
#             encoded = self.encoder(x)
#             quantized, _ = self.vq_layer(encoded)
#         return quantized
#
#     def save(self):
#         model_dir = os.path.dirname(self.model_file)
#         if not os.path.exists(model_dir):
#             os.makedirs(model_dir, exist_ok=True)
#         torch.save(self.state_dict(), self.model_file)
#         torch.save(self.encoder.state_dict(), self.encoder_model_file)
#         torch.save(self.decoder.state_dict(), self.decoder_model_file)
#         print(f"Saving model to: {self.model_file}")
#
#     def load(self):
#         self.load_state_dict(torch.load(self.model_file))
#         # Load encoder and decoder weights individually
#         self.encoder.load_state_dict(torch.load(self.encoder_model_file))
#         self.decoder.load_state_dict(torch.load(self.decoder_model_file))
#         print("Model weights loaded successfully.")
#
#
# def calculate_loss(x, x_hat, vq_loss, predicted_angle, true_angle, BETA=0.25, KL_ANNEALING_RATE=0.001, PRED_LOSS_WEIGHT=0.1):
#     # 重建误差：衡量解码图像和原始输入图像的差异
#     recon_loss = F.mse_loss(x_hat, x, reduction='sum')
#
#     # 预测损失，用于优化方向盘角度预测的准确性
#     pred_loss = F.mse_loss(predicted_angle, true_angle)
#
#     # 总损失，结合各部分权重
#     total_loss = recon_loss + BETA * vq_loss + KL_ANNEALING_RATE * recon_loss + PRED_LOSS_WEIGHT * pred_loss
#     return total_loss
#
#
# def train(model, trainloader, optim, device):
#     model.train()
#     train_loss = 0.0
#     for x, y in trainloader:  # x: 图像, y: 方向盘角度
#         x, y = x.to(device), y.to(device)
#
#         # 前向传播
#         x_hat, vq_loss, predicted_angle = model(x)  # 模型输出图像重建，vq损失和预测角度
#
#         # 计算总损失
#         loss = calculate_loss(x, x_hat, vq_loss, predicted_angle, y)
#
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         train_loss += loss.item()  # 累加训练损失
#
#     return train_loss / len(trainloader.dataset)  # 返回平均训练损失
#
#
# def test(model, testloader, device):
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for x, y in testloader:
#             x, y = x.to(device), y.to(device)
#
#             # 前向传播
#             x_hat, vq_loss, predicted_angle = model(x)  # 模型输出图像重建，vq损失和预测角度
#
#             # 计算总损失
#             loss = calculate_loss(x, x_hat, vq_loss, predicted_angle, y)  # 计算总损失，包括重建误差、VQ损失和预测损失
#
#             val_loss += loss.item()  # 累加验证损失
#
#     return val_loss / len(testloader.dataset)  # 返回平均验证损失
#
#
#
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     csv_file = "E:/pythonCarlaPPO/data/merged_steering_angles.csv"  # 合并后的 CSV 数据路径
#     writer = SummaryWriter(f"runs/" + "vqvae-visualization")
#
#     train_transforms = transforms.Compose(
#         [transforms.RandomRotation(30), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
#     test_transforms = transforms.Compose([transforms.ToTensor()])
#
#     # 使用合并后的 CSV 数据集
#     full_dataset = SteeringAngleDataset(csv_file=csv_file, transform=train_transforms)
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
#
#     trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     model = VQVAE(latent_dims=LATENT_SPACE, num_embeddings=512, embedding_dim=64).to(device)
#     optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
#     for epoch in range(NUM_EPOCHS):
#         train_loss = train(model, trainloader, optim, device)
#         val_loss = test(model, testloader)
#         print(f"Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#         writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
#
# if __name__ == "__main__":
#         main()
