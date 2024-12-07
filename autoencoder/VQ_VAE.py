import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torchvision import datasets
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NUM_EPOCHS = 50  # 训练轮数
BATCH_SIZE = 64  # 批次大小
LEARNING_RATE = 1e-5  # 学习率
LATENT_SPACE = 64  # 潜在空间维度
KL_ANNEALING_RATE = 0.001  # KL损失权重增加的速度
BETA = 0.1  # VQ损失权重
PRED_LOSS_WEIGHT = 0.1  # 预测损失权重
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class imageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((320, 640)),  # 调整图像大小
            transforms.ToTensor()  # 转换为张量
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # 获取图像的完整路径
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        #print(f"Shape of preprocessed image: {image.shape}")
        return image


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        return quantized, loss


class VQEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VQEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()
        # 修改全连接层的输入维度
        self.linear = nn.Linear(19 * 39 * 256, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.linear(x)


class VQDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(VQDecoder, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 1024),  # 潜在空间映射到较大特征
            nn.LeakyReLU(),
            nn.Linear(1024, 256 * 5 * 10),  # 映射到初始特征图大小
            nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 5, 10))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, output_padding=1),  # 保证到达目标尺寸
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(f"Shape of decoder input data: {x.shape}")  # Add this line
        x = self.decoder_linear(x)  # Pass through linear layers
        #print(f"Shape of decoded data: {x.shape}")  # Add this line
        x = self.unflatten(x)  # Reshape for transposed conv layers
        #print(f"Shape of unflatten data: {x.shape}")  # Add this line
        x = self.decoder(x)  # Transposed convolutions to upsample to the original size
        #print(f"Shape of decoder data: {x.shape}")  # Add this line
        x = F.interpolate(x, size=(320, 640), mode='bilinear', align_corners=False)
        return x


class VQVAE(nn.Module):
    def __init__(self, latent_dims, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.encoder = VQEncoder(latent_dims)
        self.decoder = VQDecoder(latent_dims)
        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dims)

    def forward(self, x):

        encoded = self.encoder(x)
        quantized, vq_loss = self.vq_layer(encoded)
        decoded = self.decoder(quantized)

        return decoded, vq_loss

    def save(self, path):
        """Save the model's state_dict to the specified path."""
        torch.save(self.state_dict(), path)
        print(f"Model saved successfully to {path}")

    @staticmethod
    def load(path, latent_dims, num_embeddings, embedding_dim, device):
        """Load the model from the specified path."""
        model = VQVAE(latent_dims, num_embeddings, embedding_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded successfully from {path}")
        return model


def train(model, trainloader, optim):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optim.zero_grad()
        decoded, vq_loss = model(data)
        recon_loss = F.mse_loss(decoded, data)
        total_loss = recon_loss + vq_loss

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(
                f"Batch {batch_idx}/{len(trainloader)}: recon_loss = {recon_loss.item():.4f}, vq_loss = {vq_loss.item():.4f}, total_loss = {total_loss.item():.4f}")

        total_loss.backward()
        optim.step()
        running_loss += total_loss.item()

    epoch_loss = running_loss / len(trainloader)
    return epoch_loss


def main():
    data_dir = r"E:\pythonCarlaPPO\autoencoder\dataset"
    writer = SummaryWriter(f"runs/" + "vqvae-visualization")

    transform = transforms.Compose([transforms.ToTensor()])

    merged_images_csv = "E:/pythonCarlaPPO/data/merged_images.csv"
    train_data = imageDataset(merged_images_csv, transform=transform)

    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = VQVAE(latent_dims=LATENT_SPACE, num_embeddings=512, embedding_dim=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print(f'Selected device : {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optim)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        print(
            '\nEPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, NUM_EPOCHS, train_loss))

    model.save("vqvae_model.pth")
    print("VQ-VAE model saved successfully.")

    return model

if __name__ == "__main__":
    main()
