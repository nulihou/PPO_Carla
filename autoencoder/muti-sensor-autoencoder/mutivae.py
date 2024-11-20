import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 95
# Encoder for Multi-Sensor Fusion
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.model_file = os.path.join('autoencoder/model', 'var_encoder_model.pth')

        # Encoder for Camera Input
        self.encoder_layer1_cam = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 79, 39
            nn.LeakyReLU())
        self.encoder_layer2_cam = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        # Encoder for LiDAR Input (e.g., point cloud or image-like grid)
        self.encoder_layer1_lidar = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2),  # Modify input channels for LiDAR data
            nn.LeakyReLU())
        self.encoder_layer2_lidar = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Adjust according to LiDAR data shape
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        # Shared layers for both sensor types
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # 19, 9
            nn.LeakyReLU())
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # 9, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(9 * 4 * 256 * 2, 1024),  # Double the input size for the fused sensors
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x_cam, x_lidar):
        x_cam = x_cam.to(device)
        x_lidar = x_lidar.to(device)

        # Process camera input
        cam_features = self.encoder_layer1_cam(x_cam)
        cam_features = self.encoder_layer2_cam(cam_features)

        # Process LiDAR input
        lidar_features = self.encoder_layer1_lidar(x_lidar)
        lidar_features = self.encoder_layer2_lidar(lidar_features)

        # Fuse the features from both sensors
        fused_features = torch.cat([cam_features, lidar_features], dim=1)  # Concatenate along the channel axis
        fused_features = self.encoder_layer3(fused_features)
        fused_features = self.encoder_layer4(fused_features)
        fused_features = torch.flatten(fused_features, start_dim=1)
        fused_features = self.linear(fused_features)

        # Latent space
        mu = self.mu(fused_features)
        sigma = torch.exp(self.sigma(fused_features))

        # Reparameterization trick
        z = mu + sigma * self.N.sample(mu.shape)

        # Compute KL divergence
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))


# Decoder remains unchanged, as it's based on the latent space
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.model_file = os.path.join('autoencoder/model', 'decoder_model.pth')
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
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
            nn.Sigmoid())

    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))


# Variational Autoencoder with multi-sensor fusion
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join('autoencoder/model', 'var_autoencoder.pth')
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x_cam, x_lidar):
        z = self.encoder(x_cam, x_lidar)
        return self.decoder(z)

    def save(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()


# Training and Testing functions remain unchanged
def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for (x_cam, x_lidar, _) in trainloader:
        x_cam = x_cam.to(device)
        x_lidar = x_lidar.to(device)
        x_hat = model(x_cam, x_lidar)
        loss = ((x_cam - x_hat) ** 2).sum() + model.encoder.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(trainloader.dataset)


def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_cam, x_lidar, _ in testloader:
            x_cam = x_cam.to(device)
            x_lidar = x_lidar.to(device)
            x_hat = model(x_cam, x_lidar)
            loss = ((x_cam - x_hat) ** 2).sum() + model.encoder.kl
            val_loss += loss.item()
    return val_loss / len(testloader.dataset)


# Main function
def main():
    data_dir = r"E:\pythonCarlaPPO\autoencoder\dataset"
    writer = SummaryWriter(f"runs/" + "auto-encoder")

    # Data transformation and augmentation
    train_transforms = transforms.Compose(
        [transforms.RandomRotation(30), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    # Assuming you have multi-sensor data in `train` and `test` folders
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)

    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    # Data loading
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f'Selected device: {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optim)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        val_loss = test(model, validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)
        print(f'EPOCH {epoch + 1}/{NUM_EPOCHS} \t train loss: {train_loss:.3f} \t val loss: {val_loss:.3f}')

    model.save()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')