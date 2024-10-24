# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 데이터셋 다운로드 및 로더 설정
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 제너레이터 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 3, 64, 64)

# 디스크리미네이터 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        return self.main(input.view(-1, 64 * 64 * 3))

# 모델 초기화
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = Generator().to(device)
netD = Discriminator().to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss().to(device)  # LSGAN의 경우 MSE 손실을 사용합니다.
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 훈련 루프
num_epochs = 25
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        # 진짜 데이터 로드
        real_data = data.to(device)
        b_size = real_data.size(0)

        # 디스크리미네이터 훈련
        netD.zero_grad()
        real_output = netD(real_data)
        real_labels = torch.ones(b_size, 1)
        lossD_real = criterion(real_output, real_labels)

        noise = torch.randn(b_size, 100)
        fake_data = netG(noise)
        fake_output = netD(fake_data.detach())
        fake_labels = torch.zeros(b_size, 1)
        lossD_fake = criterion(fake_output, fake_labels)

        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward()
        optimizerD.step()

        # 제너레이터 훈련
        netG.zero_grad()
        fake_output = netD(fake_data)
        lossG = criterion(fake_output, real_labels)
        lossG.backward()
        optimizerG.step()

        # 로그 출력
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(dataloader)}] Loss D: {lossD.item()}, Loss G: {lossG.item()}")
            save_image(fake_data[:16], f"./results/fake_images_epoch_{epoch+1}.png", normalize=True)
    # 매 에포크마다 생성된 이미지를 저장

