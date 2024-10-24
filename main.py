import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from model.cgan import Generator, Discriminator
from utils.dataset import FamilyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # tqdm 추가
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid

# 시드 고정 함수
def seed_everything(seed=115):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# 검증 이미지 저장 (매 1 에포크마다)
def save_images(epoch, generator, valid_loader, device, save_dir="train"):
    generator.eval()
    with torch.no_grad():
        for i, (father_img, mother_img, child_img, gender) in enumerate(valid_loader):
            father_img, mother_img, child_img, gender = father_img.to(device), mother_img.to(device), child_img.to(
                device), gender.to(device)
            generated_img = generator(father_img, mother_img, gender)

            # 부모, 실제 자녀, 예측된 자녀 이미지를 하나로 결합
            combined_images = torch.cat([father_img[:5], mother_img[:5], child_img[:5], generated_img[:5]], dim=0)

            # 4개의 이미지를 수평으로 결합하여 저장
            image_grid = make_grid(combined_images, nrow=5, normalize=True)

            # 이미지 저장
            save_image(image_grid, os.path.join(save_dir, f"generated_epoch_{epoch + 1}.png"))

            break  # 첫 번째 배치만 저장

# Train/Valid loop
def run_epoch(generator, discriminator, loader, criterion, criterion_l1, optimizer_G, optimizer_D, phase='train'):
    running_loss = 0.0
    is_train = phase == 'train'

    if is_train:
        generator.train()
    else:
        generator.eval()

    with torch.set_grad_enabled(is_train):
        loop = tqdm(loader, desc=phase)  # tqdm을 사용하여 로딩 상태 표시
        for i, (father_img, mother_img, child_img, gender) in enumerate(loop):
            father_img, mother_img, child_img, gender = father_img.to(device), mother_img.to(device), child_img.to(device), gender.to(device)

            batch_size = father_img.size(0)

            # 진짜와 가짜 라벨 설정
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            generated_img = generator(father_img, mother_img, gender)

            g_loss_adv  = criterion(discriminator(father_img, mother_img, generated_img), real_labels)
            g_loss_l1 = criterion_l1(generated_img, child_img)
            g_loss = g_loss_adv + g_loss_l1

            if is_train:
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

            # Train/Validate Discriminator
            real_loss = criterion(discriminator(father_img, mother_img, child_img), real_labels)
            fake_loss = criterion(discriminator(father_img, mother_img, generated_img.detach()), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            if is_train:
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            # tqdm으로 현재 진행 상황 표시
            loop.set_postfix(g_loss=g_loss.item(), d_loss=d_loss.item())

            running_loss += (g_loss.item() + d_loss.item()) if is_train else (real_loss.item() + fake_loss.item())

    return running_loss / len(loader)


# 학습 함수 정의
def train_model(generator, discriminator, train_loader, valid_loader, criterion, criterion_l1, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epochs=20):
    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = run_epoch(generator, discriminator, train_loader, criterion, criterion_l1, optimizer_G, optimizer_D, phase='train')
        valid_loss = run_epoch(generator, discriminator, valid_loader, criterion, criterion_l1, optimizer_G, optimizer_D, phase='valid')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        # 가장 좋은 모델 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            print(f"Best model saved with valid loss: {valid_loss:.4f}")
            torch.save(generator.state_dict(), os.path.join("train", "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join("train", "best_discriminator.pth"))

        # 스케줄러 업데이트 (ReduceLROnPlateau는 성능에 따라 학습률 조정)
        scheduler_G.step(valid_loss)  # 검증 손실을 기준으로 학습률 조정
        scheduler_D.step(valid_loss)

        # 매 에포크마다 그래프 저장
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Valid Loss')
        plt.title('Train and Valid Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join("train", f"loss_epoch_{epoch+1}.png"))
        plt.close()

        # 검증 이미지 저장
        save_images(epoch, generator, valid_loader, device)

if __name__ == "__main__":
    if not os.path.exists("train"):
        os.makedirs("train")

    # 시드 고정
    seed_everything(115)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5000

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 손실함수 및 옵티마이저 설정

    criterion = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

    # 스케줄러 설정 (ReduceLROnPlateau)
    scheduler_G = lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)


    # 사용 예시
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FamilyDataset(root_dir="dataset", transform=transform)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 학습 시작
    train_model(generator, discriminator, train_loader, valid_loader, criterion, criterion_l1, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epochs=epochs)