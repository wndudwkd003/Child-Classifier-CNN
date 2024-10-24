import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from model.cgan import Generator, Discriminator
from utils.dataset_v2 import FamilyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # tqdm 추가
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import csv

# 시드 고정 함수
def seed_everything(seed=115):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# 검증 이미지 저장 (매 1 에포크마다)
def save_images(epoch, generator, valid_loader, device, save_dir="new_gen"):
    generator.eval()
    with torch.no_grad():
        for i, (father_img, mother_img, real_child_img, other_img, child_gender, other_gender, _) in enumerate(valid_loader):
            father_image, mother_image, child_image, _, child_gender, _ = father_img.to(device), mother_img.to(device), real_child_img.to(device), other_img.to(device), child_gender.to(device),  other_gender.to(device)

            generated_img = generator(father_image, mother_image, child_gender)


            # 부모, 실제 자녀, 예측된 자녀 이미지를 하나로 결합
            combined_images = torch.cat([father_image[:5], mother_image[:5], child_image[:5], generated_img[:5]], dim=0)
            combined_images = (combined_images + 1) / 2
            # 4개의 이미지를 수평으로 결합하여 저장
            image_grid = make_grid(combined_images, nrow=5, normalize=True)

            # 이미지 저장
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(image_grid, os.path.join(save_dir, f"generated_epoch_{epoch + 1}.png"))

            break  # 첫 번째 배치만 저장

# Train/Valid loop
def run_epoch(generator, discriminator, loader, optimizer_G, optimizer_D, phase='train'):
    running_loss = 0.0
    is_train = phase == 'train'

    if is_train:
        generator.train()
        discriminator.train()
    else:
        generator.eval()
        discriminator.eval()

    # tqdm으로 로더 감싸기 (progress bar 표시)
    loader = tqdm(loader, desc=phase.capitalize())  # 'train' 또는 'valid'의 첫 글자만 대문자로 표시

    for i, (father_img, mother_img, real_child_img, other_img, child_gender, other_gender, _) in enumerate(loader):
        father_img = father_img.to(device)
        mother_img = mother_img.to(device)
        real_child_img = real_child_img.to(device)
        other_img = other_img.to(device)
        child_gender = child_gender.to(device)
        other_gender = other_gender.to(device)

        batch_size = father_img.size(0)

        # 진짜/가짜 라벨 설정
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ----------------------
        # 1. Discriminator 학습
        # ----------------------

        # 부모 이미지로부터 생성된 자녀 이미지
        fake_child_img = generator(father_img, mother_img, child_gender)

        # 부모와 진짜 자녀
        disc_pred_real, disc_kinship_real = discriminator(father_img, mother_img, real_child_img)

        # 부모와 가짜 자녀 (Generator가 생성한 자녀)
        disc_pred_fake, _ = discriminator(father_img, mother_img, fake_child_img.detach())

        # 부모와 무관한 다른 사람
        _, disc_kinship_other = discriminator(father_img, mother_img, other_img)


        # MSE 손실 (진짜/가짜 판별)
        d_loss_real = criterion_mse(disc_pred_real, real_labels)  # 진짜 자녀
        d_loss_fake = criterion_mse(disc_pred_fake, fake_labels)  # 가짜 자녀


        d_loss_kin_real = criterion_bce(disc_kinship_real, real_labels)   # 진짜 자녀
        d_loss_kin_other = criterion_bce(disc_kinship_other, fake_labels)  # 다른 자녀


        # 총 Discriminator 손실
        d_loss = (d_loss_kin_real + d_loss_kin_other) + (d_loss_real + d_loss_fake) * 0.5
        if is_train:
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        # ----------------------
        # 2. Generator 학습
        # ----------------------

        # Generator는 Discriminator를 속여야 함 (가짜 자녀를 진짜로 만들려고 시도)
        fake_fake_out, _ = discriminator(father_img, mother_img, fake_child_img)

        # Generator 손실: 생성된 자녀를 진짜로 판별되도록 함
        g_loss = criterion_mse(fake_fake_out, real_labels)  # 가짜 자녀를 진짜로
        # g_loss += criterion_mse(fake_child_img, real_child_img)

        if is_train:
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        # Loss 기록
        running_loss += (d_loss.item() + g_loss.item()) if is_train else d_loss.item()

        # tqdm progress bar에 현재 손실 표시
        loader.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

    return running_loss / len(loader)


# 학습 함수 정의
def train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epochs=20):
    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    # CSV 파일 작성
    with open('gan.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow(['Epoch', 'Train Loss', 'Valid Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss = run_epoch(generator, discriminator, train_loader, optimizer_G, optimizer_D, phase='train')
            valid_loss = run_epoch(generator, discriminator, valid_loader, optimizer_G, optimizer_D, phase='valid')

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            # 검증 성능 지표 계산
            accuracy, precision, recall, f1, roc_auc = evaluate_model(discriminator, valid_loader, device)

            # 가장 좋은 모델 저장
            if valid_loss < best_loss:
                best_loss = valid_loss
                print(f"Best model saved with valid loss: {valid_loss:.4f}")
                torch.save(generator.state_dict(), os.path.join("train", "best_generator.pth"))
                torch.save(discriminator.state_dict(), os.path.join("train", "best_discriminator.pth"))

            # 스케줄러 업데이트 (ReduceLROnPlateau는 성능에 따라 학습률 조정)
            scheduler_G.step(valid_loss)
            scheduler_D.step(valid_loss)

            # 매 에포크마다 그래프 저장
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(valid_losses, label='Valid Loss')
            plt.title('Train and Valid Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join("train", f"gan_loss_epoch.png"))
            plt.close()

            # 검증 이미지 저장
            save_images(epoch, generator, train_loader, device, "train/train")
            save_images(epoch, generator, valid_loader, device, "train/valid")

            # 매 에포크마다 CSV 파일에 성능 지표와 손실 저장
            writer.writerow([epoch + 1, train_loss, valid_loss, accuracy, precision, recall, f1, roc_auc])

def evaluate_model(discriminator, valid_loader, device):
    discriminator.eval()
    all_targets = []
    all_predictions = []
    all_kinship_scores = []  # ROC AUC 계산을 위해 raw kinship 출력값을 저장
 
    with torch.no_grad():
        for father_img, mother_img, real_child_img, other_img, child_gender, other_gender, _ in valid_loader:
            father_img = father_img.to(device)
            mother_img = mother_img.to(device)
            real_child_img = real_child_img.to(device)
            other_img = other_img.to(device)
            child_gender = child_gender.to(device)
            other_gender = other_gender.to(device)

            # 부모와 자녀 이미지로부터 친족 여부 예측
            _, real_output = discriminator(father_img, mother_img, real_child_img)
            _, other_output = discriminator(father_img, mother_img, other_img)

            # 예측값 (0.5를 기준으로 친족 여부를 결정)
            real_child_predictions = (real_output > 0.5).float().cpu().numpy()  # real_child_img의 예측
            other_predictions = (other_output <= 0.5).float().cpu().numpy()  # other_img의 예측 (비친족)

            # 친족(1)과 비친족(0) 타겟 설정 (real_child_img는 1, other_img는 0)
            real_child_targets = np.ones_like(real_child_predictions)  # 실제 자녀
            other_targets = np.zeros_like(other_predictions)  # 비친족

            # 실제값과 예측값 저장
            all_targets.extend(real_child_targets)
            all_targets.extend(other_targets)
            all_predictions.extend(real_child_predictions)
            all_predictions.extend(other_predictions)

            print("real_child_targets")
            print(real_child_targets[:5])

            print()
            print("other_targets")
            print(other_targets[:5])

            print()

            print("real_child_predictions")
            print(real_child_predictions[:5])
            print()
            
            print("other_predictions")
            print(other_predictions[:5])
            

            # raw kinship score 저장
            all_kinship_scores.extend(real_output.cpu().numpy())
            all_kinship_scores.extend(other_output.cpu().numpy())

    # Accuracy, Precision, Recall, F1 Score, ROC AUC 계산
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    roc_auc = roc_auc_score(all_targets, all_kinship_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc


if __name__ == "__main__":
    if not os.path.exists("train"):
        os.makedirs("train")

    # 시드 고정
    seed_everything(115)

    # 모델 초기화
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    epochs = 5000

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 모델 가중치 경로 설정
    generator_path = rf"train\best_generator.pth"
    discriminator_path = rf"train\best_discriminator.pth"

    # 저장된 모델 가중치 불러오기
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        print("Loaded Generator weights successfully.")

    if os.path.exists(discriminator_path):
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        print("Loaded Discriminator weights successfully.")
    else:
        print("No saved weights found. Initializing models with random weights.")

    # 손실함수 및 옵티마이저 설정
    criterion_mse = nn.MSELoss().to(device)
    criterion_bce = nn.BCELoss().to(device)
    criterion_l1 = nn.L1Loss().to(device)

    optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

    # 스케줄러 설정 (ReduceLROnPlateau)
    scheduler_G = lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

    # 사용 예시
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 범위로 정규화
    ])

    dataset = FamilyDataset(root_dir="dataset_v2", transform=transform)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    # 학습 시작
    train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epochs=epochs)
    evaluate_model(discriminator, valid_loader, device)
