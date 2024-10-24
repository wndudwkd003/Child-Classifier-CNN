import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50
from utils.dataset_v2 import FamilyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # tqdm 추가
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 시드 고정 함수
def seed_everything(seed=115):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 부모와 자녀 간의 거리 (유사도)
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        # 부모와 무관한 사람 간의 거리
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        # Triplet Loss 계산
        loss = torch.mean(torch.relu(positive_distance - negative_distance + self.margin))
        return loss

# ResNet-50을 수정하여 부모-자녀 분류에 사용
def get_resnet50_model(pretrained=True, input_channels=3):
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Identity()  # 마지막 FC 레이어 제거 (특징 추출만 수행)
    if input_channels != 3:
        # 첫 번째 Conv2d 레이어 수정 (입력 채널을 3 -> input_channels로 변경)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# Train/Valid loop
def run_epoch(model, loader, optimizer, criterion_triplet, criterion_bce, phase='train'):
    running_loss = 0.0
    is_train = phase == 'train'

    if is_train:
        model.train()
    else:
        model.eval()

    all_targets = []
    all_predictions = []
    all_kinship_scores = []  # ROC AUC 계산을 위해 raw kinship 출력값을 저장

    # tqdm으로 로더 감싸기 (progress bar 표시)
    loader = tqdm(loader, desc=phase.capitalize())

    for i, (father_img, mother_img, real_child_img, other_img, _, _, _) in enumerate(loader):
        father_img = father_img.to(device)
        mother_img = mother_img.to(device)
        real_child_img = real_child_img.to(device)
        other_img = other_img.to(device)

        # 아빠-자식-다른 사람 트리플릿
        anchor_features_father = model(father_img)  # 아빠 특징
        positive_features_child = model(real_child_img)  # 실제 자녀 특징
        negative_features_other = model(other_img)  # 무관한 사람 특징

        # 아빠-자식-다른 사람으로 Triplet Loss 계산
        triplet_loss_father = criterion_triplet(anchor=anchor_features_father, positive=positive_features_child, negative=negative_features_other)

        # 엄마-자식-다른 사람 트리플릿
        anchor_features_mother = model(mother_img)  # 엄마 특징

        # 엄마-자식-다른 사람으로 Triplet Loss 계산
        triplet_loss_mother = criterion_triplet(anchor=anchor_features_mother, positive=positive_features_child, negative=negative_features_other)

        # 전체 Triplet Loss 계산
        triplet_loss = (triplet_loss_father + triplet_loss_mother) / 2

        # BCE Loss 계산 (아빠/엄마-자녀 vs 아빠/엄마-무관한 사람)
        positive_similarity_father = F.cosine_similarity(anchor_features_father, positive_features_child)
        negative_similarity_father = F.cosine_similarity(anchor_features_father, negative_features_other)
        positive_similarity_mother = F.cosine_similarity(anchor_features_mother, positive_features_child)
        negative_similarity_mother = F.cosine_similarity(anchor_features_mother, negative_features_other)

        real_labels_father = torch.ones_like(positive_similarity_father)
        fake_labels_father = torch.zeros_like(negative_similarity_father)
        real_labels_mother = torch.ones_like(positive_similarity_mother)
        fake_labels_mother = torch.zeros_like(negative_similarity_mother)

        bce_loss_real_father = criterion_bce(positive_similarity_father, real_labels_father)
        bce_loss_fake_father = criterion_bce(negative_similarity_father, fake_labels_father)
        bce_loss_real_mother = criterion_bce(positive_similarity_mother, real_labels_mother)
        bce_loss_fake_mother = criterion_bce(negative_similarity_mother, fake_labels_mother)

        bce_loss_father = (bce_loss_real_father + bce_loss_fake_father) / 2
        bce_loss_mother = (bce_loss_real_mother + bce_loss_fake_mother) / 2

        # 전체 BCE Loss 계산
        bce_loss = (bce_loss_father + bce_loss_mother) / 2

        # 총 손실 계산
        loss = triplet_loss + bce_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss 기록
        running_loss += loss.item()

        # Validation 단계에서 평가 지표 계산
        if not is_train:
            real_child_predictions_father = (positive_similarity_father > 0.5).float().cpu().numpy()
            other_predictions_father = (negative_similarity_father <= 0.5).float().cpu().numpy()
            real_child_predictions_mother = (positive_similarity_mother > 0.5).float().cpu().numpy()
            other_predictions_mother = (negative_similarity_mother <= 0.5).float().cpu().numpy()

            # 친족(1)과 비친족(0) 타겟 설정
            real_child_targets_father = np.ones_like(real_child_predictions_father)
            other_targets_father = np.zeros_like(other_predictions_father)
            real_child_targets_mother = np.ones_like(real_child_predictions_mother)
            other_targets_mother = np.zeros_like(other_predictions_mother)

            # 실제값과 예측값 저장
            all_targets.extend(real_child_targets_father)
            all_targets.extend(other_targets_father)
            all_predictions.extend(real_child_predictions_father)
            all_predictions.extend(other_predictions_father)

            all_targets.extend(real_child_targets_mother)
            all_targets.extend(other_targets_mother)
            all_predictions.extend(real_child_predictions_mother)
            all_predictions.extend(other_predictions_mother)

            # raw kinship score 저장
            all_kinship_scores.extend(positive_similarity_father.detach().cpu().numpy())
            all_kinship_scores.extend(negative_similarity_father.detach().cpu().numpy())
            all_kinship_scores.extend(positive_similarity_mother.detach().cpu().numpy())
            all_kinship_scores.extend(negative_similarity_mother.detach().cpu().numpy())

        # tqdm progress bar에 현재 손실 표시
        loader.set_postfix(loss=loss.item())

    if not is_train:
        # Accuracy, Precision, Recall, F1 Score, ROC AUC 계산
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        roc_auc = roc_auc_score(all_targets, all_kinship_scores)

        print(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    return running_loss / len(loader)

# 학습 함수 정의
def train_model(model, train_loader, valid_loader, optimizer, scheduler, epochs=20):
    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = run_epoch(model, train_loader, optimizer, criterion_triplet, criterion_bce, phase='train')
        valid_loss = run_epoch(model, valid_loader, optimizer, criterion_triplet, criterion_bce, phase='valid')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        # 가장 좋은 모델 저장
        if valid_loss < best_loss:
            best_loss = valid_loss
            print(f"Best model saved with valid loss: {valid_loss:.4f}")
            torch.save(model.state_dict(), os.path.join("train", "best_kinship_resnet50_triplet.pth"))

        # 스케줄러 업데이트
        scheduler.step(valid_loss)

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

if __name__ == "__main__":
    if not os.path.exists("train"):
        os.makedirs("train")

    # 시드 고정
    seed_everything(115)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5

    kinship_model = get_resnet50_model(pretrained=True, input_channels=3).to(device)

    # 손실함수 및 옵티마이저 설정
    criterion_triplet = TripletLoss().to(device)
    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.AdamW(kinship_model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

    # 스케줄러 설정 (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FamilyDataset(root_dir="dataset_v2", transform=transform)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 학습 시작
    train_model(kinship_model, train_loader, valid_loader, optimizer, scheduler, epochs=epochs)
