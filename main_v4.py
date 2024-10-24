import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50, densenet121
from efficientnet_pytorch import EfficientNet
from utils.dataset_v2 import FamilyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import csv

from tqdm import tqdm  # tqdm 추가
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class KinshipClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(KinshipClassifier, self).__init__()
        layers = []

        # 입력을 한번 활성화 함수와 드롭아웃을 거친 후 바로 출력 레이어로 연결
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # 출력 레이어 (출력 차원 1)
        layers.append(nn.Linear(hidden_dim, 1))

        # 레이어들을 Sequential로 정의
        self.classifier = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x

        
# 시드 고정 함수
def seed_everything(seed=115):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# 모델을 정의하는 함수
def get_model(model_name, pretrained=True, input_channels=3):
    if model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Identity()  # 마지막 FC 레이어 제거 (특징 추출만 수행)
    elif model_name == 'efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model._fc = nn.Identity()  # 마지막 FC 레이어 제거
    elif model_name == 'densenet121':
        model = densenet121(pretrained=pretrained)
        model.classifier = nn.Identity()  # 마지막 FC 레이어 제거
    else:
        raise ValueError("Invalid model name. Choose from ['resnet50', 'efficientnet-b7', 'densenet121']")

    if input_channels != 3:
        # 입력 채널 변경
        # model.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model._conv_stem = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)

        # model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    return model

# Train/Valid loop
def run_epoch(model, classifier, loader, optimizer, criterion_bce, phase='train'):
    running_loss = 0.0
    is_train = phase == 'train'

    if is_train:
        model.train()
        classifier.train()
    else:
        model.eval()
        classifier.eval()

    all_targets = []
    all_predictions = []
    all_kinship_scores = []

    # tqdm으로 로더 감싸기 (progress bar 표시)
    loader = tqdm(loader, desc=phase.capitalize())

    for i, (father_img, mother_img, real_child_img, other_img, _, _, _) in enumerate(loader):
        father_img = father_img.to(device)
        mother_img = mother_img.to(device)
        real_child_img = real_child_img.to(device)
        other_img = other_img.to(device)

        # 아빠-자식-다른 사람 트리플릿

        combined_input_child = torch.cat((father_img, mother_img, real_child_img), dim=1)  # (Batch, 9, Height, Width)
        output_child = model(combined_input_child)
        kinship_positive = classifier(output_child)

        combined_input_other = torch.cat((father_img, mother_img, other_img), dim=1)  # (Batch, 9, Height, Width)
        output_other = model(combined_input_other)
        kinship_negative = classifier(output_other)

        # BCE Loss 계산
        real_labels = torch.ones_like(kinship_positive)
        fake_labels = torch.zeros_like(kinship_negative)

        bce_loss_real = criterion_bce(kinship_positive, real_labels)
        bce_loss_fake = criterion_bce(kinship_negative, fake_labels)

        loss = (bce_loss_real + bce_loss_fake) / 2

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss 기록
        running_loss += loss.item()

        # Validation 단계에서 평가 지표 계산
        if not is_train:
            predictions = torch.cat((kinship_positive, kinship_negative), dim=0).detach().cpu().numpy() > 0.5
            targets = torch.cat((real_labels, fake_labels), dim=0).detach().cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # raw kinship score 저장
            all_kinship_scores.extend(kinship_positive.detach().cpu().numpy())
            all_kinship_scores.extend(kinship_negative.detach().cpu().numpy())

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

        return running_loss / len(loader), accuracy, precision, recall, f1, roc_auc

    else:
        return running_loss / len(loader)

# 학습 함수 정의
def train_model(model, classifier, train_loader, valid_loader, optimizer, scheduler, epochs=20, model_name='model'):
    best_loss = float('inf')
    train_losses = []
    valid_losses = []

    # CSV 파일 생성 및 열기
    csv_file_path = os.path.join("train", f"{model_name}_results_plus22222.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])  # 헤더 추가

        for epoch in range(epochs):

            print(f"Epoch {epoch+1}/{epochs} - Model: {model_name}")
            train_loss = run_epoch(model, classifier, train_loader, optimizer, criterion_bce, phase='train')
            valid_loss, accuracy, precision, recall, f1, roc_auc = run_epoch(model, classifier, valid_loader, optimizer, criterion_bce, phase='valid')

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            # 가장 좋은 모델 저장
            if valid_loss < best_loss:
                best_loss = valid_loss
                print(f"Best model saved with valid loss: {valid_loss:.4f}")
                torch.save(model.state_dict(), os.path.join("train", f"best_{model_name}_triplet.pth"))
                torch.save(classifier.state_dict(), os.path.join("train", f"best_{model_name}_classifier.pth"))

            # 스케줄러 업데이트
            scheduler.step(valid_loss)

            writer.writerow([epoch + 1, train_loss, valid_loss, accuracy, precision, recall, f1, roc_auc])

            # 매 에포크마다 그래프 저장
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(valid_losses, label='Valid Loss')
            plt.title(f'Train and Valid Loss - {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join("train", f"{model_name}_loss_epoch_{epoch+1}.png"))
            plt.close()

if __name__ == "__main__":
    if not os.path.exists("train"):
        os.makedirs("train")

    # 시드 고정
    seed_everything(115)

    # 모델 초기화
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 500

    # model_names = ['resnet50', 'efficientnet-b7', 'densenet121']
    model_names = ['efficientnet-b7']
    for model_name in model_names:
        print(f"Training {model_name} model")
        model = get_model(model_name, pretrained=True, input_channels=9).to(device)

        dummy_input = torch.randn(1, 9, 224, 224).to(device)

        dummy_output = model(dummy_input)

        print("features: ", dummy_output.shape)

        classifier = KinshipClassifier(input_dim=dummy_output.shape[-1]).to(device)  # 특징 벡터 차원 설정

        criterion_bce = nn.BCELoss().to(device)

        optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)

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

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

        # 학습 시작
        train_model(model, classifier, train_loader, valid_loader, optimizer, scheduler, epochs=epochs, model_name=model_name)
