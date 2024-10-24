import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

def debug_discriminator():
    # 디버깅을 위한 샘플 이미지 생성 (batch_size=2, 채널=3, 크기=64x64)
    father_img = torch.randn(2, 3, 64, 64)
    mother_img = torch.randn(2, 3, 64, 64)
    child_img = torch.randn(2, 3, 64, 64)

    # 모델 초기화 및 forward 패스
    discriminator = Discriminator()
    output = discriminator(father_img, mother_img, child_img)

    print("Discriminator output:", output)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 부모 및 자녀 이미지 처리 부분 (아버지, 어머니, 자녀 이미지를 결합하여 입력)
        self.feature_extractor = EfficientNet.from_name('efficientnet-b0')
        self.feature_extractor._conv_stem = nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 입력 채널을 9로 변경 (아버지, 어머니, 자녀 이미지)

        # 진짜/가짜 판별 출력 (1 채널)
        self.fc_real_fake = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, father_img, mother_img, child_img):
        # 부모 이미지와 자녀 이미지 결합 (아버지, 어머니, 자녀)
        combined_img = torch.cat([father_img, mother_img, child_img], dim=1)  # (batch_size, 9, height, width)

        # 이미지 특징 벡터 추출
        features = self.feature_extractor.extract_features(combined_img)  # EfficientNet 특징 추출
        print("Extracted features shape:", features.shape)
        features = F.adaptive_avg_pool2d(features, 1).reshape(features.size(0), -1)  # (batch_size, 1280)
        print("features:, ", features.shape)
        # 진짜/가짜 판별
        real_fake_out = self.fc_real_fake(features)

        return real_fake_out

if __name__ == "__main__":
    debug_discriminator()