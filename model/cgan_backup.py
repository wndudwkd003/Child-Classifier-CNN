import torch
import torch.nn as nn
from model.cbam import CBAM

# U-Net의 인코더 (다운샘플링)
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pooled = self.pool(conv)
        return conv, pooled

# U-Net의 디코더 (업샘플링)
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)  # 스킵 연결
        x = self.conv(x)
        return x

# U-Net 기반 Generator (부모 얼굴 + 성별 정보 입력)
class Generator(nn.Module):
    def __init__(self, features=[64, 128, 256, 512]):
        super(Generator, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # 성별 임베딩 (0: 남성, 1: 여성)
        self.gender_embedding = nn.Embedding(2, 64)  # 성별을 64차원 임베딩으로 변환

        # 인코더 구성 (부모 이미지 2장을 합쳐서 처리하므로 입력 채널은 6)
        in_channels = 6
        for feature in features:
            self.encoder_blocks.append(UNetEncoder(in_channels, feature))
            in_channels = feature

        # 디코더 구성
        for feature in reversed(features):
            self.decoder_blocks.append(UNetDecoder(feature * 2, feature))

        # 마지막 컨볼루션 레이어 (출력 채널 수는 3: R, G, B)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1] + 64, features[-1] * 2, kernel_size=3, padding=1),  # 성별 정보와 결합
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(features[0], 3, kernel_size=1)  # 3채널 이미지 생성

    def forward(self, father_img, mother_img, gender):
        # 부모 이미지 결합 (채널 기준으로 결합, 6채널 이미지)
        x = torch.cat([father_img, mother_img], dim=1)

        # 인코더 진행
        skip_connections = []
        for encoder in self.encoder_blocks:
            x, pooled = encoder(x)
            skip_connections.append(x)
            x = pooled

        # 성별 정보 추가 (성별 임베딩을 공간 차원으로 변환)
        gender_embed = self.gender_embedding(gender).view(-1, 64, 1, 1)  # (batch_size, 64, 1, 1)
        gender_embed = gender_embed.expand(-1, 64, x.size(2), x.size(3))  # 성별 정보를 인코더 출력 차원에 맞추어 확장

        # 성별 정보와 인코더 출력 결합
        x = torch.cat([x, gender_embed], dim=1)  # (batch_size, features[-1] + 64, h, w)

        # 병목 부분
        x = self.bottleneck(x)

        # 디코더 진행 (스킵 연결 포함)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, skip_connections[i])

        return self.final_conv(x)




class Discriminator(nn.Module):
    def __init__(self, channels=1024, r=16):
        super(Discriminator, self).__init__()

        # 학습 가능한 스케일 가중치
        self.scale1_weight = nn.Parameter(torch.tensor(0.5))  # scale1의 초기 가중치
        self.scale2_weight = nn.Parameter(torch.tensor(0.5))  # scale2의 초기 가중치

        # 멀티스케일 처리 (특징을 다양한 크기로 처리하기 위한 두 가지 경로)
        self.scale1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=4, stride=2, padding=1),  # 부모 2명 + 자식 (9채널 입력)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=4, stride=1, padding=1),  # 더 작은 stride로 작은 변화 포착
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # 결합 후 특징 맵 처리 (점진적으로 채널 수를 증가시킴)
        self.merge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 256 -> 512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, channels, kernel_size=4, stride=2, padding=1),  # 512 -> 1024
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
        )

        # CBAM (채널 및 공간 어텐션 모듈 추가)
        self.cbam = CBAM(channels=channels, r=r)

        # 친족 여부 판별을 위한 분류 헤드
        self.relatedness_head = nn.Sequential(
            nn.Linear(channels * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # 출력은 친족 여부 (0 또는 1)
            nn.Sigmoid()  # 확률 출력
        )

    def forward(self, father_img, mother_img, child_img):
        # 부모 이미지와 자식 이미지 결합 (9채널)
        x = torch.cat([father_img, mother_img, child_img], dim=1)

        # 멀티스케일 경로로 처리
        scale1_out = self.scale1(x)
        scale2_out = self.scale2(x)

        # 가중합으로 멀티스케일 출력 결합 (가중치 학습 가능)
        combined = self.scale1_weight * scale1_out + self.scale2_weight * scale2_out  # 가중합

        # 결합된 특징 맵에 CBAM 적용
        combined = self.merge(combined)
        combined = self.cbam(combined)  # CBAM 적용

        # 친족 여부 판별
        combined_flat = torch.flatten(combined, start_dim=1)
        relatedness = self.relatedness_head(combined_flat)

        return relatedness
