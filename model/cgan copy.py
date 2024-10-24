import torch
import torch.nn as nn

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


import torch
import torch.nn as nn


import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 성별 임베딩 (0: 남성, 1: 여성) -> 성별 정보를 4차원 벡터로 변환
        self.gender_embedding = nn.Embedding(2, 4)  # 임베딩 차원을 64에서 4로 줄임

        # 부모 이미지 처리 부분 (아빠와 엄마 -> 6채널 + 성별 정보 -> 14채널)
        self.parent_model = nn.Sequential(
            nn.Conv2d(6 + 4 + 4, 64, kernel_size=4, stride=2, padding=1),  # 6채널 + 4 + 4 = 14채널
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # 부모 특징 벡터를 Fully Connected로 변환하여 고정된 차원으로 맞춤
        self.fc_parent = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),  # 4x4에서 512차원으로 맞추기 위한 FC 레이어
            nn.LeakyReLU(0.2)
        )

        # 자녀 이미지 처리 부분 (자녀 이미지 -> 부모 특징과 결합 후)
        self.child_model = nn.Sequential(
            nn.Conv2d(3 + 4, 512, kernel_size=4, stride=2, padding=1),  # 자녀 성별 정보도 포함
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0),  # 특징 벡터 추출
        )

        # 자녀 특징 벡터를 Fully Connected로 변환하여 고정된 차원으로 맞춤
        self.fc_child = nn.Sequential(
            nn.Linear(430592, 512),  # 자녀 이미지 특징을 부모와 같은 512차원으로 맞춤
            nn.LeakyReLU(0.2)
        )

        # 진짜/가짜 판별 출력 (1 채널)
        self.fc_real_fake = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

        # 친족 여부 판별 출력 (1 채널)
        self.fc_kinship = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
            )

    def forward(self, father_img, mother_img, child_img, child_gender):
        # 아버지 성별 (고정: 0), 어머니 성별 (고정: 1)
        father_gender = torch.zeros(father_img.size(0), dtype=torch.long, device=father_img.device)
        mother_gender = torch.ones(mother_img.size(0), dtype=torch.long, device=mother_img.device)

        # 성별 임베딩 (아버지, 어머니, 자녀)
        father_gender_embed = self.gender_embedding(father_gender).view(-1, 4, 1, 1)  # (batch_size, 4, 1, 1)
        mother_gender_embed = self.gender_embedding(mother_gender).view(-1, 4, 1, 1)  # (batch_size, 4, 1, 1)
        child_gender_embed = self.gender_embedding(child_gender).view(-1, 4, 1, 1)  # (batch_size, 4, 1, 1)

        # 성별 임베딩을 이미지 차원에 맞게 확장
        father_gender_embed = father_gender_embed.expand(-1, 4, father_img.size(2), father_img.size(3))
        mother_gender_embed = mother_gender_embed.expand(-1, 4, mother_img.size(2), mother_img.size(3))
        child_gender_embed = child_gender_embed.expand(-1, 4, child_img.size(2), child_img.size(3))

        # 부모 이미지 결합 (아빠, 엄마, 성별 정보)
        parent_combined_img = torch.cat([father_img, father_gender_embed, mother_img, mother_gender_embed], dim=1)

        # 부모 이미지 특징 벡터 추출
        parent_features = self.parent_model(parent_combined_img)
        parent_features = parent_features.view(parent_features.size(0), -1)  # (batch_size, 512 * 4 * 4)

        # 부모 특징 벡터를 FC로 차원 맞춤
        parent_features = self.fc_parent(parent_features)  # (batch_size, 512)

        # 부모 특징과 자녀 이미지 결합
        child_combined_img = torch.cat([child_img, child_gender_embed], dim=1)

        # 자녀 특징 벡터 추출
        child_features = self.child_model(child_combined_img)
        child_features = child_features.view(child_features.size(0), -1)  # (batch_size, 512 * 4 * 4)

        # 자녀 특징 벡터를 FC로 차원 맞춤
        # print("child_features: ", child_features.shape)
        child_features = self.fc_child(child_features)  # (batch_size, 512)

        # 진짜/가짜 판별
        real_fake_out = self.fc_real_fake(child_features)

        # 친족 여부 판별
        kinship_out = self.fc_kinship(child_features)

        # 부모 특징 벡터도 함께 출력
        return real_fake_out, kinship_out, child_features, parent_features
