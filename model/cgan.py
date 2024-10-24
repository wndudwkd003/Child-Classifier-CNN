import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

import torch.nn.functional as F


# U-Net의 인코더 (다운샘플링)
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        # 업샘플링 레이어의 입력 채널 수 수정
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        # print("x, skip_connection: ", x.shape, " ", skip_connection.shape)
        x = self.up(x)
        # print("x = self.up(x): ", x.shape)

        x = torch.cat([x, skip_connection], dim=1)  # 결합 후 채널 수가 두 배로 늘어남
        
        # print("x cat : ", x.shape)
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], gender_emb_dim=4):
        super(Generator, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.gender_emb_dim = gender_emb_dim

        # 성별 임베딩 (0: 남성, 1: 여성)
        self.gender_embedding = nn.Embedding(2, gender_emb_dim)  


        # 인코더 구성 (부모 이미지 2장을 합쳐서 처리하므로 입력 채널은 6)
        in_channels = 6
        for feature in features:
            self.encoder_blocks.append(UNetEncoder(in_channels, feature))
            in_channels = feature

        # 디코더 구성
        for feature in reversed(features):
            self.decoder_blocks.append(UNetDecoder(feature * 2, feature))

        # 마지막 컨볼루션 레이어 (출력 채널 수는 3: R, G, B)

        self.emb_bottleneck = nn.Sequential(
            nn.Conv2d(features[-1] + gender_emb_dim, features[-1] // 2, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(features[-1] // 2),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1] // 2, features[-1], kernel_size=3, padding=1), 
            nn.InstanceNorm2d(features[-1]),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(features[0] // 2, 3, kernel_size=1)  # 3채널 이미지 생성
        self.tan = nn.Tanh()

    def forward(self, father_img, mother_img, gender):
        # 부모 이미지 결합 (채널 기준으로 결합, 6채널 이미지)
        x = torch.cat([father_img, mother_img], dim=1)

        # 인코더 진행
        skip_connections = []
        for encoder in self.encoder_blocks:
            x, pooled = encoder(x)
            skip_connections.append(x)
            x = pooled

        # 성별 정보 추가 (성별 임베딩을 [batch, 1, bottleneck_h, bottleneck_w]로 변환)
        gender_embed = self.gender_embedding(gender).view(-1, self.gender_emb_dim, 1, 1)  # (batch_size, self.gender_emb_dim, 1, 1)
        gender_embed = gender_embed.expand(-1, self.gender_emb_dim, x.size(2), x.size(3))  # (batch_size, 1, bottleneck_h, bottleneck_w)

        # 성별 정보와 인코더 출력 결합
        x = torch.cat([x, gender_embed], dim=1)  

        # 병목 부분
        x = self.emb_bottleneck(x)
        x = self.bottleneck(x)


        # 디코더 진행 (스킵 연결 포함)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, skip_connections[i])
            # print(f"x = self.decoder_blocks[{i}](x, skip_connections[{i}]): ", x.shape)

        return self.tan(self.final_conv(x))



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 부모 및 자녀 이미지 처리 부분 (아버지, 어머니, 자녀 이미지를 결합하여 입력)
        self.feature_extractor = EfficientNet.from_name('efficientnet-b7')
        self.feature_extractor._conv_stem = nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False)

        # 진짜/가짜 판별 출력 (1 채널)
        self.fc = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.kinship_fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.real_fake_fc = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, father_img, mother_img, child_img):
        # 부모 이미지와 자녀 이미지 결합 (아버지, 어머니, 자녀)
        combined_img = torch.cat([father_img, mother_img, child_img], dim=1)  # (batch_size, 9, height, width)

        # 이미지 특징 벡터 추출
        features = self.feature_extractor.extract_features(combined_img)  # EfficientNet 특징 추출

        features = F.adaptive_avg_pool2d(features, 1).reshape(features.size(0), -1)  # (batch_size, 1280)

        # 진짜/가짜 판별
        fc = self.fc(features)
        
        real_fake = self.real_fake_fc(fc)
        kinship = self.kinship_fc(fc)


        return real_fake, kinship   # 0: real image or fake image, 1: kinship image or non-kinship image