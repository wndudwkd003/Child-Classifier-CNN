from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import numpy as np

# FaceNet 모델 로드 (사전 학습된 모델)
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩 추출 함수
def get_face_embedding(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = cv2.resize(face_image, (160, 160))  # FaceNet은 160x160 입력을 사용
    face_image = torch.tensor(face_image).permute(2, 0, 1).unsqueeze(0).float()  # PyTorch 텐서로 변환
    face_image = (face_image - 127.5) / 128.0  # 정규화

    # 얼굴 임베딩 추출
    with torch.no_grad():
        embedding = model(face_image)

    return embedding.squeeze().numpy()

# 두 얼굴 간의 유사도 계산 (L2 거리)
def calculate_similarity(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# 두 얼굴 이미지를 로드
face_image1 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\child\child_man_6.jpg')
face_image2 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\father\father_man_6.jpg')

# 얼굴 임베딩 추출
embedding1 = get_face_embedding(face_image1)
embedding2 = get_face_embedding(face_image2)

# 두 얼굴 간의 유사도 계산
similarity = calculate_similarity(embedding1, embedding2)
print(f"두 얼굴 간의 유사도: {similarity}")
