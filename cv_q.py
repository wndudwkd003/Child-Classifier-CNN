import cv2
import numpy as np
import torch
import insightface
from sklearn.metrics.pairwise import cosine_similarity

# ArcFace 모델 로드
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)  # CPU를 사용하려면 -1, GPU 사용 시 GPU ID


# 얼굴 임베딩 추출 함수
# 얼굴 임베딩 추출 함수
def get_face_embedding(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # 이미지 크기 조정 (112x112)
    face_image = cv2.resize(face_image, (112, 112))

    # det-size를 변경하여 더 작은 얼굴도 검출할 수 있도록 조정
    model.prepare(ctx_id=-1, det_size=(112, 112))  # det-size를 낮춰서 작은 얼굴도 감지

    # 모델이 처리할 수 있도록 전처리
    faces = model.get(face_image)

    if len(faces) == 0:
        print("얼굴을 감지하지 못했습니다.")
        return None

    # 첫 번째 얼굴의 임베딩 추출
    embedding = faces[0].normed_embedding
    return embedding


# 두 얼굴 간의 유사도 계산 (코사인 유사도)
def calculate_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


# 두 얼굴 이미지를 로드
face_image1 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\other\child_man_5.jpg')
face_image2 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\father\father_man_5.jpg')

# 얼굴 임베딩 추출
embedding1 = get_face_embedding(face_image1)
embedding2 = get_face_embedding(face_image2)

# 두 얼굴 간의 유사도 계산
cosine_similarity = calculate_cosine_similarity(embedding1, embedding2)

if cosine_similarity is not None:
    print(f"두 얼굴 간의 코사인 유사도: {cosine_similarity}")
else:
    print("유사도를 계산할 수 없습니다.")
