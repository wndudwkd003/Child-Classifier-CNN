import cv2
import dlib
import numpy as np

# Dlib의 얼굴 검출기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r"C:\Development\Projects\ChildGenerator\utils\shape_predictor_68_face_landmarks.dat")  # Dlib 모델 다운로드 필요


# 얼굴과 랜드마크를 검출하는 함수
def detect_face_and_landmarks(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) == 0:
        print("얼굴을 감지하지 못했습니다.")
        return None

    face_coords = faces[0]
    landmarks = predictor(gray_image, face_coords)

    # (x, y) 형태의 랜드마크 좌표 추출
    landmarks_points = {}
    for n in range(0, 68):  # 68개의 랜드마크 ID에 대해 좌표 추출
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points[n] = (x, y)  # ID를 키로 좌표 저장

    return landmarks_points


# 랜드마크 간의 유사도를 계산하는 함수 (ID가 동일한 랜드마크만 비교)
def calculate_similarity(landmarks1, landmarks2):
    # 두 얼굴에서 감지된 랜드마크 ID 교집합만 사용
    common_ids = set(landmarks1.keys()) & set(landmarks2.keys())

    if len(common_ids) == 0:
        print("공통으로 감지된 랜드마크가 없습니다.")
        return None

    # 각 공통 랜드마크 좌표 간의 유클리드 거리 계산
    distances = [np.linalg.norm(np.array(landmarks1[id]) - np.array(landmarks2[id])) for id in common_ids]

    # 평균 유클리드 거리 계산
    avg_distance = np.mean(distances)

    # 유사도 계산 (거리가 작을수록 유사함)
    similarity_score = 1 / (1 + avg_distance)  # 거리가 작을수록 큰 값

    return similarity_score


# 이미지 로드
image1 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\father\father_man_1.jpg')
image2 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\child\child_man_2.jpg')

# 두 이미지에서 랜드마크 검출
landmarks1 = detect_face_and_landmarks(image1)
landmarks2 = detect_face_and_landmarks(image2)

# 두 얼굴의 유사도 계산
if landmarks1 is not None and landmarks2 is not None:
    similarity = calculate_similarity(landmarks1, landmarks2)
    if similarity is not None:
        print(f"두 얼굴의 유사도 점수: {similarity}")
else:
    print("얼굴 랜드마크를 감지하지 못했습니다.")
