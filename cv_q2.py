import cv2
import numpy as np

# 얼굴 감지 모델 파일 경로
prototxt_path = rf"C:\Development\Projects\ChildGenerator\utils\deploy.prototxt.txt"
caffemodel_path = rf"C:\Development\Projects\ChildGenerator\utils\res10_300x300_ssd_iter_140000.caffemodel"

# DNN 모델 로드
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


# 얼굴 감지 함수
def detect_face(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # 얼굴 검출
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 검출된 얼굴의 신뢰도 (0.5 이상인 경우만 사용)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            if face.size == 0:
                print("검출된 얼굴이 너무 작습니다.")
                return None
            face = cv2.resize(face, (112, 112))  # 임베딩을 위해 크기 조정
            cv2.imshow("Detected Face", face)  # 감지된 얼굴 확인
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return face
    print("얼굴을 감지하지 못했습니다.")
    return None



# 두 얼굴 간의 유사도 계산 (코사인 유사도)
def calculate_cosine_similarity(face1, face2):
    face1 = detect_face(face1)
    face2 = detect_face(face2)

    if face1 is None or face2 is None:
        print("얼굴을 감지하지 못했습니다.")
        return None

    # 두 얼굴의 특징을 비교 (간단하게 픽셀 값으로 코사인 유사도 계산)
    face1 = face1.flatten().reshape(1, -1)
    face2 = face2.flatten().reshape(1, -1)

    similarity = np.dot(face1, face2.T) / (np.linalg.norm(face1) * np.linalg.norm(face2))
    return similarity[0][0]


# 두 얼굴 이미지를 로드
face_image1 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\other\child_man_5.jpg')
face_image2 = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\father\father_man_5.jpg')

if face_image1 is None or face_image2 is None:
    print("이미지 로드 실패!")
else:
    print("이미지 로드 성공!")


# 얼굴 유사도 계산
similarity = calculate_cosine_similarity(face_image1, face_image2)

if similarity is not None:
    print(f"두 얼굴 간의 코사인 유사도: {similarity}")
else:
    print("유사도를 계산할 수 없습니다.")
