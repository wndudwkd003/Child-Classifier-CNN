import cv2
import dlib

# Dlib의 얼굴 검출기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Development\Projects\ChildGenerator\utils\shape_predictor_68_face_landmarks.dat")  # Dlib 모델 다운로드 필요


# 얼굴과 랜드마크를 검출하는 함수
def detect_face_and_landmarks(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) == 0:
        print("얼굴을 감지하지 못했습니다.")
        return None, None

    face_coords = faces[0]
    landmarks = predictor(gray_image, face_coords)

    # (x, y) 형태의 랜드마크 좌표 추출
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    return face_coords, landmarks_points


# 랜드마크를 이미지에 표시하는 함수
def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image


# 이미지 로드
image = cv2.imread(r'C:\Development\Projects\ChildGenerator\dataset_v2\F0001_1_F\father\father_man_1.jpg')

# 얼굴 및 랜드마크 검출
face_coords, landmarks = detect_face_and_landmarks(image)

if landmarks is not None:
    # 랜드마크 그리기
    image_with_landmarks = draw_landmarks(image, landmarks)

    image_resized = cv2.resize(image_with_landmarks, (800, 800))  # 800x800으로 크기 조정
    cv2.imshow("Resized Landmarks", image_resized)
    cv2.waitKey(0)
else:
    print("얼굴 랜드마크를 감지하지 못했습니다.")
