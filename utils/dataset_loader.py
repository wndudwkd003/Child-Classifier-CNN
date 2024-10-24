import shutil
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Step 1: 기본 경로 설정 및 폴더 생성
base_dir = Path('C:/Users/ymail/Downloads/fiw')  # 기본 경로
splits = ['train', 'val', 'test']  # 데이터셋 폴더들
output_dir = Path('../dataset')  # 최종 저장할 경로
output_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성

current_family_number = 1
last_family_id = None

family_map = {}  # 자식 ID를 기준으로 아빠, 엄마 정보를 매핑

# Step 2: train, val, test 폴더 각각을 처리
for split in splits:
    labels_file = base_dir / split / 'labels.csv'  # 각 split 폴더 내의 labels.csv 경로
    labels_df = pd.read_csv(labels_file)  # labels.csv 파일 로드

    # 부모-자식 관계 필터링 (fs, ms, fd, md 모두 추출)
    filtered_df = labels_df[labels_df['ptype'].isin(['fs', 'ms', 'fd', 'md'])]

    # Step 3: 부모와 자식 관계를 매핑하여 family_map 생성
    for _, row in filtered_df.iterrows():
        # p1과 p2 경로에서 두 번째 슬래시까지만 사용하여 폴더를 설정
        p1_parent_id, p1_man_id = row['p1'].split('/')[:2]
        p2_parent_id, p2_man_id = row['p2'].split('/')[:2]
        ptype = row['ptype']

        # family_map에 자식 정보를 추가
        parent_id = f"{p2_parent_id}_{p2_man_id}"
        family_map.setdefault(parent_id, {'father': None, 'mother': None, 'child': p2_man_id, 'parent_folder': p1_parent_id})

        if ptype in ['fs', 'fd']:  # 아빠와 자식의 관계
            family_map[parent_id]['father'] = p1_man_id
        elif ptype in ['ms', 'md']:  # 엄마와 자식의 관계
            family_map[parent_id]['mother'] = p1_man_id

    # Step 4: family_map을 사용하여 가족 구성 생성 및 폴더 설정
    for parent_id, members in family_map.items():
        current_family_id = parent_id.split("_")[0]

        if last_family_id != current_family_id:
            current_family_number = 1
        else:
            current_family_number += 1

        last_family_id = current_family_id

        father_id = members['father']
        mother_id = members['mother']
        child_id = members['child']
        parent_folder = members['parent_folder']

        # 아빠, 엄마, 자식이 있는 경우만 처리
        if father_id and mother_id:
            family_id = f"{current_family_id}_{current_family_number}"

            # 가족 폴더 생성
            family_folder = output_dir / family_id
            family_folder.mkdir(parents=True, exist_ok=True)

            # 아빠, 엄마, 자식 폴더 설정
            father_folder = family_folder / "father"
            mother_folder = family_folder / "mother"
            child_folder = family_folder / "child"

            # 각각의 폴더 생성
            father_folder.mkdir(exist_ok=True)
            mother_folder.mkdir(exist_ok=True)
            child_folder.mkdir(exist_ok=True)

            # 부모와 자식의 이미지 목록을 불러오기
            father_imgs = list((base_dir / split / 'data' / parent_folder / father_id).glob('*')) if father_id else []
            mother_imgs = list((base_dir / split / 'data' / parent_folder / mother_id).glob('*')) if mother_id else []
            child_imgs = list((base_dir / split / 'data' / parent_folder / child_id).glob('*'))

            # 부모와 자식 간의 이미지 수를 맞추기 위해 가장 많은 수로 맞춤
            max_images = max(len(father_imgs), len(mother_imgs), len(child_imgs))

            # 이미지 비틀기 변환 함수 (좌우 회전 없이 작은 변형)
            def distort_image(image_path):
                img = cv2.imread(str(image_path))
                rows, cols, _ = img.shape

                # 이미지에 약간의 비틀림 적용 (Affine 변환 행렬을 사용, 좌우 회전 없음)
                src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
                random_offset = random.uniform(-0.03, 0.03)  # 랜덤하게 아주 작은 값으로 비틀기
                dst_points = np.float32([[cols * random_offset, rows * random_offset],
                                         [cols * (1 - random_offset), rows * random_offset],
                                         [cols * random_offset, rows * (1 - random_offset)]])
                affine_matrix = cv2.getAffineTransform(src_points, dst_points)

                # 이미지 변환 적용
                distorted_img = cv2.warpAffine(img, affine_matrix, (cols, rows))
                return distorted_img

            # 성별에 따라 man 또는 woman을 파일 이름에 포함
            def copy_images(image_list, target_folder, max_count, prefix, gender):
                for i in range(max_count):
                    img_path = image_list[i % len(image_list)]  # 중복된 이미지를 사용
                    if i < len(image_list):  # 원본 이미지를 그대로 복사
                        shutil.copy(img_path, target_folder / f"{prefix}_{gender}_{i + 1}.jpg")
                    else:  # 부족한 이미지에 대해서만 비틀기 적용
                        distorted_img = distort_image(img_path)
                        target_path = target_folder / f"{prefix}_{gender}_{i + 1}.jpg"
                        cv2.imwrite(str(target_path), distorted_img)

            # 아빠, 엄마, 자식 이미지 복사 (성별에 따라 man 또는 woman 추가)
            if father_imgs:
                copy_images(father_imgs, father_folder, max_images, "father", "man")
            if mother_imgs:
                copy_images(mother_imgs, mother_folder, max_images, "mother", "woman")
            if child_imgs:
                # 자식 성별은 ptype을 기반으로 추정 (fs, ms = son, fd, md = daughter)
                child_gender = "man" if any(ptype in ['fs', 'ms'] for ptype in filtered_df['ptype']) else "woman"
                copy_images(child_imgs, child_folder, max_images, "child", child_gender)
                print(child_folder)