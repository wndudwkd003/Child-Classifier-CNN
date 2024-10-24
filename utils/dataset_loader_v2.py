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
output_dir = Path('../dataset_v2')  # 최종 저장할 경로
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


        if ptype in ['fs', 'ms']:       # 자식 성별 지정
            family_map[parent_id]['child_gender'] = "man"

        elif ptype in ['fd', 'md']:
            family_map[parent_id]['child_gender'] = "girl"




    # 모든 family_id 목록을 수집 (랜덤 선택용)
    all_family_ids = list(family_map.keys())

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
        child_gender = members['child_gender']

        # 아빠, 엄마, 자식이 있는 경우만 처리
        if father_id and mother_id:
            # R (Real) 폴더 생성
            family_id_R = f"{current_family_id}_{current_family_number}_R"
            family_folder_R = output_dir / family_id_R
            family_folder_R.mkdir(parents=True, exist_ok=True)

            # F (False) 폴더 생성
            family_id_F = f"{current_family_id}_{current_family_number}_F"
            family_folder_F = output_dir / family_id_F
            family_folder_F.mkdir(parents=True, exist_ok=True)

            # 아빠, 엄마, 자식 폴더 설정
            father_folder_R = family_folder_R / "father"
            mother_folder_R = family_folder_R / "mother"
            child_folder_R = family_folder_R / "child"
            other_folder_R = family_folder_R / "other"  # Real 경우 other에 실제 자녀 복사

            father_folder_F = family_folder_F / "father"
            mother_folder_F = family_folder_F / "mother"
            child_folder_F = family_folder_F / "child"
            other_folder_F = family_folder_F / "other"  # False 경우 other에 랜덤 자녀 복사

            # 각각의 폴더 생성
            father_folder_R.mkdir(exist_ok=True)
            mother_folder_R.mkdir(exist_ok=True)
            child_folder_R.mkdir(exist_ok=True)
            other_folder_R.mkdir(exist_ok=True)

            father_folder_F.mkdir(exist_ok=True)
            mother_folder_F.mkdir(exist_ok=True)
            child_folder_F.mkdir(exist_ok=True)
            other_folder_F.mkdir(exist_ok=True)

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

            # R (Real) 폴더에 이미지 복사
            if father_imgs:
                copy_images(father_imgs, father_folder_R, max_images, "father", "man")
            if mother_imgs:
                copy_images(mother_imgs, mother_folder_R, max_images, "mother", "woman")
            if child_imgs:
                copy_images(child_imgs, child_folder_R, max_images, "child", child_gender)
                copy_images(child_imgs, other_folder_R, max_images, "child", child_gender)  # Real인 경우 other에도 child 복사


            # F (False) 폴더에 여러 가족의 랜덤 자녀 이미지 복사
            def copy_random_child_images(all_family_ids, parent_id, max_images, other_folder_F, prefix, gender):
                selected_images = []

                # 중복되지 않도록 선택할 가족 ID 필터링 (자신의 가족은 제외)
                available_families = []

                for fid in all_family_ids:
                    if current_family_id != fid.split('_')[0]:
                        available_families.append(fid)
                    

                print(available_families)

                while len(selected_images) < max_images:
                    random_family_id = random.choice(available_families)
                    random_family = family_map[random_family_id]
                    random_child_id = random_family['child']
                    random_child_imgs = list(
                        (base_dir / split / 'data' / random_family['parent_folder'] / random_child_id).glob('*'))

                    # 랜덤 자녀 이미지 중에서 하나 선택
                    if random_child_imgs:
                        selected_images.append(random.choice(random_child_imgs))

                # 선택된 이미지들을 other 폴더로 복사 (이미지 수 맞추기)
                for i in range(max_images):
                    img_path = selected_images[i % len(selected_images)]  # 이미지를 순환하면서 사용
                    shutil.copy(img_path, other_folder_F / f"{prefix}_{gender}_{i + 1}.jpg")


            # F (False) 폴더에 랜덤 자녀 이미지 복사
            if father_imgs:
                copy_images(father_imgs, father_folder_F, max_images, "father", "man")
            if mother_imgs:
                copy_images(mother_imgs, mother_folder_F, max_images, "mother", "woman")
            if child_imgs:
                copy_images(child_imgs, child_folder_F, max_images, "child", "man")
            # 랜덤하게 여러 가족에서 자녀 이미지를 복사하여 other 폴더에 넣음
            copy_random_child_images(all_family_ids, parent_id, max_images, other_folder_F, "child", "man")
