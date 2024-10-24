import os
from PIL import Image
from torch.utils.data import Dataset

class FamilyDataset(Dataset):
    def __init__(self, root_dir, transform=None, filter_by_id=True, include_R=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.include_R = include_R  # R을 포함할지 여부를 결정하는 플래그 변수
        processed_families = set()  # 처리된 가족 ID를 추적하기 위한 집합

        # 각 가족 폴더 탐색
        for family_folder in os.listdir(root_dir):
            # include_R이 False일 경우 폴더 이름이 F로 끝나는 경우만 처리
            if not self.include_R and family_folder.endswith('_R'):
                continue  # R이 포함된 폴더는 건너뜀

            family_path = os.path.join(root_dir, family_folder)

            if os.path.isdir(family_path):
                # 가족 ID 추출 (예: F0001)
                family_id = family_folder.split('_')[0]

                # 이미 처리한 가족 ID는 건너뜀
                if filter_by_id and family_id in processed_families:
                    continue

                # 해당 가족 ID를 처리했다고 기록
                processed_families.add(family_id)

                # 폴더 경로 설정
                father_path = os.path.join(family_path, "father")
                mother_path = os.path.join(family_path, "mother")
                child_path = os.path.join(family_path, "child")
                other_path = os.path.join(family_path, "other")

                # father, child, other 폴더 내 파일 리스트 정렬하여 동일 인덱스의 이미지를 페어링
                father_images = sorted(os.listdir(father_path))
                mother_images = sorted(os.listdir(mother_path))  # 같은 이미지를 사용할 수도 있지만 유지
                child_images = sorted(os.listdir(child_path))
                other_images = sorted(os.listdir(other_path))

                # 이미지 개수가 같다고 가정하고 페어링
                for i in range(len(child_images)):
                    father_image_path = os.path.join(father_path, father_images[i])
                    mother_image_path = os.path.join(mother_path, mother_images[0])  # 한 장만 사용할 경우
                    child_image_path = os.path.join(child_path, child_images[i])
                    other_image_path = os.path.join(other_path, other_images[i])

                    # 이미지 열기 및 변환
                    father_image = Image.open(father_image_path).convert('RGB')
                    mother_image = Image.open(mother_image_path).convert('RGB')
                    child_image = Image.open(child_image_path).convert('RGB')
                    other_image = Image.open(other_image_path).convert('RGB')

                    # child 및 other 이미지의 성별 판단 (파일명에 "man" 포함 여부로 결정)
                    child_gender = 0 if "man" in child_images[i] else 1  # 0: man, 1: woman
                    other_gender = 0 if "man" in other_images[i] else 1  # 0: man, 1: woman

                    # 데이터 리스트에 추가
                    self.data.append({
                        'family_folder': family_folder,  # 가족 폴더 이름도 저장
                        'father_image': father_image,
                        'mother_image': mother_image,
                        'child_image': child_image,
                        'other_image': other_image,
                        'child_gender': child_gender,
                        'other_gender': other_gender
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        father_image = sample['father_image']
        mother_image = sample['mother_image']
        child_image = sample['child_image']
        other_image = sample['other_image']
        child_gender = sample['child_gender']
        other_gender = sample['other_gender']
        family_folder = sample['family_folder']  # 저장된 가족 폴더 정보 가져오기

        # family_folder가 '_R'로 끝나는 경우 True 반환 (R 폴더 구분)
        is_same = family_folder.endswith('_R')

        if self.transform:
            father_image = self.transform(father_image)
            mother_image = self.transform(mother_image)
            child_image = self.transform(child_image)
            other_image = self.transform(other_image)

        return father_image, mother_image, child_image, other_image, child_gender, other_gender, is_same
