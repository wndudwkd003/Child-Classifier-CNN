import os
from PIL import Image
from torch.utils.data import Dataset


class FamilyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # 각 F1, F2 폴더 탐색
        for family_folder in os.listdir(root_dir):
            family_path = os.path.join(root_dir, family_folder)
            if os.path.isdir(family_path):
                father_path = os.path.join(family_path, "father")
                mother_path = os.path.join(family_path, "mother")
                child_path = os.path.join(family_path, "child")

                # father, mother, child 이미지 파일 불러오기
                father_image = self._load_image_from_folder(father_path)
                mother_image = self._load_image_from_folder(mother_path)

                for child_image_name in os.listdir(child_path):
                    child_image_path = os.path.join(child_path, child_image_name)

                    # 성별 판단 (파일 이름에 'man'이 있으면 남자, 'woman'이 있으면 여자)
                    if "man" in child_image_name:
                        gender = 0  # 남성
                    elif "woman" in child_image_name:
                        gender = 1  # 여성

                    child_image = Image.open(child_image_path).convert('RGB')

                    # 데이터 리스트에 추가
                    self.data.append({
                        'father_image': father_image,
                        'mother_image': mother_image,
                        'child_image': child_image,
                        'gender': gender
                    })

    def _load_image_from_folder(self, folder_path):
        # 폴더 내 이미지 하나만 사용한다고 가정
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        father_image = sample['father_image']
        mother_image = sample['mother_image']
        child_image = sample['child_image']
        gender = sample['gender']

        if self.transform:
            father_image = self.transform(father_image)
            mother_image = self.transform(mother_image)
            child_image = self.transform(child_image)

        return father_image, mother_image, child_image, gender

