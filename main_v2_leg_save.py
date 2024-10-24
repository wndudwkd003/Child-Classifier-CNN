import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from model.cgan import Generator
from utils.dataset_v2 import FamilyDataset
import os
from tqdm import tqdm

# 시드 고정 함수
def seed_everything(seed=115):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 특정 이미지로 생성된 자녀 이미지 저장 함수
def generate_and_save_legacy(generator, data_loader, device, save_dir="gen_legacy"):
    generator.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        loader = tqdm(data_loader, desc="Generating Images")
        for i, (father_img, mother_img, real_child_img, _, child_gender, _, _) in enumerate(loader):
            # 이미지와 성별 데이터를 장치에 올림
            father_img = father_img.to(device)
            mother_img = mother_img.to(device)
            real_child_img = real_child_img.to(device)
            child_gender = child_gender.to(device)

            # 부모 이미지로부터 생성된 자녀 이미지
            generated_img = generator(father_img, mother_img, child_gender)

            # 부모, 실제 자녀, 생성된 자녀 이미지를 하나로 결합
            combined_images = torch.cat([father_img[:5], mother_img[:5], real_child_img[:5], generated_img[:5]], dim=0)

            # 결합된 이미지를 수평으로 결합하여 저장
            image_grid = make_grid(combined_images, nrow=5, normalize=True)

            # 이미지 저장 (gen_legacy 폴더에 저장)
            save_image(image_grid, os.path.join(save_dir, f"generated_child_{i + 1}.png"))

            # 첫 번째 배치만 생성 후 저장하도록 설정 (모든 데이터셋을 저장하려면 break 제거)
            break

if __name__ == "__main__":
    # 시드 고정
    seed_everything(115)

    # 모델 초기화
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)

    # 모델 가중치 경로 설정
    generator_path = rf"train/best_generator.pth"

    # 저장된 모델 가중치 불러오기
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        print("Loaded Generator weights successfully.")
    else:
        raise FileNotFoundError("No saved generator weights found.")

    # 데이터셋 로드 및 분할
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FamilyDataset(root_dir="dataset_v2", transform=transform)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # train 데이터셋을 이용하여 이미지 생성 및 저장
    generate_and_save_legacy(generator, train_loader, device, save_dir="gen_legacy")
