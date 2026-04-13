import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# 1. 아까 설명한 데이터셋 불러오기
from dataset import InstancePointCloudDataset
# 2. 여러분이 짤 모델 불러오기
from model import DummyModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [여기에 들어갑니다!] 데이터셋 초기화
    # 미리 data_generator.py를 통해 data/train_scenes 폴더에 npy 파일들을 잔뜩 만들어 두어야 합니다.
    train_dataset = InstancePointCloudDataset(data_dir="./data", split="train")
    
    # DataLoader를 이용해 배치(Batch) 단위로 쪼개기
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # 모델 초기화 (DummyModel 안에 실제 네트워크 구조를 구현해야 함)
    model = DummyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 데이터 로더에서 배치 단위로 데이터 뽑기
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # batch["features"]의 모양: [B, 9, N]
            features = batch["features"].to(device)
            # batch["instance_labels"]의 모양: [B, N]
            gt_labels = batch["instance_labels"].to(device)

            optimizer.zero_grad()

            # [주의!] 여기서 모델의 '학습용' forward를 호출해야 합니다.
            # (model.py에 있는 run_inference 함수는 '평가용'이므로 학습할 땐 안 씁니다.)
            pred_logits = model(features) 

            # Loss 계산 (Semantic loss, Clustering loss 등 여러분이 설계)
            # loss = compute_loss(pred_logits, gt_labels)
            # loss.backward()
            # optimizer.step()

            # total_loss += loss.item()
        
        # print(f"Epoch {epoch} Loss: {total_loss/len(train_loader)}")

        # 체크포인트 저장 (이 저장된 파일을 나중에 initialize_model에서 부르게 됨)
        # torch.save(model.state_dict(), f"best_model.pth")

if __name__ == "__main__":
    train()