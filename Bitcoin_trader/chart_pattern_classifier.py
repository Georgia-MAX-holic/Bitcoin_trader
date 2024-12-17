from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
class ChartPatternClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ChartPatternClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 첫 번째 합성곱 계층
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 두 번째 합성곱 계층
        self.fc1 = nn.Linear(64 * 32 * 32, 128)                            # 첫 번째 완전 연결층
        self.fc2 = nn.Linear(128, num_classes)                             # 출력층
        
    def forward(self, x):
        x = F.relu(self.conv1(x))    # 활성화 함수로 ReLU 사용
        x = F.max_pool2d(x, 2)       # 최대 풀링 계층
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 32 * 32) # 텐서를 평탄화
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 모든 이미지를 128x128로 크기 조정
    transforms.ToTensor(),          # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 데이터셋 불러오기
dataset = datasets.ImageFolder("dataset", transform=transform)

# 데이터 로더 생성
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 초기화
num_classes = len(dataset.classes)
model = ChartPatternClassifier(num_classes=num_classes)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # 에포크 수를 설정합니다 (예: 10 에포크)
    for images, labels in dataloader:
        optimizer.zero_grad()          # 옵티마이저 초기화
        outputs = model(images)        # 모델에 이미지 입력
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()                # 역전파
        optimizer.step()               # 가중치 업데이트
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'model.pth')