# AD_Model
#필요한 라이브러리 설치
!pip install torch torchvision numpy scikit-learn matplotlib pandas

#Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

#RD4AD 리포지토리 클론
!git clone http://github.com/hq-deng/RD4AD.git
%cd RD4AD

#데이터 경로 설정
data_path = '/content/drive/MyDrive/RD4AD_data'

#main.py 파일 생성 및 수정
main_py_content = """
import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


#커스텀 데이터셋 클래스
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(CustomImageFolder, self).__getitem__(index)
        except UnidentifiedImageError:
            print(f"Skipping corrupt image: {self.imgs[index][0]}")
            return self.__getitem__((index + 1) % len(self.imgs))

def calculate_mean_std(loader):
  #데이터셋의 mean과 std 계산하는 함수
  mean = 0.0
  std = 0.0
  total_images_count = 0
  for images, _ in loader:
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)
      total_images_count += batch_samples

  # 평균과 표준편차 계산
  mean /= total_images_count
  std /= total_images_count
  return mean, std


def main(data_path, model_type, epochs):
  #데이터 디렉토리 확인
  data_file = os.path.join(data_path, 'mvtec_anomaly_detection')
  train_dir = os.path.join(data_file, 'train')
  test_dir = os.path.join(data_file, 'test')

  if not os.path.exists(train_dir) or not os.path.exists(test_dir):
      raise FileNotFoundError(f"The directories '{train_dir}' or '{test_dir}' does not exist.")

  # 데이터 로드 및 전처리
  print("데이터 로드 및 전처리 시작")


  initial_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
  ])
  trainset = datasets.ImageFolder(root=train_dir, transform=initial_transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

  mean, std = calculate_mean_std(trainloader)
  print(f"Calculated mean: {mean}, std: {std}")

  transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
  ])

  trainset.transform = transform
  testset = datasets.ImageFolder(root=test_dir, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
  print("데이터 로드 및 전처리 완료")

  #데이터셋에서 클래스 수 확인
  num_classes = len(trainset.classes)
  print(f"Number of classes in the dataset: {num_classes}")


  # 모델 정의 및 컴파일
  print("모델 정의 및 컴파일 시작")

  class SimpleCNN(nn.Module):
      def __init__(self, num_classes):
          super(SimpleCNN, self).__init__()
          self.conv1 = nn.Conv2d(3, 32, 3, 1)
          self.conv2 = nn.Conv2d(32, 64, 3, 1)
          self.pool = nn.MaxPool2d(2, 2)
          self.fc1 = nn.Linear(64 * 62 * 62, 128)
          self.fc2 = nn.Linear(128, num_classes)

      def forward(self, x):
          x = self.pool(torch.relu(self.conv1(x)))
          x = self.pool(torch.relu(self.conv2(x)))
          x = torch.flatten(x, 1)
          x = torch.relu(self.fc1(x))
          x = self.fc2(x)
          return x

  model = SimpleCNN(num_classes=num_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  print("모델 정의 및 컴파일 완료")

  #모델 학습
  print("모델 학습 시작")

  train_losses = []

  def train_model(model, trainloader, criterion, optimizer, epochs):
      for epoch in range(epochs):
          running_loss = 0.0
          for i, data in enumerate(trainloader, 0):
              inputs, labels = data
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              if i % 100 == 99:
                  print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                  train_losses.append(running_loss / 100)
                  running_loss = 0.0
      print("Finsihed Training")


  train_model(model, trainloader, criterion, optimizer, epochs)
  
  print("모델 학습 완료")

  #모델 평가

  print("모델 평가 시작")

  def evaluate_model(model, testloader):
      correct = 0
      total = 0
      with torch.no_grad():
          for data in testloader:
              images, labels = data
              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      accuracy = 100 * correct / total
      print(f'Accuracy of the network on the test images: {accuracy} %')
      return accuracy

  accuracy = evaluate_model(model, testloader)
  print("모델 평가 완료")

  #결과 시각화 및 출력

  print("결과 시각화 및 출력 시작")
  plt.figure(figsize=(10, 5))
  plt.title("Training Loss")
  plt.plot(train_losses, label="Training Loss")
  plt.xlabel("Iterations")
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

  print(f"Final Accuracy: {accuracy}%")
  print("결과 시각화 및 출력 완료")

if __name__== "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
  #필요한 다른 인자들을 추가
   parser.add_argument('--model_type', type=str, help='Type of the model')
   parser.add_argument('--epochs', type=int, help='Number of epochs to train')
   args = parser.parse_args()

   main(args.data_path, args.model_type, args.epochs)
"""
with open("main.py", "w") as file:
    file.write(main_py_content)

 #모델 학습 및 테스트 실행
!python main.py  --data_path "$data_path" --model_type CNN --epochs 10
