import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple, Dict
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST


DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"

def load_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(),  transforms.Normalize((0.2859), (0.3530))]
    )

    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)


    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // 10
    lengths = [partition_size] * 10
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    testloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        testloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, testloaders, testloader


trainloaders, testloaders, testloader = load_datasets() 
        
class Net(nn.Module):
  
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4* 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

#training for fedavg
def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)


    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            
            

def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


#same funnction as test, different name as a reminder when implelemnting pfed_me, or pfed_me_new
def test_global(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct_global, loss_global =  0, 0.0
    # Evaluate the network which will participate in the global
    net.to(device)
    net.eval()
    with torch.no_grad():
      for data in testloader:
          images, labels = data[0].to(device), data[1].to(device)
          outputs = net(images)
          loss_global += criterion(outputs, labels).item()
          _, predicted_global = torch.max(outputs.data, 1)  
          correct_global += (predicted_global == labels).sum().item()
    accuracy_global = correct_global / len(testloader.dataset)


    return loss_global, accuracy_global

#same logic
def test_local(
  local_model: Net,
  testloader: torch.utils.data.DataLoader,
  device: torch.device,  #
  ) -> Tuple[float, float]:
  """Validate the network on the entire test set."""
  # Define loss and metrics
  criterion = nn.CrossEntropyLoss()

  correct_person, loss_person=0, 0.0
  # Evaluate the personalized network

  local_model.to(device)
  local_model.eval()
  with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = local_model(images)
        loss_person += criterion(outputs, labels).item()
        _, predicted_person = torch.max(outputs.data, 1)  #
        correct_person += (predicted_person == labels).sum().item()
  accuracy_person = correct_person / len(testloader.dataset)


  return loss_person, accuracy_person

#check if one class is dominant forexample

def partition_check(dataset):
  class_counts = torch.zeros(10)  # Assuming there are 10 classes in the dataset

  for _, label in dataset:
      class_counts[label] += 1

  total_samples = len(dataset)

  for class_idx, count in enumerate(class_counts):
      percentage = (count / total_samples) * 100
      print(f"Class {class_idx}: {count} samples, {percentage:.2f}%")