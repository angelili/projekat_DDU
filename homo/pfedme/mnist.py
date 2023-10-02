

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import FashionMNIST

import random


DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"
Dominant_class=False
Non_uniform_cardinality=False



def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load FashionMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859,), (0.3530,))]
    )
    # Load the FashionMNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    if Non_uniform_cardinality==True:
        sample_size_train = random.randint(4000, 6000)
        sample_size_test =  int(sample_size_train*0.1)
    else:
        sample_size_train=5000
        sample_size_test=500

    indices_train = random.sample(range(len(trainset)), sample_size_train)
    sampler_train= torch.utils.data.SubsetRandomSampler(indices_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, sampler=sampler_train)
    indices_test = random.sample(range(len(testset)), sample_size_test)
    sampler_test = torch.utils.data.SubsetRandomSampler(indices_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, sampler=sampler_test)
    num_examples = {"trainset": sample_size_train, "testset": sample_size_test}

    return trainloader, testloader, testset, num_examples


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


def load_data_server():
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    testset_server = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True)
    

    return testset_server

#training the personalized network which, infacts trains the one that goes to the global
def train(
    model: Net,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    local_epochs: int,
    local_iterations: int,
    local_rounds: int,
    device: torch.device,
    eta: float,
    lambda_reg: float 
) -> None:
    """Train the network."""
       
    # Define the personalized objective function using the Moreau envelope algorithm
    global_params = [val.detach().clone() for val in model.parameters()]
    model.train()
  
    for r in range(local_epochs):
        # Local update on client 
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for r in range(local_rounds):
            data_iterator = iter(trainloader)
            data, target = next(data_iterator)
            data, target = data.to(device), target.to(device) #sample a batch
            for i in range(local_iterations):
                optimizer.zero_grad()
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss = criterion(model(data), target) + (lambda_reg/2) * proximal_term
                loss.backward()
                optimizer.step()


                #update the model
                
            
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_params):
                    global_param.data = global_param.data-eta*lambda_reg*(global_param.data-param.data)
            

    loss_person, accuracy_person= test_local(local_model=model, testloader=testloader, device=device)
    with torch.no_grad():     
        for param, global_param in zip(model.parameters(), global_params):
            param.data = global_param.data
    loss_global, accuracy_global= test_global(net=model, testloader=testloader, device=device)

    print("Accuracy_personalized: ", accuracy_person)
    print("Accuracy_global: ", accuracy_global)

    return model

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, testset, _ = load_data()
    net = Net().to(device)

    print("Start training")
    net= train(net=net, trainloader=trainloader, testloader=testloader, local_epochs=1, local_iterations=10, local_rounds=120, device=device, eta=0.005, lambda_reg=15)
    print("Evaluate model")
  
    loss_person, accuracy_person= test_local(local_model=net, testloader=testloader, device=device)
    print("Loss_personalized: ", loss_person)
    print("Accuracy_personalized: ", accuracy_person)

if __name__ == "__main__":
    main()
