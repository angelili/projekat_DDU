from torchvision.datasets.utils import T

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import FashionMNIST
import copy
import random
import numpy as np

DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"



def gradient_norm_stop_callback(threshold=1e-5):
    """
    Callback function to monitor the optimization process and stop it once the gradient norm falls below a certain
    threshold.

    Args:
        threshold (float): Gradient norm threshold. Default is 1e-5.
    """

    def callback_function(optimizer):
        gradient_norm = 0.0
        for group in optimizer.param_groups:

            for param in group['params']:

                if isinstance(param.grad, torch.Tensor):
                    gradient_norm += torch.norm(param.grad)**2
        gradient_norm = gradient_norm.sqrt().item()
        if gradient_norm < threshold:
            print(f'Gradient norm ({gradient_norm:.6f}) is below the threshold ({threshold:.6f}). Stopping optimization.')
            return True

    return callback_function

# Define the personalized objective function using the Moreau envelope algorithm
def objective_function(local_model, global_model, lambda_reg, data, target):
    output = local_model(data)
    loss = F.cross_entropy(output, target)

    local_params = torch.cat([param.view(-1) for param in local_model.parameters()])
    global_params = torch.cat([param.view(-1) for param in global_model.parameters()])

    proba=torch.norm(local_params - global_params)**2

    proba_leaf = torch.tensor(proba, requires_grad=True)
    objective = loss + (lambda_reg/2) * proba_leaf

    return objective, loss, output, target


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

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4* 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x
   


    
    
def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )
    # Load the MNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    selected_classes=[0,1]

    #train
    # Filter the dataset to include only the selected classes
    indices = torch.where(torch.logical_or(trainset.targets == selected_classes[0],
                                        trainset.targets == selected_classes[1]))[0]
    indices=indices.numpy()
    np.random.shuffle(indices)
    num_samples= random.randint(10000,12000)
    indices=indices[:num_samples]
    subset_indices=torch.from_numpy(indices)
    subset_dataset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(subset_dataset, batch_size=16, shuffle=True)
    
    #test
    # Filter the dataset to include only the selected classes
    indices = torch.where(torch.logical_or(testset.targets == selected_classes[0],
                                        testset.targets == selected_classes[1]))[0]
    indices=indices.numpy()
    np.random.shuffle(indices)
    num_samples= int(num_samples*0.1)
    indices=indices[:num_samples]
    subset_indices=torch.from_numpy(indices)
    subset_dataset = torch.utils.data.Subset(testset, subset_indices)
    testloader = torch.utils.data.DataLoader(subset_dataset, batch_size=16, shuffle=True)
    
    num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}

    return trainloader, testloader, testset, num_examples

#training the personalized network which, infacts trains the one that goes to the global
def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    local_epochs: int,
    device: torch.device,
    eta: float,
    lambda_reg: float  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    
    # Define the personalized objective function using the Moreau envelope algorithm
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()

    for r in range(local_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # Local update on client 
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(net(data), target) + (lambda_reg/2) * proximal_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += target.size(0)
            correct += (torch.max(net(data).data, 1)[1] == target).sum().item()

        loss_person, accuracy_person = test_local(local_model= net, testloader=testloader, device=device)
        with torch.no_grad():
            for param, global_param in zip(net.parameters(), global_params):
                global_param.data = global_param.data-eta*lambda_reg*(global_param.data-param.data)

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
            
        print(f"Epoch {r+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    
        
        with torch.no_grad():     
          for param, global_param in zip(net.parameters(), global_params):
                param.data = global_param.data
        loss_global, accuracy_global = test_global(net=net, testloader=testloader, device=device)
        print("Accuracy_personalized: ", accuracy_person)
        print("Accuracy_global: ", accuracy_global)

    return net

def test_global(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
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
          _, predicted_global = torch.max(outputs.data, 1)  # pylint: disable=no-member
          correct_global += (predicted_global == labels).sum().item()
    accuracy_global = correct_global / len(testloader.dataset)


    return loss_global, accuracy_global


def test_local(
  local_model: Net,
  testloader: torch.utils.data.DataLoader,
  device: torch.device,  # pylint: disable=no-member
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
        _, predicted_person = torch.max(outputs.data, 1)  # pylint: disable=no-member
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
    net= train(net=net, trainloader=trainloader, testloader=testloader, local_epochs=300, device=device, eta=0.005, lambda_reg=15)
    print("Evaluate model")
  
    loss_person, accuracy_person= test_local(local_model=net, testloader=testloader, device=device)
    print("Loss_personalized: ", loss_person)
    print("Accuracy_personalized: ", accuracy_person)
   

if __name__ == "__main__":
    main()
