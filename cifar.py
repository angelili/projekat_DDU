
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import MNIST
import copy
import random

DATA_ROOT = "./dataset"



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
# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

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
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    trainset = MNIST(DATA_ROOT, train=True, download=True, transform=transform)

    sample_size=500
    indices = random.sample(range(len(trainset)), sample_size)
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, sampler=sampler)

    testset = MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    sample_size=100
    indices = random.sample(range(len(testset)), sample_size)
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, sampler=sampler)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    
    return trainloader, testloader, testset, num_examples


#training the personalized network which, infacts trains the one that goes to the global 
def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    eta: float,
    lambda_reg: float  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
  

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    local_model=copy.deepcopy(net).to(device)

    local_model.train()
    for epoch in range(epochs):
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)# loop over the dataset multiple times
        correct, total, epoch_loss = 0, 0, 0.0   
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            # zero the parameter gradients
            objective, loss, output, target = objective_function(local_model, net, 15, images, labels)
            objective.backward()
            optimizer.step()
            # forward + backward + optimize
         
            epoch_loss += loss
            total += target.size(0)
            correct += (torch.max(output.data, 1)[1] == target).sum().item()
                          
            # Check if the gradient norm is below a threshold
            if gradient_norm_stop_callback(threshold=1e-5)(optimizer):
                    break
            # print statistics
          
        if i % 100 == 99:  # print every 100 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss / 2000))
        with torch.no_grad():
            for param, global_param in zip(local_model.parameters(), net.parameters()):
                global_param.data=global_param.data-eta*lambda_reg*(global_param.data-param.data)
      
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")     
    return net, local_model

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


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _, _ = load_data()
    net = Net().to(DEVICE)
 
    print("Start training")
    net, local_model = train(net=net, trainloader=trainloader, epochs=2, device=DEVICE, eta=0.005, lambda_reg=15)
    print("Evaluate model")
    loss_global, accuracy_global= test_global(net=net, testloader=testloader, device=DEVICE)
    loss_person, accuracy_person= test_local(local_model=local_model, testloader=testloader, device=DEVICE)
    print("Loss_personalized: ", loss_person)
    print("Accuracy_personalized: ", accuracy_person)
    print("Loss_global: ", loss_global)
    print("Accuracy_global: ", accuracy_global)

if __name__ == "__main__":
    main()
