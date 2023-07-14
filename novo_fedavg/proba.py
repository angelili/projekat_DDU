import torch

data = torch.load('/home/s124m21/projekat_DDU/novo_fedavg/data_2.pth')
        # Retrieve the variables
trainloader = data['trainloader']
num_examples = data['num_examples']
testloader = data['testloader']

print(num_examples)


import client_1

trainloader, testloader, _, num_examples = client_1.load_data()

print(num_examples)