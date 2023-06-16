# projekat_DDU
Implementation of Personalized FL w Moreau Envelopes in Flower
https://slideslive.com/38937057/personalized-federated-learning-with-moreau-envelopes?ref=recommended
```
git clone  https://github.com/angelili/projekat_DDU   
```
This will create a new directory called projekat_DDU with the neccesary files.
 To do the setup, we will use Poetry. Poetry simplifies the management of Python projects by providing a unified solution for dependency management, virtual environments. 
Project dependencies (such as torch and flwr) are defined in pyproject.toml and requirements.txt.
```
poetry install
poetry shell
```
Alternative is the standard pip.
```
pip install -r requirements.txt
```
In order to understand how the Federated Learning works, one must comprehend how Centralized Learning works,
and why does FL arise as a better solution in some cases. 
You can simply start the centralized training as described in the tutorial by running cifar.py:
It loads the dataset(MNIST in this case), trains a convolutional neural network (CNN) on the training set, and evaluates the trained model on the test set.
```
python3 cifar.py
```
The results are not good as one would wish. The dataset is small even with the augmentation, and overfits. By using FL, one can leverage a larger and more diverse dataset distributed across multiple devices or users, which helps in reducing overfitting.

To build a federated learning system based on your existing project code, you'll need to create a Flower server and a Flower client. The server will coordinate the federated learning process, and the client will connect your existing model and data to Flower framework.
First, one needs to start the server in a terminal.
```
python3 server.py
```
Assuming that we have a distributed system and that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process. To do so simply open two more terminal windows and run the following scripts, which setup each client on a different GPU node in the computer cluster being used. SErver is being run on the main, access node of the cluster. This approach imitates the true FL setup in reallife scenario.
 ```
 sbatch client.sh
```
Be patient for the FL to get initial parameters and start, however if it stalls, check the submitted clientjobs via squeue.
If the jobs are completed in a very short time cca 14sec, they did not connect with the server. Resubimission via sbatch is needed.
NOTE: the ideally function train from cifar would be used for the client, that is the idea of flower framework, however due to the specific nature of this algorithm with 2 models at one client, this is not the case here.
However, if you do not have the appropriate resources and time, feel free to check the simulation provided, one only has to have a Google Colab account. The simulation is done on the CIFAR dataset.
