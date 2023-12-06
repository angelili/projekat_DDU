# Master thesis
Implementation of Personalized FL w Moreau Envelopes in Flower

This directory contains the code from the Master's thesis that addresses the problem of Personalized Federated Learning.
The main algorithm is the pFedMe, from the following paper : [Personalized Federated Learning with Moreau
Envelopes](https://arxiv.org/pdf/2006.08848.pdf). The pracitcal novelty of this thesis is its implementation in [Flower](https://flower.dev/).
This framework simplifies the development and deployment of FL(Federated Learning) systems. The thesis also provides implementations of FedAvg, and of a modified version of pFedMe, pFedMe_new.
The focus is on the heterogeneity among data distributions of clients, since that is the key issue that Personalized FL tackles. 
First a brief overview of pFedMe, and pFedMe_new:
# An Introduction to pFedMe, pFedMe_new


Alongside the strandard loss of the client, here we optimize also for the penalty term, which controls the personalized model $\theta$ of the client remaining close to the global model x. Mathematically the main objective is:
$$\min_{x\in \mathbb{R}^d} { F(x) := \frac{1}{n} \sum_{i=1}^n F_i(x) }$$
$$F_i(x) := \min_{\theta_i \in \mathbb{R}^d} { f_i(\theta_i) + \frac{\lambda}{2} \||\theta_i - x \||^2 }$$


$$\tilde{f}_i(\theta_i;\mathcal{D}_i) + \frac{\lambda}{2} || \theta _i -  x _{i,r}^t||^2$$


So pFedme algorithm is outlined as follows:


*  At a global round $t= 0,\ldots, T-1$ server broadcasts the global model $x_t$ to each client $i$
*  At local round, the local model is set $r=0$ $x_{i,0}^t=x_t$



*  At a local round $r=0,\ldots, R-1$ the client samples a batch $D_i$ and takes for $k=0, \ldots, K-1$ steps using some  optimizer:


$$\tilde{f}_i(\theta_i;\mathcal{D}_i) + \frac{\lambda}{2} || \theta _i -  x _{i,r}^t||^2$$

*   After K iterations obtains $\tilde{\theta}_ i(x_{i,r}^t)$ the personliazed model approximate
*   Computes the new local model which signifies the end of one local round!
   $$x_{i,r+1}^t = x_{i,r}^t-\eta\underbrace{\lambda(x_{i,r}^t-\tilde{\theta}_ i(x_{i,r}^t)}_ {:=\nabla F_i(x_{i,r}^t)}$$

* Server uniformly samples a subset of clients $\mathcal{S}^t$ with size $S$, each of the sampled client sends the \textbf{local} model $x_{i,R}^t$ to the server


*    Server updates the global model: $x_{t+1}=(1-\beta)x_t + \beta \sum_{i \in \mathcal{S}^t} \frac{x_{i,R}^t}{S}$

The difference between pFedMe and pFedMe_new is in local iterations & batch sampling. pFedMe_new takes a new batch every inner iteration, while pFedMe takes a batch and performs iterations on it.

In this experimentation, there are 10 different clients. Each client has its dataset, and its client.py.  Each client was trained on a CUDA partition node, this is specified in client.sh while the main,access node of the cluster hosted the server. The setup of client.sh files is tailored based on Faculty of Sciences, computer cluster Axiom, and its CUDA nodes.

# Project setup
pip
Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.
```
pip install -r requirements.txt
```
Then in every experiment scenario with FL in this directory, we start the server in a terminal as follows:
```
python3 server.py
```
After the server is up and running, we invoke the clients with:
```
./run.sh
```
NOTE: Initializing the server leads to bottleneck of data download for some reason in the FashionMNIST, I suggest running isolation.py from one of the fedavg, and then stopping after the download has been completed.

# Project internal setup
The two 'flags', FED_BN, Benchmark are set in `general_mnist.py`. The flag FED_BN is based on [FedBN] strategy , which suggests that Batch Normalization layers on clients could become biased and lead to overfitting the local model when data heterogeneity takes place,
so in FedBN, those layers are ignored while sending and recieveing the parameters. In my experimentation implementing this `FedBN=True` produced worse results, since data heterogenity is not so strong. So I suggest leaving `FedBN=False`.
Benchmarking flag is used for benchmarking purposes. If `Benchmark=True`, in pfedme, pfedme_new, we will use the datasets on clients from fedavg scenario, we will make comparison plots etc. If `Benchmark=True` newdatasets are loaded, and no comparison plots are made.

# Project modules
Two most important modules are `general_server.py` and `general_mnist.py`, alongside `client.py`.
In the `general_server.py` we have : `load_data_server()` yielding `testset_server` , which will be placed on the server. This is only necessary for building research, since in really the server should not contain any private data. 
The server coordinates the evaluation based on clients' data in the setup. Those evaluation functions are defined here. The main idea is to initialize a dictionary, that for keys, has accuracies, the value of the key is a an empty list, that in every round gets filled with respective accuracy. The functions are used to calculate for example, the average of local accuracies obtained from each client, and to fill the dictionary.
`general_mnist.py` contains a neural network model class (Net), training functions for two different federated learning scenarios (train_pfedme and train_fedavg). Both of the training functions will be used in their respective scenarios. This leads us to `client.py`, where we define the `MnistClient`.
# Heart of FL, the server
Throughout all setups, we have a `server.py` 

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](https://github.com/angelili/projekat_DDU/blob/main/LICENSE) file for details.

## Acknowledgments

This project includes code from the [Flower project](https://github.com/adap/flower) on GitHub. I give my best gratitude for their contributions to open source.

### Disclaimer

[Projekat_DDU](https://github.com/angelili/projekat_DDU) is endorsed by the [Marvel project](https://www.marvel-project.eu/). This project has received funding from the European Union’s Horizon 2020 Research and Innovation program under grant agreement No 957337. [Projekat_DDU](https://github.com/angelili/projekat_DDU) was used as a part of the [TaRDIS project](https://www.project-tardis.eu/), which has received funding under granta agreement No 101093006, by the Swiss State Secretariat for Education, Research and Innovation (SERI).

