
## Report

## Agent is using Proximal Policy Optimization Algorithm. 

Proximal Policy Optimization:
PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small.

This algorithm uses function aproximator in a form of neural network. 

The first couple layers calculate policy distribution and return log(policy)
Actions are sampled from the gaussian distribution built upon mean and std

Value is calculated from a separate head in the network Critic network part

The implemenation is based on the implementation from the paper but it uses Huber-Loss loss function to calculate the cost(loss). In my experiments Huber-Loss had better performance over standard loss functions.

On average my best score was between around 38 points

Config also includes all hyperparemeters which I found to work best.

## Model 

Model consists of 2 layers per each component (body and head), body normally has 400 and head 300 nodes, details of the model are in the table below. 

|        Layer (type)   |           Output Shape   |      Param #|
| --- | --- | --- | 
|Input                 | [-1, 1, 400]      |    13,600
|            Hidden     |          [-1, 1, 300]   |      120,300
|            Policy     |          [-1, 1, 300]    |      90,300
|            Actions     |            [-1, 1, 4]    |       1,204
|            Critic     |          [-1, 1, 300]    |      90,300
|            Value     |            [-1, 1, 1]    |         301

Total params: 316,005
Trainable params: 316,005
Non-trainable params: 0

## Training chart: 
![](/images/Reacher-Udacity.png)

## Parameters used (please see config.py): 
### Agent / network specific

 gae_tau = 0.95  
 gradient_clip = 4.7         - network clipping
 rollout_length = 1000       - trajectory length when recording actions / states
 optimization_epochs = 10    - training length
 mini_batch_size = 200       - batch size in training
 ppo_ratio_clip = 0.1        - gradient clipping value as per paper left 0.2
 lr = 0.001                  - initial learning rate decayed over time

## Future improvements: 
Better algorithm such as Soft Actor Critic could perform better. 
