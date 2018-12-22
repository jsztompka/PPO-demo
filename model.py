import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from collections import namedtuple

PolicyModel = namedtuple('PolicyModel',['log_prob', 'entropy', 'action','value'])

class PPONetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PPONetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)


        second_hidden_size = hidden_size - 100
        third = second_hidden_size - 100

        self.input = nn.Linear(state_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, second_hidden_size)

        self.actor_body = nn.Linear(third, third)
        self.actor_head = nn.Linear(third, action_size)

        self.critic_body =  nn.Linear(third, third)
        self.critic_head = nn.Linear(third, 1)

        self.policy_body = nn.Linear(second_hidden_size, third)
        self.policy_head = nn.Linear(third, third)

        init_layers = [self.input, self.hidden, self.actor_body, self.critic_body, self.policy_body]
        self.init_weights(init_layers)

        self.batch_norm = nn.BatchNorm1d(second_hidden_size)
        self.batch_norm_input = nn.BatchNorm1d(hidden_size)

        self.alpha = nn.Linear(third, 4)
        self.beta = nn.Linear(third, 4)
        #
        # # init the networks....
        self.alpha.weight.data.mul_(0.5)
        self.alpha.bias.data.mul_(0.0)

        self.beta.weight.data.mul_(0.5)
        self.beta.bias.data.mul_(0.0)

        # self.alpha_param = nn.Parameter(torch.zeros(4))
        # self.alfa = nn.Parameter(torch.zeros(action_dim))

        self.std = nn.Parameter(torch.zeros(4))

        device = 'cuda:0'
        self.to(device)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.mul_(0.1)


    def forward(self, state, action = None):
        x = state
        x = F.leaky_relu(self.batch_norm_input(self.input(x)))
        x = F.leaky_relu(self.batch_norm(self.hidden(x)))
        x = F.leaky_relu(self.policy_body(x))

        act_x = F.leaky_relu(self.actor_body(x))

        mean = F.tanh(self.actor_head(act_x))

        #alpha = F.softplus(self.alpha(act_x)) + 1
        #beta = F.softplus(self.beta(act_x)) + 1


        # policy distribution
        #policy_dist = torch.distributions.Beta(alpha, beta)

        policy_dist = torch.distributions.Normal(mean, F.softplus(self.std))


        if action is None:
            action = policy_dist.sample()

        log_prob = policy_dist.log_prob(action).sum(-1).unsqueeze(-1)

        entropy = policy_dist.entropy().sum(-1).unsqueeze(-1)
        # entropy = (alpha - beta) / 2

        # critic value
        critic_x = F.leaky_relu(self.critic_body(x))
        value = self.critic_head(critic_x)

        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': value}

        # return PolicyModel(log_prob, entropy, action, value)





