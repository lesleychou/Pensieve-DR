'''
An A3C agent implementation, modified for an MORL ABR algorithm.
'''

# TODO:
# - Replace camel case with snake case, where appropriate

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from src.model import model_Pensieve as pensieve

class A3C_Pensieve(object):
    def __init__(self, is_central, s_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3):
        self.s_dim = s_dim
        self.a_dim = action_dim
        self.discount = 0.99
        self.entropy_weight = 0.5
        # Not sure if the entropy epsilon is being used here in this implementation.
        self.entropy_eps = 1e-6

        self.is_central = is_central
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actorNetwork = pensieve.ActorNetwork(self.s_dim, self.a_dim).to(self.device)
        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            self.actorOptim = torch.optim.RMSprop(self.actorNetwork.parameters(), lr=actor_lr, alpha=0.9, eps=1e-10)
            self.actorOptim.zero_grad()
            self.criticNetwork = pensieve.CriticNetwork(self.s_dim, self.a_dim).to(self.device)
            self.criticOptim = torch.optim.RMSprop(self.criticNetwork.parameters(), lr=critic_lr, alpha=0.9,
                                                   eps=1e-10)
            self.criticOptim.zero_grad()
        else:
            self.actorNetwork.eval()

        self.loss_function = nn.MSELoss()

    def getNetworkGradient(self, s_batch, a_batch, r_batch, terminal):
        s_batch = torch.cat(s_batch).to(self.device)
        a_batch = torch.LongTensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)
        R_batch = torch.zeros(r_batch.shape).to(self.device)

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0] - 1)):
            R_batch[t] = r_batch[t] + self.discount * R_batch[t + 1]

        with torch.no_grad():
            v_batch = self.criticNetwork.forward(s_batch).squeeze().to(self.device)
        td_batch = R_batch - v_batch

        probability = self.actorNetwork.forward(s_batch)
        m_probs = Categorical(probability)
        log_probs = m_probs.log_prob(a_batch)
        actor_loss = torch.sum(log_probs * (-td_batch))
        entropy_loss = -self.entropy_weight * torch.sum(m_probs.entropy())
        actor_loss = actor_loss + entropy_loss
        actor_loss.backward()

        critic_loss = self.loss_function(R_batch, self.criticNetwork.forward(s_batch).squeeze())
        # alternate critic loss with td
        # v_batch = self.criticNetwork.forward(s_batch[:-1]).squeeze()
        # next_v_batch = self.criticNetwork.forward(s_batch[1:]).squeeze().detach()
        # critic_loss = self.loss_function(r_batch[:-1] + self.discount * next_v_batch, v_batch)
        critic_loss.backward()

        return actor_loss.detach().numpy().tolist()

    def select_action(self, state):
        """
        Returns action and current entropy based on a state.
        """
        if not self.is_central:
            with torch.no_grad():
                probability = self.actorNetwork.forward(state)
                m = Categorical(probability)
                entropy = m.entropy()
                action = m.sample().item()
                return action, entropy


    def hardUpdateActorNetwork(self, actor_net_params):
        for target_param, source_param in zip(self.actorNetwork.parameters(), actor_net_params):
            target_param.data.copy_(source_param.data)

    def updateNetwork(self):
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actorOptim.step()
            self.actorOptim.zero_grad()
            self.criticOptim.step()
            self.criticOptim.zero_grad()

    def getActorParam(self):
        return list(self.actorNetwork.parameters())

    def getCriticParam(self):
        return list(self.criticNetwork.parameters())
