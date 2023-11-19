import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

batch_size = 64
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(38, 512)
        self.layer2 = nn.Linear(512 + 2, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 1)

    def forward(self, s, a):
        hidden_layer_1 = F.relu(self.layer1(s))
        hidden_layer_1_a = torch.cat((hidden_layer_1, a), 1)
        hidden_layer_2 = F.relu(self.layer2(hidden_layer_1_a))
        hidden_layer_3 = F.relu(self.layer3(hidden_layer_2))
        return self.layer4(hidden_layer_3)


class PolicyNetGaussian(nn.Module):
    def __init__(self):
        super(PolicyNetGaussian, self).__init__()
        self.layer1 = nn.Linear(38, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer_4_mean = nn.Linear(512, 2)
        self.layer_4_standard_log = nn.Linear(512, 2)

    def forward(self, s):
        hidden_layer_1 = F.relu(self.layer1(s))
        hidden_layer_2 = F.relu(self.layer2(hidden_layer_1))
        hidden_layer_3 = F.relu(self.layer3(hidden_layer_2))
        return self.layer_4_mean(hidden_layer_3), torch.clamp(self.layer_4_standard_log(hidden_layer_3),
                                                              min=LOG_SIG_MIN, max=LOG_SIG_MAX)

    def sample(self, s):
        a_mean, standard_log = self.forward(s)
        a_std = standard_log.exp()
        flow = Normal(a_mean, a_std)
        position_x = flow.rsample()
        A_ = torch.tanh(position_x)
        log_prob = flow.log_prob(position_x) - torch.log(1 - A_.pow(2) + epsilon)
        return A_, log_prob.sum(1, keepdim=True), torch.tanh(a_mean)


class SAC:
    def __init__(self, model, b_size=64):
        self.tau = 0.01
        self.alpha = 0.5
        self.criterion = nn.MSELoss()
        self.n_actions = 2
        self.lr = [0.0001, 0.0001]
        self.gamma = 0.99
        self.memory_size = 10000
        self.batch_size = b_size
        self.memory_counter = 0
        self.recollection = {"s": [], "a": [], "r": [], "sn": [], "end": []}
        self.actor = model[0]()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic = model[1]()
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.c_net_target = model[1]()
        self.c_net_target.eval()
        self.t_entropy = -torch.Tensor(self.n_actions)
        self.A_log = torch.zeros(1, requires_grad=True)
        self.A_optim = optim.Adam([self.A_log], lr=0.0001)

    def save_load_model(self, op, path):
        actor_net_path = path + "A_SAC.pt"
        critic_net_path = path + "C_SAC.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), critic_net_path)
            torch.save(self.actor.state_dict(), actor_net_path)
        elif op == "load":
            self.critic.load_state_dict(torch.load(critic_net_path))
            self.c_net_target.load_state_dict(torch.load(critic_net_path))
            self.actor.load_state_dict(torch.load(actor_net_path))

    def choose_action(self, s, eval=False):
        state_to_be = torch.FloatTensor(np.expand_dims(s, 0))
        if not eval:
            action, _, _ = self.actor.sample(state_to_be)
        else:
            _, _, action = self.actor.sample(state_to_be)

        action = action.cpu().detach().numpy()[0]
        return action

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:
            self.recollection["s"].append(s)
            self.recollection["a"].append(a)
            self.recollection["r"].append(r)
            self.recollection["sn"].append(sn)
            self.recollection["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.recollection["s"][index] = s
            self.recollection["a"][index] = a
            self.recollection["r"][index] = r
            self.recollection["sn"][index] = sn
            self.recollection["end"][index] = end

        self.memory_counter += 1

    def softie(self):
        with torch.no_grad():
            for target_p, eval_p in zip(self.c_net_target.parameters(), self.critic.parameters()):
                target_p.copy_((1 - self.tau) * target_p.data + self.tau * eval_p.data)

    def learn(self):
        if self.memory_counter < self.memory_size:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        s_batch = [self.recollection["s"][index] for index in sample_index]
        a_batch = [self.recollection["a"][index] for index in sample_index]
        r_batch = [self.recollection["r"][index] for index in sample_index]
        sn_batch = [self.recollection["sn"][index] for index in sample_index]
        end_batch = [self.recollection["end"][index] for index in sample_index]

        s_ts = torch.FloatTensor(np.array(s_batch))
        a_ts = torch.FloatTensor(np.array(a_batch))
        r_ts = torch.FloatTensor(np.array(r_batch)).unsqueeze(1)
        sn_ts = torch.FloatTensor(np.array(sn_batch))
        end_ts = torch.FloatTensor(np.array(end_batch)).unsqueeze(1)

        with torch.no_grad():
            a_next, policy_next, _ = self.actor.sample(sn_ts)
            q1 = self.c_net_target(sn_ts, a_next)
            # print(q1.shape)
            q_target = r_ts + end_ts * self.gamma * q1 - self.alpha * policy_next
        # print((end_ts * self.gamma * q1).shape)
        self.c_loss = self.criterion(self.critic(s_ts, a_ts), q_target)

        self.critic_optim.zero_grad()
        self.c_loss.backward()
        self.critic_optim.step()

        a_curr, policy_current, _ = self.actor.sample(s_ts)
        val_current = self.critic(s_ts, a_curr)
        self.a_loss = ((self.alpha * policy_current) - val_current).mean()

        self.actor_optim.zero_grad()
        self.a_loss.backward()
        self.actor_optim.step()

        self.softie()

        alpha_loss = -(self.A_log * (policy_current + self.t_entropy).detach()).mean()
        self.A_optim.zero_grad()
        alpha_loss.backward()
        self.A_optim.step()

        self.alpha = float(self.A_log.exp().detach().cpu().numpy())
        return float(self.a_loss.detach().cpu().numpy()), float(self.c_loss.detach().cpu().numpy())
