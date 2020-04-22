import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import math
import hyperparameters as hp
from functions import *
import physics
from classes import distances, is_out
from random import random, sample, randint
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """ Saves a transition """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cuda(hp.DEVICE)
        self.lin1 = nn.Linear(input_dim, 10)
        self.relu = torch.relu
        self.lin2 = nn.Linear(10, 10)
        self.relu2 = torch.relu
        self.action = nn.Linear(10, 9)
        self.relu3 = torch.relu


    def forward(self, inputs):
        output = self.relu(self.lin1(inputs))
        output = self.relu2(self.lin2(output))
        output = self.relu3(self.action(output))
        return output


class RL_Driver():
    def __init__(self, car=None, track=None):
        self.car = car
        self.track = track
        self.prediction_net = DQN(hp.NB_SENSORS+1).to(hp.DEVICE)
        self.target_net = DQN(hp.NB_SENSORS+1).to(hp.DEVICE)
        self.target_net.load_state_dict(self.prediction_net.state_dict())
        self.target_net.eval()  # target net will not be trained
        self.memory = ReplayMemory(10_000)
        self.optimizer = opt.Adam(self.prediction_net.parameters(),
                                  lr=hp.LEARNING_RATE)
        self.steps_done = 0
        self.episode_steps = 0
        self.last_state = None
        self.last_action = None

    def initiate(self, track=None, car=None):
        if car is not None:
            self.car = car
        if track is not None:
            self.track = track

    def sensors(self):
        if is_out(self.track, self.car):
            #TODO: make car see even when not on track
            return torch.tensor([0] * (hp.NB_SENSORS + 1), device=hp.DEVICE,
                                dtype=torch.float32)
        else:
            normalized = [d/500 for d in distances(self.car, self.track,
                                                   nb_angles=hp.NB_SENSORS,
                                                   debugging=True)]
            inputs = [self.car.speed/physics.MAX_SPEED, *normalized]

            return torch.tensor(inputs, device=hp.DEVICE, dtype=torch.float32)


    def decision(self, state):
        """probabilities returned by net correspond to:
           [up, up-right, right, right-down, down, down-left,
            left, left-up]"""
        sample = random()
        eps_threshold = hp.EPS_END + (hp.EPS_START - hp.EPS_END) * \
        math.exp(-1. * self.steps_done * hp.EPS_DECAY)

        if sample > eps_threshold:
            inputs = state
            with torch.no_grad():
                output = self.prediction_net.forward(inputs)
                imax = output.max(0).indices.item()
        else:
            print('taking random action')
            imax = randint(0, 8)

        self.steps_done += 1
        if self.steps_done % hp.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.prediction_net.state_dict())
            print("updating target network")

        return torch.tensor([imax], device=hp.DEVICE)

    def action_to_keys(self, action):
        keys_pressed = []

        if action in (0, 1, 7):
            keys_pressed.append(self.car.up)
        if action in (1, 2, 3):
            keys_pressed.append(self.car.right)
        if action in (3, 4, 5):
            keys_pressed.append(self.car.down)
        if action in (5, 6, 7):
            keys_pressed.append(self.car.left)

        return keys_pressed

    def new_episode(self):
        self.episode_steps = 0


    def train(self):
        if hp.BATCH_SIZE > len(self.memory):
            return

        transitions = self.memory.sample(hp.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=hp.DEVICE)
        next_state_batch = torch.stack(batch.next_state)
        q_sa = self.prediction_net(state_batch).gather(1, action_batch).squeeze(1)

        final_states_locations = next_state_batch.max(dim=1)[0].eq(0).type(torch.bool)
        non_final_states_locations = (final_states_locations == False)
        non_final_states = next_state_batch[non_final_states_locations]
        q_ns = torch.zeros(hp.BATCH_SIZE).to(hp.DEVICE)
        q_ns[non_final_states_locations] = self.target_net(non_final_states).max(1)[0].detach()


        expected_qsa = reward_batch + hp.GAMMA * q_ns

        loss = F.mse_loss(q_sa, expected_qsa)
        print(round(loss.item(), 2))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





