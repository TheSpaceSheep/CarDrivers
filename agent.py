import torch
import torch.nn as nn
import torch.optim as opt
import hyperparameters as hp
from functions import *
from classes import is_out, Point


class Driver(nn.Module):
    def __init__(self, car):
        super().__init__()
        self.car = car
        self.next_checkpoint = None
        self.track = None
        self.info_tens = torch.tensor([], device=hp.DEVICE, dtype=torch.float32)
        self.learning = False

        self.lin1 = nn.Linear(38, 30)
        self.relu1 = torch.relu
        self.classifier = nn.Linear(30, 4)
        self.probabilities = torch.sigmoid

        self.cuda(hp.DEVICE)

        self.optimizer = opt.SGD(self.parameters(), lr=0.0005)

    def initiate(self, track, car=None, next_check=False):
        if car is not None:
            self.car = car

        if track is not None:
            self.track = track

        if next_check:
            self.next_checkpoint = track.current_checkpoint[self.car.name]

        info = []

        self.info_tens = torch.tensor(info, dtype=torch.float32, device=hp.DEVICE)

    def next_checkpoint(self, checkpoint):
        self.next_checkpoint = checkpoint

    def forward(self, car=None, track=None, checkpoint=None):
        if car is None:
            car = self.car
        if track is None:
            track = self.track
        if checkpoint is None:
            if self.next_checkpoint is None:
                return torch.tensor([0., 0., 0., 0.], device=hp.DEVICE)
            else:
                checkpoint = self.next_checkpoint

        sensors = []
        for y in range(-60, 0, 30):
            top_left = 0
            for x in range(-40, 0, 20):
                X, Y = rotate(car.x + x, car.y + y, car.heading, car.x, car.y)
                top_left += is_out(track, x=X, y=Y)
                sensors.append(int(is_out(track, x=X, y=Y)))

            top_right = 0
            for x in range(0, 250, 50):
                X, Y = rotate(car.x + x, car.y + y, car.heading, car.x, car.y)
                top_right += is_out(track, x=X, y=Y)
                sensors.append(int(is_out(track, x=X, y=Y)))

        for y in range(0, 90, 30):
            bottom_left = 0
            for x in range(-40, 0, 20):
                X, Y = rotate(car.x + x, car.y + y, car.heading, car.x, car.y)
                bottom_left += is_out(track, x=X, y=Y)
                sensors.append(int(is_out(track, x=X, y=Y)))

            bottom_right = 0
            for x in range(0, 250, 50):
                X, Y = rotate(car.x + x, car.y + y, car.heading, car.x, car.y)
                bottom_right += is_out(track, x=X, y=Y)
                sensors.append(int(is_out(track, x=X, y=Y)))

        top_left /= 25
        top_right /= 25
        bottom_left /= 25
        bottom_right /= 25

        t = torch.tensor([car.speed/80,
                          car.heading/360,
                          -car.heading/360,
                          *sensors],
                         dtype=torch.float32,
                         device=hp.DEVICE)
        inp = torch.cat([t, self.info_tens])
        # inp = torch.cat([inp, torch.tensor([0.] * 6, device=hp.DEVICE)])
        hidden1 = self.lin1(inp)
        hidden1 = self.relu1(hidden1)
        out = self.classifier(hidden1)
        probas = self.probabilities(out)

        return probas

    def decision(self):
        with torch.no_grad():
            probas = self.forward()

            decision = []
            if probas[0] > 0.5:
                decision.append(self.car.up)
            if probas[2] > 0.5:
                decision.append(self.car.down)
            if probas[1] > 0.5:
                decision.append(self.car.left)
            if probas[3] > 0.5:
                decision.append(self.car.right)

            return decision

    def watch_and_learn(self, car, checkpoint, history):
        actual_decision = [h for h in history if h in [car.up, car.left, car.down, car.right]]
        actual_probas = torch.tensor([float(car.up in actual_decision), float(car.left in actual_decision),
                                      float(car.down in actual_decision), float(car.right in actual_decision)],
                                     device=hp.DEVICE,
                                     requires_grad=True)

        self.optimizer.zero_grad()
        probas = self.forward(car=car, checkpoint=checkpoint)

        loss = (actual_probas - probas) ** 2
        loss = loss.sum()
        print(loss.item(), probas)

        loss.backward()
        self.optimizer.step()
