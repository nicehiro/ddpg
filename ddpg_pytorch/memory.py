from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'state_next'))


class ReplayBuffer():
    def __init__(self, capacity: int, batch_size):
        self.capacity = capacity
        self.position = 0
        self.batch_size = batch_size
        self.memory = []

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def append(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.position = self.position % self.capacity
        trans = Transition(*args)
        self.memory[self.position] = trans
        self.position += 1

    def can_sample(self):
        return self.position > 10 * self.batch_size
