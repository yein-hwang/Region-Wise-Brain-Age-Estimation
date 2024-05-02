import math
import matplotlib.pyplot as plt
import numpy as np

class CustomCosineAnnealingWarmUpRestarts:
    def __init__(self, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = -1
    
    def get_lr(self):
        if self.T_cur == -1:
            return [0.0]  # Warm-up phase
        elif self.T_cur < self.T_up:
            return [(self.eta_max - 0.0) * self.T_cur / self.T_up + 0.0]
        else:
            return [0.0 + (self.eta_max - 0.0) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2]
    
    def step(self, epoch=None):
        if epoch is None:
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)

# Define the number of steps
total_steps = 602

# Define the scheduler
scheduler = CustomCosineAnnealingWarmUpRestarts(T_0=100, T_up=10, T_mult=2, eta_max=1e-3, gamma=0.5)

# Lists to store learning rate and step number
lr_values = []
step_numbers = []

# Simulate learning rate updates for each step
for step in range(total_steps):
    # Get the current learning rate for this step
    current_lr = scheduler.get_lr()[0]
    
    # Append the learning rate and step number to the lists
    lr_values.append(current_lr)
    step_numbers.append(step)
    
    # Perform scheduler step (updating internal state for the next step)
    scheduler.step()

# Convert lists to NumPy arrays
lr_values = np.array(lr_values)
step_numbers = np.array(step_numbers)

# Plot the learning rate over steps
plt.figure(figsize=(10, 6))
plt.plot(step_numbers, lr_values, marker='o', linestyle='-', color='b')
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)
plt.show()
