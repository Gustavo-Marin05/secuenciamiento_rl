import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SecuenciamientoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_pedidos = 5
        self.max_steps = 5
        self.current_step = 0
        self.pedidos = np.random.randint(1, 11, size=(self.num_pedidos,))
        self.action_space = spaces.Discrete(self.num_pedidos)
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.num_pedidos,), dtype=np.int32)

    def reset(self, **kwargs):
        self.pedidos = np.random.randint(1, 11, size=(self.num_pedidos,))
        self.current_step = 0
        return self.pedidos.copy(), {}

    def step(self, action):
        if self.pedidos[action] == 0:
            reward = -5
            done = False
            return self.pedidos.copy(), reward, done, False, {}

        tiempo = self.pedidos[action]
        reward = -tiempo
        self.pedidos[action] = 0

        done = np.all(self.pedidos == 0)
        return self.pedidos.copy(), reward, done, False, {}

    def render(self):
        print("Pedidos actuales (tiempos):", self.pedidos)
