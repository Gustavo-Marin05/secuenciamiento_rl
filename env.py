import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SecuenciamientoEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_pedidos = 5
        self.max_steps = 20
        self.current_step = 0
        self.last_action = None

        self.invalid_counter = 0
        self.max_invalid_steps = 5

        self.pedidos = np.random.randint(1, 11, size=(self.num_pedidos,))
        self.costos_cambio = np.random.randint(1, 5, size=(self.num_pedidos, self.num_pedidos))
        np.fill_diagonal(self.costos_cambio, 0)

        self.action_space = spaces.Discrete(self.num_pedidos)
        self.observation_space = spaces.Dict({
            "pedidos": spaces.Box(low=0, high=10, shape=(self.num_pedidos,), dtype=np.int32),
            "ultimo_pedido": spaces.Discrete(self.num_pedidos + 1)
        })

    def reset(self, **kwargs):
        self.pedidos = np.random.randint(1, 11, size=(self.num_pedidos,))
        self.costos_cambio = np.random.randint(1, 5, size=(self.num_pedidos, self.num_pedidos))
        np.fill_diagonal(self.costos_cambio, 0)

        self.current_step = 0
        self.last_action = None
        self.invalid_counter = 0

        return {"pedidos": self.pedidos.copy(), "ultimo_pedido": self.num_pedidos}, {}

    def step(self, action):
        if action not in range(self.num_pedidos):
            raise ValueError(f"Acción inválida: {action}")

        if self.pedidos[action] == 0:
            self.invalid_counter += 1
            reward = -10
            self.current_step += 1
            done = self.invalid_counter >= self.max_invalid_steps
            return {
                "pedidos": self.pedidos.copy(),
                "ultimo_pedido": self.last_action if self.last_action is not None else self.num_pedidos
            }, reward, done, False, {}

        self.invalid_counter = 0

        tiempo = self.pedidos[action]
        costo_cambio = 0
        if self.last_action is not None and action != self.last_action:
            costo_cambio = self.costos_cambio[self.last_action][action]

        reward = -(tiempo + costo_cambio)
        self.pedidos[action] = 0
        self.last_action = action
        self.current_step += 1

        done = np.all(self.pedidos == 0) or (self.current_step >= self.max_steps)

        return {
            "pedidos": self.pedidos.copy(),
            "ultimo_pedido": action
        }, reward, done, False, {}

    def render(self):
        pass  # Sin salida durante el entrenamiento
