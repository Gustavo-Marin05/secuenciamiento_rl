from stable_baselines3 import PPO
from env import SecuenciamientoEnv

def train():
    env = SecuenciamientoEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("modelo_secuenciamiento")

if __name__ == "__main__":
    train()
