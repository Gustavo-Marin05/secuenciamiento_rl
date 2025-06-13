from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import SecuenciamientoEnv

def train():
    # Crear entorno vectorizado y normalizado
    env = DummyVecEnv([lambda: SecuenciamientoEnv()])
    env = VecNormalize(env, norm_obs=True)

    # Crear modelo con logging
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")

    # Entrenar el modelo
    model.learn(total_timesteps=100000)

    # Guardar el modelo y entorno
    model.save("modelo_secuenciamiento")
    env.save("env_normalizado.pkl")

if __name__ == "__main__":
    train()
