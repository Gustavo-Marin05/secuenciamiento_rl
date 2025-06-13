import gymnasium as gym
from stable_baselines3 import PPO
from env import SecuenciamientoEnv

def evaluate_agent(model_path, num_episodes=10):
    env = SecuenciamientoEnv()
    model = PPO.load(model_path)

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        actions_taken = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            actions_taken.append(int(action))

        print(f"Ejemplo {episode}: Recompensa total = {total_reward:.2f}")
        print("Secuencia de pedidos seleccionados:", actions_taken)
        env.render()
        print("-" * 40)

if __name__ == "__main__":
    modelo_path = "modelo_entrenado.zip"
    evaluate_agent(modelo_path)
