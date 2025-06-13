import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import SecuenciamientoEnv

def evaluate_agent(model_path, episodes=20):
    env = SecuenciamientoEnv()
    model = PPO.load(model_path)

    rewards = []
    all_sequences = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        secuencia = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            secuencia.append(action)

        rewards.append(total_reward)
        all_sequences.append(secuencia)
        print(f"Ejemplo {ep+1}: Recompensa total = {total_reward:.2f}")
        print(f"Secuencia de pedidos seleccionados: {secuencia}\n")

    # Graficar recompensas
    plt.plot(range(1, episodes+1), rewards, marker='o')
    plt.title("Recompensa total por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.show()

    # Guardar secuencias en un archivo de texto
    with open("secuencias_agente.txt", "w") as f:
        for i, seq in enumerate(all_sequences):
            f.write(f"Episodio {i+1}: {seq}\n")

    print("Las secuencias de acciones se guardaron en secuencias_agente.txt")

if __name__ == "__main__":
    evaluate_agent("modelo_secuenciamiento")  # Cambia si usas otro nombre
