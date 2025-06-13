from stable_baselines3 import PPO
from env import SecuenciamientoEnv
import matplotlib.pyplot as plt
import os

def evaluate_agent(model_path, episodes=5):
    print(f"\n=== EVALUANDO MODELO: {model_path} ===")
    
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}.zip")
    
    env = SecuenciamientoEnv()
    model = PPO.load(model_path)
    print(f"Modelo cargado: {model_path}")

    rewards = []
    all_sequences = []

    for ep in range(episodes):
        print(f"\n=== Episodio {ep+1} ===")
        obs, info = env.reset()
        done = False
        total_reward = 0
        secuencia = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            secuencia.append(action)
            env.render()

        rewards.append(total_reward)
        all_sequences.append(secuencia)
        print(f"\nResultado Episodio {ep+1}:")
        print(f"Secuencia: {secuencia}")
        print(f"Recompensa Total: {total_reward:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, 'o-')
    plt.title('Recompensas por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.grid(True)
    plt.savefig("resultados_evaluacion.png")
    plt.show()

if __name__ == "__main__":
    try:
        evaluate_agent("modelo_secuenciamiento_mejorado")
    except Exception as e:
        print(f"\nERROR: {e}")
    print("\n=== EVALUACIÓN COMPLETADA ===")
