from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import SecuenciamientoEnv
import os

def train():
    print("\n=== INICIANDO ENTRENAMIENTO ===")
    
    os.makedirs("./ppo_logs", exist_ok=True)
    
    env = DummyVecEnv([lambda: SecuenciamientoEnv()])
    print("Entorno creado correctamente")
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        learning_rate=0.001,
        ent_coef=0.1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="auto"
    )
    print("Modelo PPO configurado")

    print("\nComenzando entrenamiento...")
    model.learn(total_timesteps=20000)
    
    model.save("modelo_secuenciamiento_mejorado")
    print("\nEntrenamiento completado. Modelo guardado")

if __name__ == "__main__":
    train()
    print("\n=== ENTRENAMIENTO FINALIZADO ===")
