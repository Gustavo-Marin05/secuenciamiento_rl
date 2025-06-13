import gymnasium as gym
from env import SecuenciamientoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle

def make_env():
    return SecuenciamientoEnv()

def main():
    env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    rewards_history = []
    num_iterations = 5  # Puedes cambiarlo

    for i in range(num_iterations):
        model.learn(total_timesteps=2048, reset_num_timesteps=False)

        # Evaluar recompensa media en 5 episodios
        mean_reward = 0
        n_eval_episodes = 5
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            mean_reward += episode_reward
        mean_reward /= n_eval_episodes

        print(f"Iteración {i+1}: recompensa media eval = {mean_reward}")
        rewards_history.append(mean_reward)

    model.save("modelo_entrenado")

    with open("rewards_history.pkl", "wb") as f:
        pickle.dump(rewards_history, f)

if __name__ == "__main__":
    main()
