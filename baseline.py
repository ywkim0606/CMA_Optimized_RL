import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import tensorflow as tf

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])
param=[0.99,0.0005,50000,0.1,0.02,500,0.6,0.4,1e-06]

def evaluate(model):
    """
    Return mean fitness (sum of episodic rewards) for given model
    """
    episode_rewards = []
    for _ in range(100):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            episode_rewards.append(reward_sum)
    return np.mean(episode_rewards)

def main():
    """
    use tensorboard
    tensorboard --logdir ./dqn_cartpole_tensorboard/
    http://localhost:6006/
    """
    model = DQN(MlpPolicy, env, gamma=param[0], learning_rate=param[1],
        buffer_size=param[2], exploration_fraction=param[3],
        exploration_final_eps=param[4],
        target_network_update_freq=param[5],
        prioritized_replay=True,
        prioritized_replay_alpha=param[6],
        prioritized_replay_beta0=param[7],
        prioritized_replay_eps=param[8], verbose=1,
        tensorboard_log="./dqn_cartpole_tensorboard/")

    model.learn(5000,tb_log_name="cart_baseline", log_interval=1)
    model.save('./dqn_model/dqn_cart_baseline.pkl')
    total_reward = evaluate(model)
    print(total_reward)

if __name__ == "__main__":
    main()