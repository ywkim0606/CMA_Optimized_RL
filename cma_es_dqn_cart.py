import os

import cma
import gym
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

#env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

es = cma.CMAEvolutionStrategy(
        [0.99,0.0005,50000,0.1,0.02,500,0.6,0.4,1e-06,32,32],
        0.1, {'popsize': 20})

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
            reward_sum -= reward
            episode_rewards.append(reward_sum)
    result = np.mean(episode_rewards)
    print(result)
    return result


def objective(param):
    """
    gamma, learning_rate, buffer_size, exploration_fraction, exploration_final_eps,
    target_network_update_freq, prioritized_replay_alpha, prioritized_replay_beta0,
    prioritized_replay_eps, total_timesteps
    """
    # let all values be positive
    param = list(map(abs, param))
    param[2] = int(param[2]+abs(param[2]-50000)*100)
    param[5] = int(param[5]+abs(param[5]-500)*100)
    param[9] = int(param[9]+abs(param[9]-32)*100)
    param[10] = int(param[10]+abs(param[10]-32)*100)

    policy_kwargs = dict(act_fun=tf.nn.relu, layers=[param[9], param[10]])
    model = DQN(MlpPolicy, env, gamma=param[0], learning_rate=param[1],
        buffer_size=int(param[2]), exploration_fraction=param[3],
        exploration_final_eps=param[4],
        target_network_update_freq=int(param[5]),
        prioritized_replay=True,
        prioritized_replay_alpha=param[6],
        prioritized_replay_beta0=param[7],
        prioritized_replay_eps=param[8], policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=5000)
    total_reward = evaluate(model)
    if total_reward <= -es.result[1]:
        model.save('./dqn_model/cma_optimized_cart_dqn.pkl')
    return total_reward

def main():
    
    d = {}
    j = 0
    history = []
    while True:
        # ask for a list of new solutions
        param_list = es.ask(es.popsize)
        print(param_list)

        # create an array to hold the rewards
        rewards = []

        # calculate the reward for each solution using the objective function
        for i in range(es.popsize):
            print(str(i)+" pop: ", param_list[i])
            reward = objective(param_list[i])
            rewards.append(reward)
        
        # give the rewards back to es
        rewards = [float(i) for i in rewards]

        d['gen'+str(j)] = rewards

        es.tell(param_list, rewards)

        history.append(min(rewards))
        print(es.result)
        print(history)
        print("reward at iteration", (j+1), es.result[1])
        if es.result[1] <= -500 or j==10:
            break
        j+=1
    
    print("finished on ",str(j+1)," iteration")
    print("local optimum discovered by solver:\n", es.result[0])
    print("fitness score at this local optimum:", es.result[1])
    print(history)
    df = pd.DataFrame.from_dict(d)
    df.to_csv('cme_optimized_cart_dqn_data.csv')

if __name__ == '__main__':
  main()