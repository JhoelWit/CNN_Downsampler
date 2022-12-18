import os
import sys
from datetime import datetime
import yaml
from robot_env import RobotEnv
from eval_env import EvalEnv
# from bayes_eval_env import BayesSwarm

# from bayes_gym_env import BayesSwarm
from bayes_cnn_policy import CNNPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback, CallbackList
import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback



# Testing case studies and the general training environment.
env_check = False
if env_check:
    # env = RobotEnv()
    env = EvalEnv()
    check_env(env)

    env.reset() 
    step = 0
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        step += 1
        if done:
            print(f"termination on step {step}")
            env.reset()

def make_env(rank, seed=0):
    def _init():
        env = Monitor(RobotEnv())
        env.seed(seed + rank)
        return env
    
    set_random_seed(seed)
    return _init

# env = Monitor(RobotEnv())


# run.finish()

if __name__ == "__main__":

    with open('policy_kwargs.yaml', 'r') as file:
            policy_kwargs = yaml.safe_load(file)


    run = wandb.init(project="bayes_swarm_T-RO", 
                entity="jhoelwit",
                sync_tensorboard=True,
                save_code=True,
                config=policy_kwargs,
                )

    env = SubprocVecEnv([make_env(i) for i in range(40)]) #Sunya has 40 cores
    # env = Monitor(RobotEnv())
    # eval_env = Monitor(BayesSwarm(case="CNN"))
    current_date_time = datetime.now()
    timestamp = current_date_time.strftime("%H_%M_%S")
    #Loading a model to continue training
    # model = PPO.load("ATC_Model/ATC_GRL_Model_9_11_22_690000_steps", env=env, verbose=1, n_steps=20_000,batch_size=10_000,gamma=1,learning_rate=0.00001, tensorboard_log='ATC_GRL_Model/', device="cuda")
    # model.learn(total_timesteps=2_000_000, callback=custom_callback, reset_num_timesteps=False)
    checkpoint_callback = CheckpointCallback(save_freq=1_000, save_path=f"models/{run.id}", name_prefix="bayes_agent"+timestamp)
    # eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=5, best_model_save_path=f"models/{run.id}", eval_freq=1_000)
    wand_callback=WandbCallback(
                    gradient_save_freq=100,
                    verbose=2,
                    log="all")
    callbacks = CallbackList([checkpoint_callback, wand_callback])

    model = PPO(CNNPolicy,
                env=env,
                tensorboard_log=f"runs/{run.id}",
                verbose=1,
                n_steps=500, 
                batch_size=5_000, #1_000
                gamma=0.90,
                learning_rate=1e-6,
                ent_coef=0.01,
                device='cuda',
                policy_kwargs=policy_kwargs
                )

    model.learn(total_timesteps=10_000_000, 
                callback=callbacks,
                )
    model.save("Final_CNN_model_"+timestamp)