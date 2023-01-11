import numpy as np
import gym
import pickle
import torch

import generate
import generate_nx
import subprocess
import os
import sys
import traceback
import time
import random
from pathlib import Path
from os.path import join
from gym import spaces
from torch_geometric.utils.convert import from_networkx
from stable_baselines3.common.env_checker import check_env

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from typing import List, Optional
from dataset import train_set, test_set

import torch_geometric.transforms as T
from gae import GCNEncoder

from rl import WSLRLEnv, EPISODE_LENGTH

if __name__ == "__main__":
    local_dataset = train_set + test_set

    print("env results")
    for program_name in local_dataset:
        if "--skip-global" in sys.argv:
            break

        # for every program export best metric and best metric value step and initial metric value
        env = pickle.load(open(f"models/env_{program_name}.pkl", "rb"))
        print(
            program_name,
            env.best_metric_value,
            env.best_metric_value_step,
            env.initial_metric_value,
        )
        if "--print-programs" in sys.argv:
            print("*" * 5)
            print(env.best_metric_value_path)
            print(env.best_metric_value_program)
            print("*" * 5)

    print("*" * 20)

    print("global model results")
    # global model results on test dataset
    model = MaskablePPO.load(
        "models/rl_model_linkedlist.wsl_global.model"
    )  # last model
    for test_program in local_dataset:
        if "--skip-global" in sys.argv:
            break
        print("testing global model on program", test_program)

        env = WSLRLEnv(
            program_name="workdir/" + test_program,
            ep_length=EPISODE_LENGTH,
            emb_size=128,
        )

        obs = env.reset()
        while True:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            print(done, reward, env.transformations_rev[action])
            if done:
                print(
                    "testing completed: global model on program",
                    test_program,
                    "metric value",
                    env.best_metric_value,
                )
                break

    print("*" * 20)

    print("single trained results")
    # matrix of results for all the models on all the programs
    for program_name in local_dataset:

        model = MaskablePPO.load(f"models/rl_model_{program_name}.model")

        for test_program in local_dataset:
            print("testing model trained for", program_name, "on program", test_program)

            env = WSLRLEnv(
                program_name="workdir/" + test_program,
                ep_length=EPISODE_LENGTH,
                emb_size=128,
            )

            obs = env.reset()
            while True:
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=action_masks)
                obs, reward, done, info = env.step(action)
                print(done, reward, env.transformations_rev[action])
                if done:
                    print(
                        "testing completed: model trained for",
                        program_name,
                        "on program",
                        test_program,
                        "metric value",
                        env.best_metric_value,
                    )
                    break
