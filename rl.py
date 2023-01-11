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


possible_transf_script = "wsl-experiments/bin/possible-transformations.wsl"
transform_program_script = "wsl-experiments/bin/transform-program.wsl"
metric_script = "wsl-experiments/bin/chosen-metric.wsl"

PATIENCE = 25
TIMESTEPS = 50000
EPISODE_LENGTH = 500

device = "cpu"  # "cuda" or "cpu"

transform = T.Compose([T.NormalizeFeatures(), T.ToDevice(device),])

print("Loading GAE...")
gae = torch.load("gae.pt").to(device)
print("GAE loaded")


class WSLRLEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is the env for transforming WSL programs. 
    """

    metadata = {"render.modes": ["console"]}

    def __init__(self, program_name, emb_size=128, ep_length=500):
        super().__init__()

        # all known transformations
        transformations = dict()
        transformations_rev = dict()
        with open("transformations.txt", "r") as ft:
            for i, line in enumerate(ft):
                transformations[line.strip()] = i
                transformations_rev[i] = line.strip()

        n_actions = len(transformations)

        self.program_name = program_name

        self.model = gae
        self.transformations = transformations
        self.transformations_rev = transformations_rev
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(emb_size,), dtype=np.float32
        )
        self.current_step = 0
        self.reward_history = []
        self.current_program_name = program_name
        self.current_program_size = -1
        self.current_obs = None
        self.best_metric_value = np.inf
        self.best_metric_value_step = np.inf
        self.best_metric_value_program = None
        self.best_metric_value_path = None

        # constants for the initial program
        self.initial_program_size = -1
        self.initial_metric_value = -1

        self.ep_length = ep_length
        self.total_steps = 0

    def get_embedding(self):
        print("Converting to csv:", self.current_program_name)
        filename_adj = generate.convert_wsl_to_csv(self.current_program_name)
        print("Converting to nx graph:", filename_adj)
        nx_graph = generate_nx.convert_csv_to_nx(filename_adj)
        pt_graph = from_networkx(nx_graph)
        pt_graph.x = (
            torch.stack((pt_graph.generic_type, pt_graph.specific_type), dim=1)
            .float()
            .to(device)
        )
        pt_graph = transform(pt_graph)
        self.current_obs = self.model.encode(pt_graph.x, pt_graph.edge_index)
        self.current_program_size = len(self.current_obs)
        if self.initial_program_size == -1:
            self.initial_program_size = self.current_program_size
            print("Initial program size:", self.initial_program_size)
        self.current_obs = self.current_obs.mean(axis=0).cpu().detach().numpy()

    def reset(self):
        self.current_program_name = self.program_name
        self.current_step = 0
        self.reward_history = []
        self.get_embedding()
        return self.current_obs

    def get_possible_transformations(self):
        # search for all possible transformations for current program
        possible_transformations = []

        poss_transf_csv = f"{self.current_program_name}_pos_transfs.csv"
        subprocess.call(
            generate.base_command
            + [possible_transf_script, self.current_program_name, poss_transf_csv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(poss_transf_csv, "r") as f:
            for line in f:
                if line:
                    data = line.strip().split(",")
                    # check if any transformation is possible
                    if data and len(data) == 2:
                        transf_name, pos = data
                        possible_transformations.append((transf_name, pos))
        return possible_transformations

    def action_masks(self) -> List[bool]:
        possible_transformations = self.get_possible_transformations()

        transformations_for_this_file = set([p[0] for p in possible_transformations])
        a_mask = []
        for _, transf in self.transformations_rev.items():
            a_mask.append(transf in transformations_for_this_file)

        return a_mask

    def step(self, action):
        # execution info:
        info = {}
        # decode transformation from int index to name
        transformation = self.transformations_rev[action]

        # Perform transformation:
        possible_transformations = self.get_possible_transformations()

        # find all transformations of this type
        positions = [p for p in possible_transformations if p[0] == transformation]
        # take random one
        chosen_transf, chosen_pos = random.choice(positions)
        in_file = self.current_program_name
        out_file = self.program_name.replace(".wsl", f"_{time.time_ns()}.wsl")

        self.total_steps += 1

        print("Total steps:", self.total_steps)
        print("Current program size:", self.current_program_size)
        print("Reward history:", self.reward_history)
        print("Current step in episode:", self.current_step)
        print(
            "Applying transformation:",
            generate.base_command
            + [transform_program_script, in_file, chosen_transf, chosen_pos, out_file],
        )

        subprocess.call(
            generate.base_command
            + [transform_program_script, in_file, chosen_transf, chosen_pos, out_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # new program becomes current program
        self.current_program_name = out_file
        self.get_embedding()

        # Calculate reward
        # Reward is the difference in metric between current program and its predecessor

        metric_in = subprocess.check_output(
            generate.base_command + [metric_script, in_file], stderr=subprocess.DEVNULL
        )
        metric_out = subprocess.check_output(
            generate.base_command + [metric_script, out_file], stderr=subprocess.DEVNULL
        )

        metric_in = int(metric_in.decode().split("\n")[3])
        print(metric_script, in_file, out_file, metric_out)
        metric_out = int(metric_out.decode().split("\n")[3])

        print("Current metric value:", metric_out)
        if self.initial_metric_value == -1:
            self.initial_metric_value = metric_in
            print("Initial metric value:", metric_in)

        if metric_out < self.best_metric_value:
            self.best_metric_value = metric_out
            self.best_metric_value_step = self.total_steps
            self.best_metric_value_path = out_file
            self.best_metric_value_program = open(out_file, "r").read()
        print(
            f"Best metric value so far (step: {self.best_metric_value_step}):",
            self.best_metric_value,
        )

        reward = metric_in - metric_out
        self.reward_history.append(reward)

        self.current_step += 1

        # basic condition based on episode length
        done = self.current_step >= self.ep_length

        # patience based exits, if no improvement last PATIENCE steps => abort
        if (
            len(self.reward_history) >= PATIENCE
            and sum(self.reward_history[-PATIENCE:]) <= 0
        ):
            done = True

        # program size grows too much
        if self.current_program_size > 3 * self.initial_program_size:
            done = True

        # metric value grows too much
        if metric_out > 3 * self.initial_metric_value:
            done = True

        if self.total_steps > TIMESTEPS:
            done = True

        return self.current_obs, reward, done, info

    def render(self, mode="console"):
        print("TODO: rendering")

    def close(self):
        pass


if __name__ == "__main__":
    # running for all programs
    local_dataset = train_set + test_set

    if "SLURM_PROCID" in os.environ:
        # running on slurm!
        procid = int(os.environ["SLURM_PROCID"])
        n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        print(f"Node {procid}, nodes: {n_nodes}")
        new_dataset = []
        while procid < len(local_dataset):
            new_dataset.append(local_dataset[procid])
            procid += n_nodes

        local_dataset = new_dataset

    global_model = None
    model = None

    if "--single" in sys.argv:
        print("--single is present")
        local_dataset = train_set

    print("working with files:", local_dataset)

    for program_name in local_dataset:
        print("Building model for:", program_name)
        env = WSLRLEnv(
            program_name="workdir/" + program_name,
            ep_length=EPISODE_LENGTH,
            emb_size=128,
        )

        if "--single" not in sys.argv:
            model = MaskablePPO(
                "MlpPolicy", env, gamma=0.4, seed=32, verbose=1
            )  # for tests add: n_steps=10, n_epochs=1
        else:
            if not global_model:
                global_model = MaskablePPO(
                    "MlpPolicy", env, gamma=0.4, seed=32, verbose=1
                )  # for tests add: n_steps=10, n_epochs=1
            model = global_model
            model.set_env(env)  # Switching model to new env
            program_name += "_global"

        try:
            model.learn(total_timesteps=TIMESTEPS, log_interval=1)
        except Exception as e:
            print(e)
            traceback.print_exc()
            model.save(f"models/rl_model_exc_{program_name}.model")
            pickle.dump(env, open(f"models/env_exc_{program_name}.pkl", "wb"))

        model.save(f"models/rl_model_{program_name}.model")
        pickle.dump(env, open(f"models/env_{program_name}.pkl", "wb"))

        if "--eval" in sys.argv:
            evaluate_policy(
                model, env, n_eval_episodes=20, reward_threshold=90, warn=False
            )

        if "--single" not in sys.argv:
            obs = env.reset()
            while True:
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=action_masks)
                obs, reward, done, info = env.step(action)
                print(done, reward, env.transformations_rev[action])
                if done:
                    break
