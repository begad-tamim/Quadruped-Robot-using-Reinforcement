#!/usr/bin/env python

import os
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from multiprocessing import Pipe
from doggy.util.gui import GUI
from doggy.OpenLoopd.DoggyOL import BezierStepper
from doggy.GaitGenerator.Bezier import BezierGait
from doggy.Kinematics.DoggyKinematics import DoggyModel
from doggy.GymEnvs.doggy_bezier_env import DoggyBezierEnv
from doggy.doggy_env_randomizer import DoggyEnvRandomizer
from .ars_lib.ars import ARSAgent, Normalizer, Policy, ParallelWorker

sys.path.append('../../')

# ARGUMENTS
descr = "Doggy ARS Agent Trainer."
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("-hf",
                    "--HeightField",
                    help="Use HeightField",
                    action='store_true')
parser.add_argument("-nc",
                    "--NoContactSensing",
                    help="Disable Contact Sensing",
                    action='store_true')
parser.add_argument("-dr",
                    "--DontRandomize",
                    help="Do NOT Randomize State and Environment.",
                    action='store_true')
parser.add_argument("-s", "--Seed", help="Seed (Default: 0).")
ARGS = parser.parse_args()

# Messages for Pipe
_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def main():
    """ The main() function. """
    # Hold mp pipes
    mp.freeze_support()

    print("STARTING Doggy TRAINING ENV")
    seed = 0
    if ARGS.Seed:
        seed = int(ARGS.Seed)
    print("SEED: {}".format(seed))
    max_timesteps = 4e6
    eval_freq = 1e1
    save_model = True
    file_name = "doggy_ars_"

    if ARGS.HeightField:
        height_field = True
    else:
        height_field = False

    if ARGS.NoContactSensing:
        contacts = False
    else:
        contacts = True

    if ARGS.DontRandomize:
        env_randomizer = None
        rand_name = "norand_"
    else:
        env_randomizer = DoggyEnvRandomizer()
        rand_name = "rand_"

    # Find abs path to this file
    my_path = os.path.abspath(os.path.dirname(__file__))
    results_path = os.path.join(my_path, "./results")
    if contacts:
        models_path = os.path.join(my_path, "./models/contact")
    else:
        models_path = os.path.join(my_path, "./models/no_contact")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    env = DoggyBezierEnv(render=False,
                        on_rack=False,
                        height_field=height_field,
                        draw_foot_path=False,
                        contacts=contacts,
                        env_randomizer=env_randomizer)

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    print("STATE DIM: {}".format(state_dim))
    action_dim = env.action_space.shape[0]
    print("ACTION DIM: {}".format(action_dim))
    max_action = float(env.action_space.high[0])

    env.reset()

    g_u_i = GUI(env.doggy.quadruped)

    doggy = DoggyModel()
    T_bf = doggy.WorldToFoot

    bz_step = BezierStepper(dt=env._time_step)
    bzg = BezierGait(dt=env._time_step)

    # Initialize Normalizer
    normalizer = Normalizer(state_dim)

    # Initialize Policy
    policy = Policy(state_dim, action_dim, seed=seed)

    # Initialize Agent with normalizer, policy and gym env
    agent = ARSAgent(normalizer, policy, env, bz_step, bzg, doggy)
    agent_num = 0
    if os.path.exists(models_path + "/" + file_name + str(agent_num) +
                      "_policy"):
        print("Loading Existing agent")
        agent.load(models_path + "/" + file_name + str(agent_num))

    env.reset(agent.desired_velocity, agent.desired_rate)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Create mp pipes
    num_processes = policy.num_deltas
    processes = []
    childPipes = []
    parentPipes = []

    # Store mp pipes
    for pr in range(num_processes):
        parentPipe, childPipe = Pipe()
        parentPipes.append(parentPipe)
        childPipes.append(childPipe)

    # Start multiprocessing
    # Start multiprocessing
    for proc_num in range(num_processes):
        p = mp.Process(target=ParallelWorker,
                       args=(childPipes[proc_num], env, state_dim))
        p.start()
        processes.append(p)

    print("STARTED Doggy TRAINING ENV")
    t = 0
    while t < (int(max_timesteps)):

        # Maximum timesteps per rollout

        episode_reward, episode_timesteps = agent.train_parallel(parentPipes)
        t += episode_timesteps
        # episode_reward = agent.train()
        # +1 to account for 0 indexing.
        # +0 on ep_timesteps since it will increment +1 even if done=True
        print(
            "Total T: {} Episode Num: {} Episode T: {} Reward: {:.2f} REWARD PER STEP: {:.2f}"
            .format(t + 1, episode_num, episode_timesteps, episode_reward,
                    episode_reward / float(episode_timesteps)))

        # Store Results (concat)
        if episode_num == 0:
            res = np.array(
                [[episode_reward, episode_reward / float(episode_timesteps)]])
        else:
            new_res = np.array(
                [[episode_reward, episode_reward / float(episode_timesteps)]])
            res = np.concatenate((res, new_res))

        # Also Save Results So Far (Overwrite)
        # Results contain 2D numpy array of total reward for each ep
        # and reward per timestep for each ep
        np.save(
            results_path + "/" + str(file_name) + rand_name + "seed" +
            str(seed), res)

        # Evaluate episode
        if (episode_num + 1) % eval_freq == 0:
            if save_model:
                agent.save(models_path + "/" + str(file_name) +
                           str(episode_num))

        episode_num += 1

    # Close pipes and hence envs
    for parentPipe in parentPipes:
        parentPipe.send([_CLOSE, "pay2"])

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()