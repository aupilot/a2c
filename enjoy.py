import argparse
import os

import gym
import numpy as np
import torch
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs, AddTimestep
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


# workaround to unpickle olf model files
import sys

from robots.munitaur_kir import MinitaurKirEnv

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
# parser.add_argument('--env-name', default='MinitaurBulletEnv-v0', #'HalfCheetahBulletEnv-v0',
#                     help='environment to train on (default: MinitaurBulletEnv-v0)')
parser.add_argument('--load-dir', default='./trained_models/ppo',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det


# minitaur must be initialised this way to render!
# env = MinitaurKirEnv(render=True)
# args.env_name = 'MinitaurBulletEnv-v0'

# args.env_name = 'AntBulletEnv-v0'
args.env_name = 'HalfCheetahBulletEnv-v0'
env = gym.make(args.env_name)
env.render(mode="human")
env = AddTimestep(env)


# args.env_name = 'CartPole-v1'
# env = gym.make(args.env_name)
# env.render(mode="human")


# Get a render function
# render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
actor_critic.eval()

# vec_norm = get_vec_normalize(env)
# if vec_norm is not None:
#     vec_norm.eval()
#     vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

obs_prev = obs
while True:
    with torch.no_grad():
        input = torch.FloatTensor([np.hstack((obs, obs_prev))])
        # input = torch.cat((obs, obs_prev), 1)
        value, action, _, recurrent_hidden_states = actor_critic.act(input, recurrent_hidden_states, masks, deterministic=args.det)
            # torch.cat((obs, obs_prev), 1), recurrent_hidden_states, masks, deterministic=args.det)

    obs_prev = obs

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action.squeeze_().numpy())
    obs = torch.FloatTensor(obs)

    masks.fill_(0.0 if done else 1.0)

    # if args.env_name.find('Bullet') > -1:
    #     if torsoId > -1:
    #         distance = 5
    #         yaw = 0
    #         humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
    #         p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    # if render_func is not None:
        # render_func('human')
    env.render('human')

