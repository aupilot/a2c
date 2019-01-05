import copy
import os
import time
from collections import deque

import numpy as np
import torch
from tensorboardX import SummaryWriter, writer

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule


'''
--env-name=HalfCheetahBulletEnv-v0
--num-processes=8
--algo=ppo
--num-mini-batch=32
--lr=1e-4
--use-gae
--num-steps=512
--entropy-coef=0.00
--value-loss-coef=0.5
--ppo-epoch=4
--num-env-steps=8000000
--base=osc
--add-timestep
'''


args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

times = time.strftime("%b%d-%H%M", time.localtime())

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    writer = SummaryWriter(args.log_dir + f'/{times}/')
    # writer = SummaryWriter()

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, True, energy=0.01)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, base=args.base,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    # obs = envs.venv.reset()
    # obs = torch.FloatTensor(obs)
    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    mean_episode_rewards = 0.0

    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_lr_decay:
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))

        for step in range(args.num_steps):

            # Sample actions only
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Get reward and next observation
            obs, reward, done, infos = envs.step(action)

            # no use
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {} timesteps {} FPS {} Last {} mean/med rewd {:.1f}/{:.1f}, min/max rewd {:.1f}/{:.1f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            mean_episode_rewards = 0.9 * mean_episode_rewards + 0.1 * np.mean(episode_rewards)

        # if (args.eval_interval is not None
        #         and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     eval_envs = make_vec_envs(
        #         args.env_name, args.seed + args.num_processes, args.num_processes,
        #         args.gamma, eval_log_dir, args.add_timestep, device, True)
        #
        #     vec_norm = get_vec_normalize(eval_envs)
        #     if vec_norm is not None:
        #         vec_norm.eval()
        #         vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
        #
        #     eval_episode_rewards = []
        #
        #     obs = eval_envs.reset()
        #     eval_recurrent_hidden_states = torch.zeros(args.num_processes,
        #                     actor_critic.recurrent_hidden_state_size, device=device)
        #     eval_masks = torch.zeros(args.num_processes, 1, device=device)
        #
        #     while len(eval_episode_rewards) < 10:
        #         with torch.no_grad():
        #             _, action, _, eval_recurrent_hidden_states = actor_critic.act(
        #                 torch.cat((obs, obs), 1),       # we use obs as both current and prev value
        #                 eval_recurrent_hidden_states,
        #                 eval_masks,
        #                 deterministic=True)
        #
        #         # Obser reward and next obs
        #         obs, reward, done, infos = eval_envs.step(action)
        #
        #         eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
        #                                         for done_ in done])
        #         for info in infos:
        #             if 'episode' in info.keys():
        #                 eval_episode_rewards.append(info['episode']['r'])
        #
        #     eval_envs.close()
        #
        #     print(" Evaluation using {} episodes: mean reward {:.5f}\n".
        #         format(len(eval_episode_rewards),
        #                np.mean(eval_episode_rewards)))

        if j % args.vis_interval == 0:
            # writer.add_scalars(f'{times}',
            #                    {'Reward':rollouts.returns[-1].sum().cpu().numpy(),
            #                     'V Loss':value_loss,
            #                    'A Loss': action_loss
            #                     }, j)
            writer.add_scalar('_Reward', mean_episode_rewards, j)
            writer.add_scalar('ValLoss', value_loss, j)
            writer.add_scalar('ActLoss', action_loss, j)

    writer.close()


if __name__ == "__main__":

    # try:
    #     os.makedirs(args.log_dir)
    # except OSError:
    #     files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    #     for f in files:
    #         os.remove(f)

    eval_log_dir = args.log_dir + "_eval"

    # try:
    #     os.makedirs(eval_log_dir)
    # except OSError:
    #     files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    #     for f in files:
    #         os.remove(f)

    main()
