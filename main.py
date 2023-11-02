#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from environment import *


def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length, debug=False):

    l = len(env.data) - 1
    episodes = []
    total_rewards = []
    for episode in range(args.max_episode_length):
        episode_reward = 0
        observation = deepcopy(env.reset())
        agent.is_training = True
        agent.reset(observation)
        action_cnt = 0

        for step in range(l-agent.nb_states):
            if step <= args.warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(observation)
            # env response with next_observation, reward, terminate_info
            observation2, reward, done, info = env.step(step+1, action)
            observation2 = deepcopy(observation2)
            if args.max_episode_length and step >= args.max_episode_length - 1:
                done = True

            # agent observe and update policy
            agent.observe(reward, observation2, done)
            if step > args.warmup:
                agent.update_policy()

            # [optional] evaluate
            if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
                policy = lambda x: agent.select_action(x, decay_epsilon=False)
                validate_reward, results = evaluate(env, policy, debug=False, visualize=False)
                if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

            # [optional] save intermideate model
            if step % 10 == 0:
                agent.save_model(output)

            if (action == 1.0) | (action == -1.0):
                action_cnt = action_cnt + 1
            episode_reward += reward

            observation = deepcopy(observation2)
            if step == l - 1 - agent.nb_states:
                episodes.append(episode)
                total_rewards.append(episode_reward)
                if(episode != 0) & (episode % 100 == 0):
                    save_results('./output/validate_reward_' + str(episode), episodes, results, total_rewards)
                # print('===================================================')
                print('#{}: episode_reward:{} | validate_reward:{} | action zero count:{}'.format(episode, round(episode_reward, 2), round(validate_reward, 2), action_cnt))
                # print('===================================================')

                agent.memory.append(
                    observation,
                    agent.select_action(observation),
                    0., False
                )

                # reset
                observation = None
                episode_reward = 0.


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=True):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward, results = evaluate(env, agent, policy, debug=False, visualize=False)
        print('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

def save_results(fn, episodes, result, reward):

    y = np.mean(result, axis=0)
    error = np.std(result, axis=0)

    x = episodes
    ax = plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    ax.errorbar(x, y, yerr=error, fmt='-o')

    plt.subplot(2, 1, 2)
    plt.plot(x, reward)
    plt.xlabel('Timestep')
    plt.ylabel('Total Reward')
    plt.savefig(fn + '.png')
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='kubernetes_pod_container_vehicle_train', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=50, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=1000, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug',  dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run5'.format(args.env)

    data = util.getResourceDataVec(args.env)
    env = Environment(data)

    if args.seed > 0:
        np.random.seed(args.seed)
        # env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes,
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate,
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))