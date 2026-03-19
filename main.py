import os
import sys
import csv
import json
import time
import numpy as np
import random
import logging
import argparse
import torch
from os import path, makedirs
from datetime import datetime
from data.patches import Patches
from data.screenshots import Screenshots
from data.interactions import Interactions
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from models.vision_encoder import VisionEncoder, VisionDecoder, VisionAE
from models.callback import TrainingCallback
from models.vision_agent import VisionAgent
from models.finger_agent import FingerAgent
from models.supervisor_agent import SupervisorAgent
from models.memory import Memory, MemoryEncoder
from typing_env.vision_env import VisionEnv
from typing_env.finger_env import FingerEnv
from typing_env.internal_env import InternalEnv
from typing_env.hybrid_env import HybridInternalEnv
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from metrics import Metrics, ChordMetrics
from tqdm import tqdm
from setting import KEYS, KEYS_FIN, PLACES, PLACES_FIN, CHARS, CHARS_FIN, KEYS_KALQ, PLACES_KALQ, CHARS_KALQ
from copy import copy
from parameters import parameters

parser = argparse.ArgumentParser()


def weights_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)


# General parameters
parser.add_argument("--vision-encoder", action="store_true", default=False, help="train the vision encoder")
parser.add_argument("--vision-agent", action="store_true", default=False, help="train the vision agent")
parser.add_argument("--peripheral-encoder", action="store_true", default=False,
                    help="train the screenshot vison encoder")
parser.add_argument("--finger-agent", action="store_true", default=False, help="train the finger agent")
parser.add_argument("--memory", action="store_true", default=False, help="train the working memory")
parser.add_argument("--supervisor-agent", action="store_true", default=False, help="train the supervisor agent")
parser.add_argument("--continue-training", action="store_true", default=False, help="continue train the previous agent")
parser.add_argument("--train", action="store_true", default=False, help="run model in train mode")
parser.add_argument("--evaluate", action="store_true", default=False, help="run model in evaluate mode")
parser.add_argument("--demo", action="store_true", default=False, help="run model in demo mode")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--log-interval", type=int, default=100, help="log interval")
parser.add_argument("--total-timesteps", type=int, default=100000, help="total timesteps for training the agent")
parser.add_argument("--phrase", type=str, default=False, metavar='path', help='phrase for evaluation')
parser.add_argument('--encoder-path', type=str, default='outputs/f_encoder.pt', metavar='path',
                    help='path for the vision encoder')
parser.add_argument('--decoder-path', type=str, default='outputs/f_decoder.pt', metavar='path',
                    help='path for the vision decoder')
parser.add_argument('--peripheral-encoder-path', type=str, default='outputs/p_encoder.pt', metavar='path',
                    help='path for the peripheral encoder')
parser.add_argument('--peripheral-decoder-path', type=str, default='outputs/p_decoder.pt', metavar='path',
                    help='path for the peripheral decoder')
parser.add_argument("--gboard", action="store_true", default=False, help="train and test on Gboard")
parser.add_argument("--chord", action="store_true", default=False, help="use hybrid chord/sequential typing (HybridInternalEnv)")
parser.add_argument("--chord-vocab", type=int, default=15, metavar='N', help="limit chord vocabulary to top N entries by word length (default: all)")

args = parser.parse_args()

if args.train:
    if args.vision_encoder:
        writer = SummaryWriter("runs/f_encoder")
        # Vision encoder parameters
        parser.add_argument('--save-path-encoder', type=str, default='outputs/f_encoder.pt', metavar='path',
                            help='output path for the vision encoder')
        parser.add_argument('--save-path-decoder', type=str, default='outputs/f_decoder.pt', metavar='path',
                            help='output path for the vision decoder')
        args = parser.parse_args()

        dataset = Patches(number_row=False, punctuation=False)
        # dataset = FoveatedImages()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        encoder = VisionEncoder()
        decoder = VisionDecoder()
        encoder.apply(weights_init)
        decoder.apply(weights_init)
        encoder.train()
        decoder.train()
        if not args.no_cuda:
            encoder.cuda()
            decoder.cuda()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

        model = VisionAE(encoder, decoder)

        runloss = None
        start_time = time.time()
        loss_idx = 0

        for epoch_nb in range(1, args.epochs + 1):
            for batch_idx, (foveated_images) in enumerate(tqdm(dataloader)):
                if not args.no_cuda:
                    foveated_images = foveated_images.cuda()

                recon_images = model(foveated_images)
                loss = model.loss(recon_images, foveated_images)

                # param update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss /= len(foveated_images)
                writer.add_scalar("Loss/train", loss, loss_idx)
                loss_idx += 1

                runloss = loss if not runloss else runloss * 0.9 + loss * 0.1

                if not batch_idx % args.log_interval:
                    print('\n Epoch {}, batch: {}/{} ({:.2f} s), loss: {:.2f}'.format(
                        epoch_nb, batch_idx, len(dataloader), time.time() - start_time, runloss))
                    start_time = time.time()

        writer.flush()
        writer.close()

        torch.save(encoder.state_dict(), args.save_path_encoder)
        torch.save(decoder.state_dict(), args.save_path_decoder)

    elif args.peripheral_encoder:
        writer = SummaryWriter("runs/peripheral_encoder")
        # Vision encoder parameters
        parser.add_argument('--save-path-encoder', type=str, default='outputs/p_encoder.pt', metavar='path',
                            help='output path for the vision encoder')
        parser.add_argument('--save-path-decoder', type=str, default='outputs/p_decoder.pt', metavar='path',
                            help='output path for the vision decoder')
        args = parser.parse_args()

        dataset = Screenshots()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        encoder = VisionEncoder()
        decoder = VisionDecoder()
        encoder.apply(weights_init)
        decoder.apply(weights_init)
        encoder.train()
        decoder.train()
        if not args.no_cuda:
            encoder.cuda()
            decoder.cuda()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

        model = VisionAE(encoder, decoder)

        runloss = None
        start_time = time.time()
        loss_idx = 0

        for epoch_nb in range(1, args.epochs + 1):
            for batch_idx, (foveated_images) in enumerate(dataloader):
                if not args.no_cuda:
                    foveated_images = foveated_images.cuda()

                recon_images = model(foveated_images)
                loss = model.loss(recon_images, foveated_images)

                # param update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss /= len(foveated_images)
                writer.add_scalar("Loss/train", loss, loss_idx)
                loss_idx += 1

                runloss = loss if not runloss else runloss * 0.9 + loss * 0.1

                if not batch_idx % args.log_interval:
                    print('Epoch {}, batch: {}/{} ({:.2f} s), loss: {:.2f}'.format(
                        epoch_nb, batch_idx, len(dataloader), time.time() - start_time, runloss))
                    start_time = time.time()

        writer.flush()
        writer.close()

        torch.save(encoder.state_dict(), args.save_path_encoder)
        torch.save(decoder.state_dict(), args.save_path_decoder)

    elif args.vision_agent:
        writer = SummaryWriter("runs/vision_agent")
        foveal_encoder = VisionEncoder()
        foveal_encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
        foveal_encoder.eval()
        peripheral_encoder = VisionEncoder()
        peripheral_encoder.load_state_dict(torch.load(args.peripheral_encoder_path, map_location=torch.device('cpu')))
        peripheral_encoder.eval()
        env = VisionEnv(img_folder='kbd1k/keyboard_dataset/', position_file='kbd1k/keyboard_label.csv',
                        places=PLACES, foveal_encoder=foveal_encoder, peripheral_encoder=peripheral_encoder)
        model_path = 'outputs/vision_agent.pt'
        # Create log dir
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        if args.continue_training:
            agent = VisionAgent(env, load=model_path)
        else:
            agent = VisionAgent(env)
        agent.learn(total_timesteps=args.total_timesteps,
                    callback=TrainingCallback(check_freq=1000, log_dir="logs/", save_path=model_path))
        print("learning finished")
        # agent.save("outputs/vision_agent.pt")

    elif args.finger_agent:
        writer = SummaryWriter("runs/finger_agent")
        encoder = VisionEncoder()
        # encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu'))) # foveal vision
        encoder.load_state_dict(
            torch.load(args.peripheral_encoder_path, map_location=torch.device('cpu')))  # peripheral vision
        encoder.eval()
        env = FingerEnv(img_folder='kbd1k/keyboard_dataset/', position_file='kbd1k/keyboard_label.csv', places=KEYS,
                        with_noise=False, encoder=encoder)
        model_path = 'outputs/finger_agent.pt'
        # Create log dir
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        if args.continue_training:
            agent = FingerAgent(env, load=model_path)
        else:
            agent = FingerAgent(env)
        agent.learn(total_timesteps=args.total_timesteps,
                    callback=TrainingCallback(check_freq=1000, log_dir="logs/", save_path=model_path))
        print("learning finished")

    elif args.memory:
        writer = SummaryWriter("runs/work_memory")
        encoder = VisionEncoder()
        encoder.load_state_dict(
            torch.load(args.peripheral_encoder_path, map_location=torch.device('cpu')))  # peripheral vision
        encoder.eval()
        env = FingerEnv(img_folder='kbd1k/keyboard_dataset/', position_file='kbd1k/keyboard_label.csv', places=KEYS,
                        with_noise=True, encoder=encoder)  # Gboard
        agent = FingerAgent(env=env, load='outputs/finger_agent.pt')
        model_path = "outputs/wm_encoder.pt"

        dataset = Interactions(env=env, agent=agent, places=KEYS)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        clf = MemoryEncoder(input_size=64 + 2, num_classes=len(KEYS))
        clf.apply(weights_init)
        clf.train()
        if not args.no_cuda:
            clf.cuda()
        optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr)

        runloss = None
        start_time = time.time()
        loss_idx = 0

        for epoch_nb in range(1, args.epochs + 1):
            for batch_idx, (x, y, key) in enumerate(dataloader):
                if not args.no_cuda:
                    x = x.to('cuda')
                    y = y.to('cuda')

                y_hat = clf(x)
                loss = clf.loss(y_hat, y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/train", loss, loss_idx)
                loss_idx += 1

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch_nb, args.epochs, loss.item()))

        writer.flush()
        writer.close()
        print("save model to ", model_path)
        torch.save(clf.state_dict(), model_path)

    elif args.supervisor_agent:
        writer = SummaryWriter("runs/supervisor_agent")
        if args.gboard:
            env = InternalEnv(img_folder='kbd1k/gboard/', position_file='kbd1k/keyboard_label.csv',
                                text_path='./data/sentences.txt', vision_path='outputs/vision_agent.pt',
                                finger_path='outputs/finger_agent.pt', chars=CHARS, places=PLACES, keys=KEYS)
        else:
            env = InternalEnv(img_folder='kbd1k/keyboard_dataset/', position_file='kbd1k/keyboard_label.csv',
                                text_path='./data/sentences.txt', vision_path='outputs/vision_agent.pt',
                                finger_path='outputs/finger_agent.pt', chars=CHARS, places=PLACES, keys=KEYS)
        model_path = "outputs/supervisor_agent.pt"
        # Create log dir
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        if args.continue_training:
            agent = SupervisorAgent(env, load=model_path)
        else:
            agent = SupervisorAgent(env)
        agent.learn(total_timesteps=args.total_timesteps, callback=TrainingCallback(check_freq=20000, log_dir = "logs/", save_path=model_path))
        print("learning finished")
        agent.save(model_path)

elif args.evaluate:
    if args.vision_agent:
        foveal_encoder = VisionEncoder()
        foveal_encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu')))
        foveal_encoder.eval()
        peripheral_encoder = VisionEncoder()
        peripheral_encoder.load_state_dict(torch.load(args.peripheral_encoder_path, map_location=torch.device('cpu')))
        peripheral_encoder.eval()
        env = VisionEnv(img_folder='kbd1k/keyboard_dataset_test/', position_file='kbd1k/keyboard_label.csv',
                        places=PLACES, foveal_encoder=foveal_encoder, peripheral_encoder=peripheral_encoder)
        agent = VisionAgent(env=env, load='outputs/vision_agent.pt')

        # Evaluate the agent
        # mean_reward, std_reward
        rewards, timesteps = evaluate_policy(agent, env, n_eval_episodes=1000, deterministic=True, render=False,
                                             callback=None, reward_threshold=None, return_episode_rewards=True)
        # print average reward
        print("mean reward: ", np.mean(rewards))
        # print average timesteps
        print("mean timesteps: ", np.mean(timesteps))
        # print proportion of episodes that only use one timestep to reach the goal
        print("success rate: ", np.sum(np.array(rewards) > 0) / len(rewards))
        # print(ret)
    elif args.finger_agent:
        encoder = VisionEncoder()
        # encoder.load_state_dict(torch.load(args.encoder_path, map_location=torch.device('cpu'))) # foveal vision
        encoder.load_state_dict(
            torch.load(args.peripheral_encoder_path, map_location=torch.device('cpu')))  # peripheral vision
        encoder.eval()
        if args.gboard:
            env = FingerEnv(img_folder='kbd1k/gboard/', position_file='kbd1k/keyboard_label.csv', places=KEYS,
                            with_noise=True, encoder=encoder)
            agent = FingerAgent(env=env, load='outputs/finger_agent.pt')
        else:
            env = FingerEnv(img_folder='kbd1k/keyboard_dataset_test/', position_file='kbd1k/keyboard_label.csv',
                            places=KEYS, with_noise=False, encoder=encoder)
            agent = FingerAgent(env=env, load='outputs/finger_agent.pt')

        # Evaluate the agent
        # mean_reward, std_reward
        rewards, timesteps = evaluate_policy(agent, env, n_eval_episodes=1000, deterministic=True, render=False,
                                             callback=None, reward_threshold=None, return_episode_rewards=True)
        # print average reward
        print("mean reward: ", np.mean(rewards))
        # print average timesteps
        print("mean timesteps: ", np.mean(timesteps))
        # print proportion of episodes that only use one timestep to reach the goal
        print("success rate: ", np.sum(np.array(rewards) > 0.5) / len(rewards))
        # print(ret)
    elif args.memory:
        interaction_num = 1000
        encoder = VisionEncoder()
        encoder.load_state_dict(
            torch.load(args.peripheral_encoder_path, map_location=torch.device('cpu')))  # peripheral vision
        encoder.eval()
        env = FingerEnv(img_folder='kbd1k/keyboard_dataset_test/', position_file='kbd1k/keyboard_label.csv',
                        places=KEYS, with_noise=True, encoder=encoder)  # Gboard
        agent = FingerAgent(env=env, load='outputs/finger_agent.pt')
        dataset = Interactions(env=env, agent=agent, places=KEYS, num=interaction_num)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True
        )
        clf = MemoryEncoder(input_size=64 + 2, num_classes=len(KEYS))
        clf.load_state_dict(torch.load('outputs/wm_encoder.pt', map_location=torch.device('cpu')))
        clf.eval()

        correct_num = 0
        for x, y, key in dataloader:
            y_hat = clf(x)
            loss = clf.loss(y_hat, y)
            _, predicted = torch.max(y_hat.data, 1)
            if predicted == y:
                correct_num += 1
            # else: # print wrong prediction
            #     print("wrong prediction:", KEYS[predicted],  KEYS[y])
            #     print(key)
        print("memory accuracy: ", correct_num / interaction_num)
        print("memory loss: ", loss.item())

    elif args.supervisor_agent:
        EnvClass = HybridInternalEnv if args.chord else InternalEnv
        MetricsClass = ChordMetrics if args.chord else Metrics

        env = EnvClass(img_folder='kbd1k/gboard/', position_file='kbd1k/keyboard_label.csv',
                       text_path='./data/sentences.txt', vision_path='outputs/vision_agent.pt',
                       finger_path='outputs/finger_agent.pt', chars=CHARS, places=PLACES, keys=KEYS)
        agent = SupervisorAgent(env=env, load='outputs/supervisor_agent.pt')
        results = []

        print('Given parameters: ', parameters)
        i = 0
        for i in tqdm(range(30)): # 30 independent runs
            while True:
                obs = env.reset(parameters=parameters)
                done = False
                while not done:
                    action, _states = agent.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                metrics = MetricsClass(log=env.log, target_text=env.target_text)
                summary = metrics.summary()
                if summary['char_error_rate'] < 0.2: break # remove outliers
                # else: continue print("remove outlier")
            results.append(summary)
        print("WPM: ", np.mean([result['WPM'] for result in results]), np.std([result['WPM'] for result in results]))
        print("IKI: ", np.mean([result['IKI'] for result in results]), np.std([result['IKI'] for result in results]))
        print("error_rate: ", (np.mean([result['char_error_rate'] * 100 for result in results])), "%", np.std([result['char_error_rate'] * 100 for result in results]))
        # print([result['num_backspaces'] for result in results])
        print("num_backspaces: ", np.mean([result['num_backspaces'] for result in results]),
              np.std([result['num_backspaces'] for result in results]))
        print("gaze_kbd_ratio: ", np.mean([result['gaze_kbd_ratio'] for result in results]),
              np.std([result['gaze_kbd_ratio'] for result in results]))
        print("gaze_shift: ",
              np.mean([result['gaze_shift'] for result in results]),
              np.std([result['gaze_shift'] for result in results]))

        if args.chord:
            print("chord_count: ",
                  np.mean([result['chord_count'] for result in results]),
                  np.std([result['chord_count'] for result in results]))
            print("chord_use_rate: ",
                  np.mean([result['chord_use_rate'] for result in results]),
                  np.std([result['chord_use_rate'] for result in results]))
            print("chord_wpm_contribution: ",
                  np.mean([result['chord_wpm_contribution'] for result in results]),
                  np.std([result['chord_wpm_contribution'] for result in results]))

        print('Average performance')
        results = []
        for i in tqdm(range(100)):  # 100 independent runs
            memory_param = random.uniform(0, 1)
            finger_param = random.uniform(0, 1)
            vision_param = random.uniform(0, 1)
            parameters = [memory_param, finger_param, vision_param]
            obs = env.reset(parameters=parameters)
            done = False
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            metrics = Metrics(log=env.log, target_text=env.target_text)
            summary = metrics.summary()
            if summary['char_error_rate'] < 1:  # remove outliers
                results.append(summary)
        print("WPM: ", np.mean([result['WPM'] for result in results]))
        print('Median performance')
        print("WPM: ", np.median([result['WPM'] for result in results]))

        print('Peak Performance (best):  memory 0, finger 0, vision 0')
        results = []
        parameters = [0, 0, 0]
        for i in tqdm(range(30)):  # 30 independent runs
            obs = env.reset(parameters=parameters)
            done = False
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                # if done: print("reward:", reward)
            metrics = Metrics(log=env.log, target_text=env.target_text)
            summary = metrics.summary()
            if summary['char_error_rate'] < 1:  # remove outliers
                results.append(metrics.summary())
        print("WPM: ", np.mean([result['WPM'] for result in results]), np.std([result['WPM'] for result in results]))

        print('Worst Performance: memory 1, finger 1, vision 1')
        results = []
        parameters = [1, 1, 1]
        for i in tqdm(range(30)):  # 30 independent runs
            obs = env.reset(parameters=parameters)
            done = False
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            metrics = Metrics(log=env.log, target_text=env.target_text)
            summary = metrics.summary()
            if summary['char_error_rate'] < 1:  # remove outliers
                results.append(summary)
        print("WPM: ", np.mean([result['WPM'] for result in results]), np.std([result['WPM'] for result in results]))

elif args.demo:
    EnvClass = HybridInternalEnv if args.chord else InternalEnv

    if args.gboard:
        env = EnvClass(img_folder='kbd1k/gboard/', position_file='kbd1k/keyboard_label.csv',
                       text_path='./data/sentences.txt', vision_path='outputs/vision_agent.pt',
                       finger_path='outputs/finger_agent.pt', chars=CHARS, places=PLACES, keys=KEYS,
                       render_mode='human')
    else:
        env = EnvClass(img_folder='kbd1k/keyboard_dataset/', position_file='kbd1k/keyboard_label.csv',
                       text_path='./data/sentences.txt', vision_path='outputs/vision_agent.pt',
                       finger_path='outputs/finger_agent.pt', chars=CHARS, keys=KEYS, render_mode='human')
    agent = SupervisorAgent(env=env, load='outputs/supervisor_agent.pt')
    
    obs = env.reset()
    if args.phrase:
        target_text = args.phrase
    else:
        target_text = None
    while (True):
        obs = env.reset(target_text=target_text, parameters=parameters, reset_kbd=True)
        done = False
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()