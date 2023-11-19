import argparse
import logging
import os
import sys

import cv2
import numpy as np
from policy import sac
from env.environment import IndoorDeepRL
from PIL import Image


def performance_measure(args, episode, agent, s_count, max_rate):
    if episode > 0 and episode % 50 == 0:
        s_rate = s_count / 50
        if s_rate >= max_rate:
            max_rate = s_rate
            if args.training:
                print("Save model to " + args.model_path)
                agent.save_load_model("save", args.model_path)
        print("Success Rate (current/max):", s_rate, "/", max_rate)
    return max_rate


def visualize(args, agent, total_eps=2, message=False, render=False, map_path="large.png", gif_path="performance/",
              gif_name="test.gif"):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    images = []

    mother_nature = IndoorDeepRL(map_path=args.terrain)
    for eps in range(total_eps):
        step = 0
        max_success_rate = 0
        success_count = 0

        state = mother_nature.createInstance()
        r_eps = []
        acc_reward = 0.

        while (True):
            action = agent.choose_action(state, eval=True)
            state_next, reward, terminated = mother_nature.step(action)
            displayed = mother_nature.render(gui=render)
            im_pil = Image.fromarray(cv2.cvtColor(np.uint8(displayed * 255), cv2.COLOR_BGR2RGB))
            images.append(im_pil)
            r_eps.append(reward)
            acc_reward += reward

            if message:
                print('\rEps: {:2d}| Step: {:4d} | action:{:+.2f}| R:{:+.2f}| Reps:{:.2f}  ' \
                      .format(eps, step, action[0], reward, acc_reward), end='')

            state = state_next.copy()
            step += 1

            if terminated or step > 200:
                if message:
                    print()
                break

    print("Create GIF ...")
    if gif_path is not None:
        images[0].save(gif_path + gif_name, save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--training', default=True)
    parser.add_argument('--render', default=True)
    parser.add_argument('--load_model', default=False)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--terrain', type=str, default="img/map.png")
    parser.add_argument('--gif_path', type=str, default="performance/")
    parser.add_argument('--model_path', type=str, default="models/")
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, default='data/output')

    args = parser.parse_args()
    log_file = os.path.join(args.output_dir, 'output.log')
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    agent_mind_sac = sac.SAC(model=[sac.PolicyNetGaussian, sac.QNet], b_size=sac.batch_size)
    # %%
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if args.load_model:
        print("Load model ...", args.model_path)
        agent_mind_sac.save_load_model("load", args.model_path)

    mother_nature = IndoorDeepRL(map_path=args.terrain)
    total_step = 0
    max_success_rate = 0
    success_count = 0
    for eps in range(4500):
        state = mother_nature.createInstance()
        step = 0
        loss_a = loss_c = 0.
        acc_reward = 0.

        while True:
            if args.training:
                action = agent_mind_sac.choose_action(state, eval=False)
            else:
                action = agent_mind_sac.choose_action(state, eval=True)

            state_next, reward, terminated = mother_nature.step(action)
            end = 0 if terminated else 1
            agent_mind_sac.store_transition(state, action, reward, state_next, end)

            displayed = mother_nature.render(gui=args.render)

            loss_a = loss_c = 0.
            if total_step > sac.batch_size and args.training:
                loss_a, loss_c = agent_mind_sac.learn()

            step += 1
            total_step += 1
            acc_reward += reward

            state = state_next.copy()
            if terminated or step > 200:
                if reward > 5:
                    success_count += 1
                break
        logging.info('eps:{}, step:{}, total_step:{}, reward:{}'.format(eps, step, total_step, acc_reward))
        max_success_rate = performance_measure(args, eps, agent_mind_sac, success_count, max_success_rate)
        success_count = 0

    visualize(args, agent_mind_sac, total_eps=8, map_path=args.terrain, gif_path=args.gif_path,
              gif_name="SAC_" + str(eps).zfill(4) + ".gif")


if __name__ == '__main__':
    main()
