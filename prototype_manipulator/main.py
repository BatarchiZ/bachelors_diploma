import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython

import csv
e = IPython.embed

def main(args):
    # with open("zz_final_aloha/ordered_products.csv") as file:
    #     orders
    orders = []
    with open("/Users/is/VSCode/bachelors_diploma/zz_final_aloha/ordered_products.csv") as csvfile:
        file_reader = csv.reader(csvfile, delimiter=",")
        for idx, row in enumerate(file_reader):
            if idx == 0:
                continue
            print(row)
            orders.append(torch.tensor(int(row[1])))
    print(orders)
    # exit()
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # # fixed parameters
    state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_name = [f'policy_best.ckpt']
        eval_bc(config, ckpt_name,orders=orders,save_episode=True)
        print("DONE")
        exit()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    return policy


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    return curr_image

# load policy and stats
def initialize_policy(ckpt_name, ckpt_dir, policy_class, policy_config):

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    print(ckpt_dir)
    print(ckpt_name)
    print("aaaaaaaaaaaaaaa")
    ckpt_path = os.path.join(str(ckpt_dir), str(ckpt_name))
    policy = make_policy(policy_class, policy_config)
    # if ckpt_dir == "/Users/is/VSCode/bachelors_diploma/z_final_aloha/models/blue":
    loading_status = policy.load_state_dict(torch.load(ckpt_path)['policy_state_dict'])
    # else:
    #     loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)

    policy.to(torch.device("cpu"))
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    return policy, pre_process, post_process


def eval_bc(config, ckpt_name, orders, save_episode=True):
    set_seed(2)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    from sim_env import make_sim_env
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    ### set task
    if 'sim_transfer_cube' in task_name:
        print(BOX_POSE[0])
        BOX_POSE[0] = np.concatenate((sample_box_pose(),sample_box_pose())) # used in sim reset
        print(BOX_POSE[0], "THE POSES OF THE BOX")
    ts = env.reset()

    ### onscreen render
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        plt.ion()

    image_list = [] # for visualization
    policy_names = "policy_best.ckpt"
    policy_paths = "/Users/is/VSCode/bachelors_diploma/zz_final_aloha/models_v2"
    for o in range(len(orders)):
        policy, pre_process, post_process = initialize_policy(ckpt_name=policy_names, 
                                                              ckpt_dir=policy_paths, 
                                                              policy_class=policy_class, 
                                                              policy_config=policy_config)
        # ts = env.reset()
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim])

        qpos_history = torch.zeros((1, max_timesteps, state_dim))
        qpos_list = []
        target_qpos_list = []
        rewards = []
        
        # GRASPING OBJECT
        with torch.inference_mode():
            for t in range(200):
            # for t in range(100):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                # qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos = torch.from_numpy(qpos).float().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, type=orders[o])
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action
                ### step the environment
                ts = env.step(target_qpos) 
                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                print(t, np.round(target_qpos, 3))
            # Open gripper and repeat for each item
            for t in range(10):
                cur = qpos_list[-1]
                cur[-1] = 1.1 # open_gripper
                ts = env.step(cur)
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
            for t in range(1):
                ts = env.step(qpos_list[0])

    plt.close()

    if save_episode:
        save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{1}.mp4'))

    return [], []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))