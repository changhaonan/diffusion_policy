"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import numpy as np
import cv2
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import collections
from matplotlib import pyplot as plt
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.control_utils.knn_policy import KNNPolicy, KNNSAPolicy
from diffusion_policy.control_utils.policy_sample_tree import PolicySampleTree
from diffusion_policy.env.pusht.pusht_env import PushTEnv


############################## Utils ##############################
def extract_skeleton_from_tree(depths, childrens):
    # Compue all leaf nodes first
    leaf_nodes = []
    for i in range(len(childrens)):
        if len(childrens[i]) == 0:
            leaf_nodes.append(i)
    # Compute node parents
    parents = [-1] * len(childrens)
    for i in range(len(childrens)):
        for child in childrens[i]:
            parents[child] = i
    # Compute all tree skeleton by traversing back from leaf nodes
    tree_skeletons = []
    for leaf_node in leaf_nodes:
        node_id = leaf_node
        depth = depths[node_id]
        skeleton = []
        while depth >= 0:
            skeleton.append(node_id)
            node_id = parents[node_id]
            depth -= 1
        tree_skeletons.append(skeleton[::-1])
    return tree_skeletons


def visualize_policy_tree(states, actions, depths, childrens):
    env = PushTEnv()
    env.seed(0)
    env.reset()
    # Get the image for the state
    image_list = []
    for i in range(len(states)):
        state = states[i][-1]
        env._set_state(state)
        image = env.render(mode="rgb_array")
        # Draw action on the image
        action = actions[i]
        # Rescale the action for push-T env
        if action is not None:
            action = action / 512 * 96
            for k in range(action.shape[0]):
                color = np.random.randint(0, 255, 3)
                color = [int(color[0]), int(color[1]), int(color[2])]
                for j in range(action.shape[1]):
                    cv2.circle(image, (int(action[k, j, 0]), int(action[k, j, 1])), 2, color, 1)
        # Resize the image
        image = cv2.resize(image, (256, 256))
        image_list.append(image)

    # Compute all tree skeletons using BFS
    tree_skeletons = extract_skeleton_from_tree(depths, childrens)
    for tree_skeleton in tree_skeletons:
        skeleton_image = [image_list[node_id] for node_id in tree_skeleton if image_list[node_id] is not None]
        skeleton_image = np.concatenate(skeleton_image, axis=1)
        skeleton_str = ", ".join([str(node_id) for node_id in tree_skeleton])
        cv2.imshow(f"skeleton-{skeleton_str}", skeleton_image)
        cv2.waitKey(0)


def test_policy_sample_tree(policy_sample_tree: PolicySampleTree, seed, n_samples=10):
    obs_horizon = policy_sample_tree.n_obs_steps
    # Init env
    env = PushTEnv()
    env.seed(seed)
    obs = env.reset()  # This is state
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    for i in range(1):
        # Expand states
        states = np.stack(obs_deque, axis=0)  # (obs_step, Do)
        obs_dict = {"state": states}
        policy_sample_tree.expand_tree(obs_dict)
        # Check the expanded tree
        states, actions, depths, childrens = policy_sample_tree.export()
        visualize_policy_tree(states, actions, depths, childrens)


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-r", "--data_root", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-a", "--algorithm", default="knn_sa", help="diffusion, knn, knn_sa")
def main(checkpoint, data_root, output_dir, device, algorithm):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if algorithm == "diffusion":
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        # Override cfg
        cfg.policy.cfg_ratio = -0.1
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device(device)
        policy.to(device)
        policy.eval()
    elif algorithm == "knn":
        knn = 2
        policy = KNNPolicy(zarr_path=f"{data_root}/kowndi_pusht_demo_v2_repulse.zarr", horizon=16, pad_before=1, pad_after=7, knn=knn)
    elif algorithm == "knn_sa":
        knn = 5
        policy = KNNSAPolicy(zarr_path=f"{data_root}/kowndi_pusht_demo_v2_repulse.zarr", horizon=16, pad_before=1, pad_after=7, knn=knn)
        sample_tree = PolicySampleTree(policy, k_sample=5, max_depth=2)

    test_policy_sample_tree(sample_tree, seed=5)
    # # run eval
    # env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    # runner_log = env_runner.run(policy)

    # # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     elif isinstance(value, wandb.sdk.data_types.image.Image):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # out_path = os.path.join(output_dir, "eval_log.json")
    # json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
