"""Analysis the score of data."""

import collections
import hydra
import click
import torch
import dill
import numpy as np
import cv2
import os
import pathlib
from tqdm.auto import tqdm
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.pusht.pusht_control import PushTControlImageEnv
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.control_utils.trajectory_filter import trajectory_filter
from diffusion_policy.control_utils.frequence_policy import FreqActionFieldPolicy


def draw_action(image, agent_pos, naction, project_matrix=None, color=None):
    agent_pos = agent_pos @ project_matrix
    naction = naction @ project_matrix

    # action_alpha = np.linspace(0.5, 1, naction.shape[1])
    action_alpha = np.ones(naction.shape[1])
    for i in range(naction.shape[0]):
        traj_color = np.random.rand(3) * 255 if color is None else (color * 0.5 + i / len(naction) * np.array([0, 0, 255])).astype(int)
        for j in range(naction.shape[1]):
            color_alpha = traj_color * action_alpha[j]
            cv2.circle(image, tuple(naction[i, j].astype(int)), 2, color_alpha, 1)
        # Connect actions
        for j in range(naction.shape[1] - 1):
            color_alpha = traj_color * action_alpha[j]
            cv2.line(image, tuple(naction[i, j].astype(int)), tuple(naction[i, j + 1].astype(int)), color_alpha, 1)
    return image


class PushTScoreAnalysis:
    def __init__(
        self,
        output_dir,
        control_type="repulse",
        default_control=True,
        legacy_test=False,
        render_size=96,
        max_steps=200,
        n_obs_steps=2,
        n_action_steps=8,
        use_filter=False,
        **kwargs,
    ):
        self.env = PushTControlImageEnv(control_type=control_type, default_control=default_control, legacy=legacy_test, render_size=render_size)
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.n_obs_steps = n_obs_steps  # Number of observation steps
        self.n_action_steps = n_action_steps  # Number of action steps executed
        self.project_matrix = np.array([[1.0, 0], [0, 1.0]]) if "project_matrix" not in kwargs else kwargs["project_matrix"]
        self.use_filter = use_filter

    def run(self, policy: BaseImagePolicy, seed=0, batch_size=64, record_step=5, gates=[0, 1], enable_render=True):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        env.seed(seed)

        obs = env.reset()
        info = env._get_info()
        policy.reset()
        policy.cfg_ratio = 0.0

        obs_deque = collections.deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)
        img_list = []
        for i in tqdm(range(int(self.max_steps / self.n_action_steps)), desc="Score Analysis: "):
            # Create obs dict
            images = np.stack([x["image"] for x in obs_deque])
            agent_poses = np.stack([x["agent_pos"] for x in obs_deque])
            controls = np.stack([x["control"] for x in obs_deque])
            np_obs_dict = {
                "image": images,
                "agent_pos": agent_poses,
                "control": controls,
            }
            # Augment obs dict by batch_size
            np_obs_dict = dict_apply(np_obs_dict, lambda x: np.repeat(x[None], batch_size, axis=0))
            # device transfer
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

            # Run policy
            gate_action_dict = {}
            with torch.no_grad():
                for gate in gates:
                    action_dict = policy.predict_action(obs_dict, gate=gate, info=info)
                    gate_action_dict[gate] = action_dict

            # device_transfer
            for gate in gates:
                gate_action_dict[gate] = dict_apply(gate_action_dict[gate], lambda x: x.to(device="cpu").numpy())

            action = gate_action_dict[gates[0]]["action"]  # First gate

            if self.use_filter:
                # Filter trajectory
                action = trajectory_filter(action, control=np_obs_dict["control"][0, -1], control_type=env.control_type)
                gate_action_dict[gates[0]]["action"] = action

            for idx in range(self.n_action_steps):
                # Step env
                obs, reward, done, info = env.step(action[0][idx])
                violate = env._compute_violate(threshold=22)
                if violate:
                    print(f"Violate at step {i} / {self.max_steps}")
                obs_deque.append(obs)
                img = env.render(mode="rgb_array")
                # Reshape to 512x512
                img = cv2.resize(img, (512, 512))
                # draw score
                for gate in gates:
                    gate_action = gate_action_dict[gate]["action"]
                    if gate == 0:
                        color = np.array([255, 0, 0])
                    elif gate == 1:
                        color = np.array([0, 255, 0])
                    elif gate == 2:
                        color = np.array([0, 0, 255])
                    else:
                        color = np.array([255, 255, 255])
                    img = draw_action(img, agent_poses[0], gate_action, self.project_matrix, color=color)

                # Render
                if enable_render:
                    # print(f"Step {i} / {self.max_steps}")
                    cv2.imshow("image", img)
                    cv2.waitKey(1)
                if done:
                    break

            if i % record_step == 0:
                img_list.append(img.copy())
            if done:
                break
        return img_list


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, device):
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_frequence_policy = True

    if not use_frequence_policy:
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        # Override cfg
        cfg.policy.cfg_ratio = 0.0
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
    else:
        # load frequence policy
        policy = FreqActionFieldPolicy(zarr_path="/home/harvey/Project/diffusion_policy/data/kowndi_pusht_demo_v2_repulse.zarr", horizon=16, pad_before=1, pad_after=7)

    # Run score analysis
    seed = 10011
    use_filter = False
    gates = [0]
    batch_size = 64
    n_action_steps = 8
    max_steps = 600
    policy.n_action_steps = n_action_steps
    score_analysis = PushTScoreAnalysis(output_dir, n_action_steps=n_action_steps, max_steps=max_steps, use_filter=use_filter)
    score_analysis.run(policy, seed=seed, batch_size=batch_size, gates=gates, enable_render=True)


if __name__ == "__main__":
    main()
