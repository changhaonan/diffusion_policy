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
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.pusht.pusht_control import PushTControlImageEnv
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def draw_action(image, agent_pos, naction, project_matrix=None, color=None):
    agent_pos = agent_pos @ project_matrix
    naction = naction @ project_matrix

    action_alpha = np.linspace(0.5, 1, naction.shape[1])
    for i in range(naction.shape[0]):
        traj_color = np.random.rand(3) * 255 if color is None else color
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
        **kwargs,
    ):
        self.env = PushTControlImageEnv(control_type=control_type, default_control=default_control, legacy=legacy_test, render_size=render_size)
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.n_obs_steps = n_obs_steps  # Number of observation steps
        self.n_action_steps = n_action_steps  # Number of action steps executed
        self.project_matrix = np.array([[1.0, 0], [0, 1.0]]) if "project_matrix" not in kwargs else kwargs["project_matrix"]

    def run(self, policy: BaseImagePolicy, seed=0, batch_size=64, enable_render=True):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        env.seed(seed)

        obs = env.reset()
        policy.reset()
        policy.cfg_ratio = 0.0

        obs_deque = collections.deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)
        for i in range(self.max_steps):
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
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict, gate=0)
                neg_action_dict = policy.predict_action(obs_dict, gate=2)

            # device_transfer
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
            action = np_action_dict["action"]
            np_neg_action_dict = dict_apply(neg_action_dict, lambda x: x.detach().to("cpu").numpy())
            neg_action = np_neg_action_dict["action"]

            for idx in range(self.n_action_steps):
                # Step env
                obs, reward, done, info = env.step(action[0][idx])
                obs_deque.append(obs)

                # Render
                if enable_render:
                    img = env.render(mode="rgb_array")
                    # Reshape to 512x512
                    img = cv2.resize(img, (512, 512))
                    # draw score
                    img = draw_action(img, agent_poses[0], action, self.project_matrix, color=np.array([0, 255, 0]))
                    img = draw_action(img, agent_poses[0], neg_action, self.project_matrix, color=np.array([0, 0, 255]))
                    cv2.imshow("image", img)
                    cv2.waitKey(1)
                    # # Press space to pause
                    # if cv2.waitKey(10) == 32:
                    #     print("Paused")
                    #     cv2.waitKey(0)
                if done:
                    break
            print(f"Step {i} / {self.max_steps}")
            cv2.waitKey(1)
            if done:
                break
        cv2.destroyAllWindows()


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, device):
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

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

    # Run score analysis
    seed = 1
    batch_size = 64
    score_analysis = PushTScoreAnalysis(output_dir)
    score_analysis.run(policy, seed=seed, batch_size=batch_size, enable_render=True)


if __name__ == "__main__":
    main()
