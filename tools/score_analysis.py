"""Analysis the score of data."""

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
        fps=10,
        crf=22,
    ):
        steps_per_render = max(10 // fps, 1)
        self.env = MultiStepWrapper(
            VideoRecordingWrapper(
                PushTControlImageEnv(control_type=control_type, default_control=default_control, legacy=legacy_test, render_size=render_size),
                video_recoder=VideoRecorder.create_h264(fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf, thread_type="FRAME", thread_count=1),
                file_path=None,
                steps_per_render=steps_per_render,
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )
        self.max_steps = max_steps
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy, seed=0, batch_size=32, enable_render=True):
        def init_fn(env, seed=seed, enable_render=enable_render):
            # setup rendering
            # video_wrapper
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            if enable_render:
                filename = pathlib.Path(self.output_dir).joinpath("media", wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env.env.file_path = filename

            # set seed
            assert isinstance(env, MultiStepWrapper)
            env.seed(seed)

        device = policy.device
        dtype = policy.dtype
        env = self.env
        init_fn(env)

        obs = env.reset()
        policy.reset()

        for i in range(self.max_steps):
            print(f"Step {i} / {self.max_steps}")
            # create obs dict
            np_obs_dict = dict(obs)

            # Augment obs dict by batch_size
            np_obs_dict = dict_apply(np_obs_dict, lambda x: np.repeat(x[None], batch_size, axis=0))

            # device transfer
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict, gate=1)

            # device_transfer
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
            action = np_action_dict["action"]

            # random select one to execute
            action = action[0]

            # step env
            obs, reward, done, info = env.step(action)

            if done:
                break

        # close env
        video_path = env.render()
        print(f"Video saved to {video_path}")


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    else:
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
    score_analysis = PushTScoreAnalysis(output_dir)
    score_analysis.run(policy)


if __name__ == "__main__":
    main()
