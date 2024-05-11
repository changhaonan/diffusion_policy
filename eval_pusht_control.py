"""Evaluate the control ability of Push-T task."""

import numpy as np
import cv2
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_control import PushTControlEnv
from diffusion_policy.demo_utils.misc_utils import get_data_stats, normalize_data, create_sample_indices, sample_sequence

import pygame


@click.command()
@click.option("-c", "--config", default="./lowdim_pusht_control_diffusion_policy_cnn.yaml", type=str)
@click.option("-ch", "--checkpoint", default="./data/checkpoints/repulse_concat.ckpt", type=str)
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(output, control, render_size, control_hz):
    """
    Evaluate the control ability of Push-T task.
    """
    # create replay buffer in read-write mode
    output = output.replace(".zarr", "") + f"_{control}.zarr"
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")

    # create PushT env with control
    env = PushTControlEnv(control_type=control.strip(), render_size=render_size)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f"starting seed {seed}")

        # set seed for env
        env.seed(seed)

        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode="human")

        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f"plan_idx:{plan_idx}")
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f"plan_idx:{plan_idx}")
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry = True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue

            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            if not act is None:
                # teleop started
                # state dim 2+3
                state = np.concatenate([info["pos_agent"], info["block_pose"]])
                # discard unused information such as visibility mask and agent pos
                # for compatibility
                data = {"img": img, "state": np.float32(state), "action": np.float32(act), "n_contacts": np.float32([info["n_contacts"]]), "control": env.get_control_image()}
                control_img = data["control"]
                # Overlay control image on img
                img = cv2.addWeighted(img, 0.5, control_img, 0.5, 0)
                cv2.imshow("control", img)
                cv2.waitKey(1)
                episode.append(data)

            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode="human")

            # regulate control frequency
            clock.tick(control_hz)
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors="disk")
            print(f"saved seed {seed}")
        else:
            print(f"retry seed {seed}")


if __name__ == "__main__":
    main()
