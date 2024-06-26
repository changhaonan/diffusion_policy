"""Demonstrate push-T with additional control"""

import numpy as np
import cv2
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_control import PushTControlEnv
from diffusion_policy.demo_utils.misc_utils import get_data_stats, normalize_data, create_sample_indices, sample_sequence

import pygame


@click.command()
@click.option("-o", "--output", default="data/kowndi_pusht_demo_v1.zarr", type=str)
@click.option("-c", "--control", default="repulse", help="region, repulse, follow")
@click.option("-dv", "--demo_violate", default=True, help="Record violating demonstrations.")
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(output, control, demo_violate, render_size, control_hz):
    """
    Collect demonstration for the Push-T task.

    Usage: python demo_pusht.py -o data/pusht_demo.zarr

    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area.
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """

    control_repeat = 2  # repeat each control for K times
    violate_repeat = 2  # repeat each violating episode for K times
    # create replay buffer in read-write mode
    output = output.replace(".zarr", "") + f"_{control}.zarr"
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")
    num_existing_episodes = replay_buffer.n_episodes
    # create PushT env with control
    env = PushTControlEnv(control_type=control.strip(), render_size=render_size)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    reset_state = None
    loop_counter = 0
    # episode-level while loop
    while True:
        episode = list()
        if not demo_violate:
            # collecting normal demonstrations
            # record in seed order, starting with 0
            seed = replay_buffer.n_episodes // control_repeat
            if replay_buffer.n_episodes % control_repeat == (control_repeat - 1):
                env.set_control(False)
                print("No control...")
            else:
                env.set_control(True)
                print("With control...")
            print(f"starting seed {seed}")

            # set seed for env
            env.seed(seed)

            # reset env and get observations (including info and render for recording)
            obs = env.reset()
        else:
            # collecting violating demonstrations, sample from existing episodes
            if reset_state is None or loop_counter % violate_repeat == 0:
                random_episode = np.random.randint(0, num_existing_episodes)
                # set seed for env
                seed = random_episode // control_repeat
                env.seed(seed)
                sampled_episode = replay_buffer.get_episode(random_episode)
                episode_len = sampled_episode["img"].shape[0]
                if episode_len < 10 or sampled_episode["demo_type"][0].item() == 2:
                    # skip if the episode is too short or already violating
                    continue
                # sample a state
                state_idx = np.random.randint(0, int(0.7 * episode_len))  # sample from the first 70% of the episode
                reset_state = sampled_episode["state"][state_idx]
            # reset env to the sampled state
            obs = env.reset_from_state(reset_state)

        info = env._get_info()
        img = env.render(mode="human")

        # loop state
        retry = False
        pause = False
        done = False
        skip = False
        plan_idx = 0
        pygame.display.set_caption(f"plan_idx:{plan_idx}")
        # step-level while loop
        while not done and not skip:
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
                    elif event.key == pygame.K_k:
                        # press "K" to skip
                        skip = True
                        print("skip")
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
                keypoint = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]
                data = {
                    "img": img,
                    "state": np.float32(state),
                    "keypoint": np.float32(keypoint),
                    "action": np.float32(act),
                    "n_contacts": np.float32([info["n_contacts"]]),
                    "control": env.get_control_image(),
                    "demo_type": np.array([int(env.is_control)]) if not demo_violate else np.array([2]),  # 0: normal, 1: control, 2: violate
                }
                control_img = data["control"]
                # Overlay control image on img
                img = cv2.addWeighted(img, 0.5, control_img, 0.5, 0)
                cv2.imshow("control", img)
                cv2.waitKey(1)
                episode.append(data)

            # step env and render
            obs, reward, done, info = env.step(act)
            violate = env.violate[-1] if env.violate else 0

            if demo_violate and violate:
                if len(episode) > 0:
                    done = True
                else:
                    retry = True
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
        print(f"n_episodes: {replay_buffer.n_episodes}; n_steps: {replay_buffer.n_steps}")
        loop_counter += 1


if __name__ == "__main__":
    main()
