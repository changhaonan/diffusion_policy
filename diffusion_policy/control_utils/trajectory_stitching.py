"""Stitching different trajectories together.
The idea of trajectory stitching is switching between similar states.
"""

import zarr
import os
import pickle
import cv2
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
import numpy as np
from tqdm.auto import tqdm


def visualize_trajectory(image, closest_image, raw_agent_pos, stitched_agent_pos, raw_actions, stitched_actions, project_matrix=None):
    raw_actions = raw_actions @ project_matrix if project_matrix is not None else raw_actions
    stitched_actions = stitched_actions @ project_matrix if project_matrix is not None else stitched_actions

    # Draw agent position
    cv2.circle(image, tuple(raw_agent_pos.astype(int)), 10, (255, 0, 255), 1)
    cv2.circle(closest_image, tuple(stitched_agent_pos.astype(int)), 10, (0, 255, 255), 1)

    # Draw raw actions
    action_alpha = np.linspace(0.5, 1, raw_actions.shape[0])
    color = np.array([0, 0, 255])
    for j in range(raw_actions.shape[0]):
        color_alpha = color * action_alpha[j]
        cv2.circle(image, tuple(raw_actions[j].astype(int)), 2, color_alpha, 1)
    # Connect actions
    for j in range(raw_actions.shape[0] - 1):
        color_alpha = color * action_alpha[j]
        cv2.line(image, tuple(raw_actions[j].astype(int)), tuple(raw_actions[j + 1].astype(int)), color_alpha, 1)

    # Draw stitched actions
    action_alpha = np.linspace(0.5, 1, stitched_actions.shape[0])
    color = np.array([0, 255, 0])
    for j in range(stitched_actions.shape[0]):
        color_alpha = color * action_alpha[j]
        cv2.circle(closest_image, tuple(stitched_actions[j].astype(int)), 2, color_alpha, 1)
    # Connect actions
    for j in range(stitched_actions.shape[0] - 1):
        color_alpha = color * action_alpha[j]
        cv2.line(closest_image, tuple(stitched_actions[j].astype(int)), tuple(stitched_actions[j + 1].astype(int)), color_alpha, 1)
    # Concatenate the images
    image = np.concatenate([image, closest_image], axis=1)
    return image


def stitch_trajectory(replay_buffer: ReplayBuffer, episode_id, episode_chunk, states, episode_ids, step_ids, mask, enable_render=False):
    """
    mask: mask out episode ids that are not allowed to be stitched
    """
    if mask is not None:
        assert mask.shape[0] == episode_ids.shape[0], "Mask shape mismatch"
    # Randomly select a state from current episode
    chunk_size = episode_chunk["state"].shape[0]
    # random_state_idx = np.random.randint(chunk_size)
    start_state_idx = 0
    random_state = episode_chunk["state"][start_state_idx]
    # Find the closest state in the current episode
    distances = np.linalg.norm(states - random_state, axis=1)
    # Mask out the current episode
    distances[episode_ids.flatten() == episode_id] = np.inf
    if mask is not None:
        distances[mask] = np.inf
    closest_state_idx = np.argmin(distances)
    closest_state = states[closest_state_idx]
    closest_episode_id = episode_ids[closest_state_idx].item()
    closest_step_id = step_ids[closest_state_idx].item()
    # Stitch the trajectory
    closest_episode = replay_buffer.get_episode(closest_episode_id)
    current_actions = episode_chunk["action"]
    closest_actions = closest_episode["action"]
    # Stitch the trajectory
    if start_state_idx != 0:
        stitched_state = np.concatenate([episode_chunk["state"][:start_state_idx], closest_episode["state"][closest_step_id:]])
        stitched_action = np.concatenate([current_actions[:start_state_idx], closest_actions[closest_step_id:]])
    else:
        stitched_state = closest_episode["state"][closest_step_id:]
        stitched_action = closest_actions[closest_step_id:]
    # Chunk & padd the stitched trajectory
    if stitched_state.shape[0] < chunk_size:
        stitched_state = np.pad(stitched_state, ((0, chunk_size - stitched_state.shape[0]), (0, 0)), mode="edge")
        stitched_action = np.pad(stitched_action, ((0, chunk_size - stitched_action.shape[0]), (0, 0)), mode="edge")
    elif stitched_state.shape[0] > chunk_size:
        stitched_state = stitched_state[:chunk_size]
        stitched_action = stitched_action[:chunk_size]
    if enable_render:
        # Log
        print(f"Closest state: {closest_state_idx} in episode {closest_episode_id} at step {closest_step_id}")
        print(f"Random state: {random_state}; Closest state: {closest_state}")
        # Visualize the stitching
        raw_image = episode_chunk["img"][start_state_idx]
        closest_image = closest_episode["img"][closest_step_id]
        raw_agent_pos = episode_chunk["state"][start_state_idx][:2]
        stitched_agent_pos = closest_episode["state"][closest_step_id][:2]
        # Resize the image
        raw_image = cv2.resize(raw_image, (512, 512))
        closest_image = cv2.resize(closest_image, (512, 512))
        stitched_image = visualize_trajectory(raw_image.copy(), closest_image.copy(), raw_agent_pos, stitched_agent_pos, current_actions, stitched_action)
        cv2.imshow("Stitched Trajectory", stitched_image)
        cv2.waitKey(0)
    return [closest_episode_id, closest_step_id, stitched_state, stitched_action]


@click.command()
@click.option("-i", "--input_path", required=True)
@click.option("-o", "--output_path", required=True)
def main(input_path: str, output_path: str):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read the replay buffer
    replay_buffer = ReplayBuffer.copy_from_path(input_path)
    # Build array pair: (state, episode_id, step_id)
    states = replay_buffer["state"]
    # Emphasize the last dim of states
    states[..., -1] = states[..., -1] / np.pi * 200
    episode_lengths = replay_buffer.episode_lengths
    episode_ids = [[i] * l for i, l in enumerate(episode_lengths)]
    episode_ids = np.stack(sum(episode_ids, [])).reshape(-1, 1)

    step_ids = [list(range(l)) for l in episode_lengths]
    step_ids = np.stack(sum(step_ids, [])).reshape(-1, 1)

    horizon = 16
    pad_before = 7
    pad_after = 1
    enable_render = True
    sampler = SequenceSampler(replay_buffer=replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=None)

    stitch_mapping = np.zeros([len(sampler), 2], dtype=np.int32)
    for idx in tqdm(range(len(sampler))):
        # idx = np.random.randint(len(sampler))
        sample = sampler.sample_sequence(idx)
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sampler.indices[idx]
        sample_episode_id = episode_ids[buffer_start_idx].item()
        [closest_episode_id, closest_step_id, stitched_state, stitched_action] = stitch_trajectory(replay_buffer, sample_episode_id, sample, states, episode_ids, step_ids, enable_render=enable_render)
        stitch_mapping[idx, 0] = closest_episode_id
        stitch_mapping[idx, 1] = closest_step_id

    # Save the stitch_dict
    np.save(output_path, stitch_mapping)


if __name__ == "__main__":
    main()
