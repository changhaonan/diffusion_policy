import os
import random
import numpy as np
import zarr
import cv2
from tqdm.auto import tqdm
from tools.dataset_manipulate import read_from_path, convert


def get_bbox(points=None, original_bbox=None, num_augmentations=2, min_margin=0.05, max_margin=0.1):
    """Get bounding box from the trajectories."""
    # Find the tightest bounding box
    if points is not None and original_bbox is None:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
    elif original_bbox is not None:
        min_x, min_y, max_x, max_y = original_bbox
    # Calculate initial width and height
    initial_width = max_x - min_x
    initial_height = max_y - min_y

    bounding_boxes = []
    margin_per = [min_margin, max_margin]
    for _ in range(num_augmentations):
        # Generate a random margin based on the percentage of the initial dimensions
        # margin_percentage = random.uniform(min_margin, max_margin)
        margin_percentage = margin_per.pop(0)
        margin = max(initial_width, initial_height) * margin_percentage
        # Define the top-left and bottom-right coordinates for clarity
        top_left = (min_x - margin, min_y - margin)
        bottom_right = (max_x + margin, max_y + margin)
        if top_left[0] < 0:
            top_left = (0, top_left[1])
        if top_left[1] < 0:
            top_left = (top_left[0], 0)
        if bottom_right[0] > 96:
            bottom_right = (96, bottom_right[1])
        if bottom_right[1] > 96:
            bottom_right = (bottom_right[0], 96)
        bounding_box = {"top_left": top_left, "bottom_right": bottom_right}
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def save_dict2zarr(data, group):
    for key, value in data.items():
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            save_dict2zarr(value, sub_group)
        else:
            group.create_dataset(key, data=np.array(value))


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    src_data = "kowndi_pusht_demo_v0_region.zarr"
    tar_data = "kowndi_pusht_demo_v0_region_aug0.zarr"
    tar_data = os.path.join(root_dir, tar_data)

    render_size = 96
    root = read_from_path(os.path.join(root_dir, src_data))

    aug_root = {
        "data": {
            "state": [],
            "control": [],
            "img": [],
            "action": [],
            "n_contacts": [],
        },
        "meta": {
            "episode_ends": [],
        },
    }

    for i in tqdm(range(0, len(root["meta"]["episode_ends"])), desc="Copying initial data"):
        aug_root["meta"]["episode_ends"].append(root["meta"]["episode_ends"][i])
        if i == 0:
            traj_indices = range(root["meta"]["episode_ends"][i])
        else:
            traj_indices = range(root["meta"]["episode_ends"][i - 1], root["meta"]["episode_ends"][i])
        for j in tqdm(traj_indices, desc="Loading data", leave=False):
            aug_root["data"]["state"].append(root["data"]["state"][j])
            aug_root["data"]["img"].append(root["data"]["img"][j])
            aug_root["data"]["action"].append(root["data"]["action"][j])
            aug_root["data"]["n_contacts"].append(root["data"]["n_contacts"][j])
            aug_root["data"]["control"].append(root["data"]["control"][j])

    for i in tqdm(range(0, len(root["meta"]["episode_ends"]), 2), desc="Augmenting New Data"):
        original_episode_ends = aug_root["meta"]["episode_ends"][-1]
        if i == 0:
            traj_indices = range(root["meta"]["episode_ends"][i])
        else:
            traj_indices = range(root["meta"]["episode_ends"][i - 1], root["meta"]["episode_ends"][i])
        aug_root["meta"]["episode_ends"].append(len(traj_indices) + original_episode_ends)

        control_img = root["data"]["control"][traj_indices[0]]

        # Detect the green channel being significantly higher than the other two in the green area
        green_mask = (control_img[:, :, 1] > control_img[:, :, 0]) & (control_img[:, :, 1] > control_img[:, :, 2])

        # Get the coordinates of the green area
        coordinates = np.argwhere(green_mask)
        assert coordinates.shape[0] > 0
        top_left = [coordinates[:, 0].min(), coordinates[:, 1].min()]
        bottom_right = [coordinates[:, 0].max(), coordinates[:, 1].max()]

        bboxes = get_bbox(points=None, original_bbox=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
        bbox = bboxes[0]
        controls = np.array([bbox["top_left"], bbox["bottom_right"]])
        control_image = np.zeros((render_size, render_size, 3), dtype=np.int32)
        # Swapping the control coordinates to match the image coordinates, I do not know why this is necessary
        control_image = cv2.rectangle(control_image, tuple((controls[0][::-1]).astype(np.int32)), tuple((controls[1][::-1]).astype(np.int32)), (0, 255, 0), -1).astype(np.uint8)
        cshow = cv2.addWeighted(control_img, 0.5, control_image, 0.5, 0)
        # cv2.imshow("control", cshow)
        # cv2.waitKey(0)
        points = []
        for j in tqdm(traj_indices, desc="Augmenting data", leave=False):
            aug_root["data"]["state"].append(root["data"]["state"][j])
            aug_root["data"]["img"].append(root["data"]["img"][j])
            aug_root["data"]["action"].append(root["data"]["action"][j])
            aug_root["data"]["n_contacts"].append(root["data"]["n_contacts"][j])
            aug_root["data"]["control"].append(control_image.astype(np.uint8))

            # points.append(root["data"]["action"][j])
            # points.append(root["data"]["state"][j][:2]) # appending the end effector position
            # min_offset, max_offset = (-140, -130), (140, 130)
            # min_pos = np.maximum(root["data"]["state"][j][2:4] + min_offset, 0)
            # max_pos = np.minimum(root["data"]["state"][j][2:4] + max_offset, 512)
            # points.append(min_pos) # appending the T-block position
            # points.append(max_pos)

        original_episode_ends = aug_root["meta"]["episode_ends"][-1]
        aug_root["meta"]["episode_ends"].append(len(traj_indices) + original_episode_ends)
        bbox = bboxes[1]
        controls = np.array([bbox["top_left"], bbox["bottom_right"]])
        control_image = np.zeros((render_size, render_size, 3), dtype=np.int32)
        # Swapping the control coordinates to match the image coordinates, I do not know why this is necessary
        control_image = cv2.rectangle(control_image, tuple((controls[0][::-1]).astype(np.int32)), tuple((controls[1][::-1]).astype(np.int32)), (0, 255, 0), -1).astype(np.uint8)
        cshow = cv2.addWeighted(control_img, 0.5, control_image, 0.5, 0)
        # cv2.imshow("control", cshow)
        # cv2.waitKey(0)
        points = []
        for j in tqdm(traj_indices, desc="Augmenting data", leave=False):
            aug_root["data"]["state"].append(root["data"]["state"][j])
            aug_root["data"]["img"].append(root["data"]["img"][j])
            aug_root["data"]["action"].append(root["data"]["action"][j])
            aug_root["data"]["n_contacts"].append(root["data"]["n_contacts"][j])
            aug_root["data"]["control"].append(control_image.astype(np.uint8))

    zarr_group = zarr.open_group(tar_data, "w")
    save_dict2zarr(aug_root, zarr_group)
