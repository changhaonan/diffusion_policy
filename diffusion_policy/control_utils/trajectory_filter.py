"""Filter out violating trajectory"""

import numpy as np
import cv2
import heapq


def trajectory_filter(trajectory, control, control_type, strategy="naive"):
    """
    control: control information.
    control_type: control type. ["repulse", "region"]
    """
    if strategy == "naive":
        return trajectory_filter_naive(trajectory, control, control_type)
    else:
        raise NotImplementedError(f"Strategy {strategy} is not implemented")


def trajectory_filter_naive(trajectory, control, control_type):
    """
    trajectory: (B, T, 2)
    control: control information.
    control_type: control type. ["repulse", "region"]
    """
    if control.shape[0] == 3:
        control = np.moveaxis(control, 0, -1)
    if control_type == "repulse" and control.max() != 0:
        # Compute the bounding box of the control
        mask = np.any(control > 0, axis=2)
        non_zero_indices = np.argwhere(mask)
        y_min, x_min = non_zero_indices.min(axis=0)
        y_max, x_max = non_zero_indices.max(axis=0)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        # map center back to action space
        x_center, y_center = (x_center / 96 * 512, y_center / 96 * 512)

        # Filtering
        # threshold = (np.linalg.norm([x_max - x_min, y_max - y_min]) / 96 * 512) / 2
        # filtered_trajectory = []
        # for i in range(trajectory.shape[0]):
        #     if np.linalg.norm(trajectory[i, trajectory.shape[1] // 2 :] - [x_center, y_center], axis=-1).min() > threshold:
        #         filtered_trajectory.append(trajectory[i])
        # if len(filtered_trajectory) < trajectory.shape[0]:
        #     print(f"Filtered {trajectory.shape[0] - len(filtered_trajectory)} trajectories")
        # if len(filtered_trajectory) == 0:
        #     print("No trajectory is found after filtering.")
        #     return trajectory
        # return np.array(filtered_trajectory)
        # Ranking:
        # Rank the trajectory based on the distance to the center
        ranked_trajectory = heapq.nlargest(trajectory.shape[0], trajectory, key=lambda x: np.linalg.norm(x - [x_center, y_center], axis=1).min())
        # Return the first k trajectory, shuffled
        ranked_trajectory = np.array(ranked_trajectory)
        np.random.shuffle(ranked_trajectory[: ranked_trajectory.shape[0] // 3])  # Shuffle the first 1/3
        return ranked_trajectory
    else:
        return trajectory
