"""Tools for visualize dataset"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def vis_naction(image, agent_pos, naction, stats, project_matrix=None, show=False):
    """Visualize multiple action sequences in one image.
    - image: (3, H, W)
    - agent_pos: (2,)
    - naction: (B, T, 2)
    """
    B = naction.shape[0]
    # Unnormalize
    if stats is not None:
        naction = unnormalize_data(naction, stats["action"])
        agent_pos = unnormalize_data(agent_pos, stats["agent_pos"])
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    elif isinstance(image, np.ndarray) and (image.shape[0] == 3 or image.shape[0] == 1):
        image = image.transpose(1, 2, 0)
    if image.max() > 1:
        image /= 255.0
    if isinstance(agent_pos, torch.Tensor):
        agent_pos = agent_pos.cpu().numpy()
    if isinstance(naction, torch.Tensor):
        naction = naction.cpu().numpy()
    # Project
    if project_matrix is not None:
        agent_pos = agent_pos @ project_matrix
        naction = naction @ project_matrix
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    # ax.plot(agent_pos[0], agent_pos[1], "bo")
    action_alpha = np.linspace(0, 1, naction.shape[1])
    for i in range(B):
        color = plt.cm.jet(i / B)
        for j in range(naction.shape[1]):
            ax.plot(naction[i, j, 0], naction[i, j, 1], "ro", alpha=action_alpha[j], color=color, markersize=3)
        # Connect actions
        ax.plot(naction[i, :, 0], naction[i, :, 1], color=color)
    if show:
        plt.show()
    # Save
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image_np


def vis_data(data, stats, project_matrix=None, annotator=None, show=False, render="plt"):
    """Visualize data in batch.
    - data: dict, keys: image, agent_pos, action
    - project_matrix: 3x3/2x2 matrix for projection, projecting action & pos to image space
    """
    image = data["image"]
    agent_pos = data["agent_pos"]
    action = data["action"]
    #
    B = image.shape[0]
    # Convert to numpy
    image = image.cpu().numpy().transpose(0, 1, 3, 4, 2)
    # image /= 255.0
    agent_pos = agent_pos.cpu().numpy()
    action = action.cpu().numpy()
    # Unnormalize
    action = unnormalize_data(action, stats["action"])
    agent_pos = unnormalize_data(agent_pos, stats["agent_pos"])
    # Project
    if project_matrix is not None:
        agent_pos = agent_pos @ project_matrix
        action = action @ project_matrix
    image_list = []
    for i in range(B):
        if render == "cv2":
            # Assuming image[i, 1] is in RGB format and needs to be converted to BGR for OpenCV.
            img = cv2.cvtColor(image[i, 1].astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Draw agent positions as blue circles
            for pos in agent_pos[i]:
                cv2.circle(img, (int(pos[0]), int(pos[1])), 3, (255, 0, 0), -1)  # Blue circles

            # Action color is red and gradually changes with time
            action_alpha = np.linspace(0, 1, action.shape[1])
            for j in range(action.shape[1]):
                # Calculate color intensity based on alpha for the red color
                intensity = int(255 * action_alpha[j])
                cv2.circle(img, (int(action[i, j, 0]), int(action[i, j, 1])), 3, (1, 1, intensity), -1)  # Red circles
            if show:
                cv2.imshow(f"Image {i}", img)
                cv2.waitKey(0)  # Wait for any key press
        elif render == "plt":
            # Plt
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.axis("off")
            ax.imshow(image[i, 1] / 255.0)
            ax.plot(agent_pos[i, :, 0], agent_pos[i, :, 1], "bo")
            # Action color is red and gradually changes with time
            action_alpha = np.linspace(0, 1, action.shape[1])
            for j in range(action.shape[1]):
                ax.plot(action[i, j, 0], action[i, j, 1], "ro", alpha=action_alpha[j])
            # Add annotation
            if annotator is not None:
                anno = data["anno"][i]
                if isinstance(anno, torch.Tensor):
                    anno = anno.cpu().numpy()
                anno_image = annotator.show_anno(anno)
                ax.imshow(anno_image, alpha=0.5)
            if show:
                plt.show()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            img = img.reshape((height, width, 3))

        image_list.append(img)
        cv2.destroyAllWindows()
    return image_list
