"""Control Interface for PushT Environment."""

import numpy as np
import pygame
import cv2
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from math import cos, sin


def sample_points_on_shape(shape, points_per_edge):
    sampled_points = []
    vertices = shape.get_vertices()
    body = shape.body

    for i in range(len(vertices)):
        # Transform vertices to world coordinates
        start = body.local_to_world(vertices[i])
        end = body.local_to_world(vertices[(i + 1) % len(vertices)])

        # Sample points along the edge
        for j in range(points_per_edge):
            t = j / (points_per_edge - 1) if points_per_edge > 1 else 0.5
            point = start * (1 - t) + end * t
            sampled_points.append(point)

    return sampled_points


class PushTControlEnv(PushTEnv):
    """Compared with image env, this env includes control image as observation.
    - control_type: str, control type, e.g. "region", "repulse", "follow"
    """

    def __init__(self, control_type="region", default_control=True, legacy=False, block_cog=None, damping=None, render_size=96, render_action=False):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping, render_size=render_size, render_action=render_action)
        ws = self.window_size
        self.control_type = control_type
        self.controls = None
        self.control_random_vals = None
        self.control_params = {
            "region": {"min_size": 200, "max_size": 400, "shape_offset": 100},
            "repulse": {"radius": 20, "sample_points": 2},
            "follow": {"grid": 10, "eps": 0.01},
        }
        self.control_counter = 0
        self.control_update_freq = 10000
        self.is_control = default_control
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32),
                "control": spaces.Box(low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32),
                "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
            }
        )
        self.violate = list()  # Measure how many times the agent violates the control

    def _compute_violate(self):
        agent_pos = np.array(self.agent.position)
        if self.control_type == "repulse" and self.controls is not None:
            for point in self.controls:
                dist = np.linalg.norm(agent_pos - point)
                if dist < self.control_params["repulse"]["radius"]:
                    return 1.0
        elif self.control_type == "follow" and self.controls is not None:
            # Compute the distance to the grid
            grid_size_x = int(self.controls[0] * self.window_size)
            grid_size_y = int(self.controls[1] * self.window_size)
            dist_x = agent_pos[0] % grid_size_x
            dist_y = agent_pos[1] % grid_size_y
            dist_x = min(dist_x, grid_size_x - dist_x)
            dist_y = min(dist_y, grid_size_y - dist_y)
            eps = int(self.control_params["follow"]["eps"] * self.window_size)
            if dist_x > eps and dist_y > eps:
                return 1.0
        return 0.0

    def set_control(self, flag):
        self.is_control = flag

    def reset(self):
        obs = super().reset()
        self.controls = None
        self.control_random_vals = None
        self.control_counter = 0
        self.violate = list()
        self._init_control()
        return obs

    def _init_control(self):
        # Generate control values
        seed = self._seed
        rs = np.random.RandomState(seed=seed)
        if self.control_type == "repulse":
            # Repulse the agent from a point on the shape
            shape_lists = []
            for shape in self.space.shapes:
                if hasattr(shape, "is_target_object") and shape.is_target_object:
                    # This is shape1
                    shape_lists.append(shape)
            if len(shape_lists) == 0:
                return np.zeros((3, self.render_size, self.render_size), dtype=np.float32)
            else:
                points = []
                for shape in shape_lists:
                    points += sample_points_on_shape(shape, self.control_params[self.control_type]["sample_points"])  # Sample one point
            self.control_random_vals = rs.choice(len(points), 1)
            self.controls = [np.array(points[i]) for i in self.control_random_vals]
        elif self.control_type == "region":
            min_size = self.control_params[self.control_type]["min_size"]
            max_size = self.control_params[self.control_type]["max_size"]
            shape_offset = self.control_params[self.control_type]["shape_offset"]
            # Control region should be a bbox includes the block pos and the goal pos
            block_pos = self.block.position
            goal_pos = self.goal_pose[:2]
            min_pos = np.minimum(block_pos, goal_pos)
            min_pos = np.maximum(min_pos - shape_offset, 0)
            max_pos = np.maximum(block_pos, goal_pos)
            max_pos = np.minimum(max_pos + shape_offset, 512)
            # Initial bbox size
            bbox_size = max_pos - min_pos
            if np.any(bbox_size < min_size):
                size_diff = np.maximum(min_size - bbox_size, 0)
                min_pos -= size_diff / 2
                max_pos += size_diff / 2

            # Randomly adjust bbox size up to max_size constraint, ensuring it includes the original points
            for i in range(2):  # For both x and y dimensions
                max_expand = max_size - (max_pos[i] - min_pos[i])
                if max_expand > 0:
                    expand = rs.uniform(0, max_expand)
                    # Randomly distribute the expansion on both sides
                    expand_min = rs.uniform(0, expand)
                    expand_max = expand - expand_min
                    min_pos[i] -= expand_min
                    max_pos[i] += expand_max
            bbox = [min_pos, max_pos]  # This is your randomized bbox
            self.controls = bbox
        elif self.control_type == "follow":
            self.controls = rs.uniform(0.05, 0.07, (2))

    def _update_control(self):
        if self.control_type == "repulse":
            # Repulse the agent from a point on the shape
            shape_lists = []
            for shape in self.space.shapes:
                if hasattr(shape, "is_target_object") and shape.is_target_object:
                    # This is shape1
                    shape_lists.append(shape)
            if len(shape_lists) == 0:
                return np.zeros((3, self.render_size, self.render_size), dtype=np.float32)
            else:
                points = []
                for shape in shape_lists:
                    points += sample_points_on_shape(shape, self.control_params[self.control_type]["sample_points"])  # Sample one point
            self.controls = [np.array(points[i]) for i in self.control_random_vals]

    def step(self, action):
        action = self.regularize_act(action)
        obs, reward, done, info = super().step(action)
        self._update_control()
        self.violate.append(self._compute_violate())
        return obs, reward, done, info

    def regularize_act(self, act):
        if act is not None and self.is_control and self.controls is not None:
            if self.control_type == "follow":
                # Regularize the action to be around the grid by eps
                eps = int(self.control_params[self.control_type]["eps"] * self.window_size)
                act_x = act[0]
                act_y = act[1]
                grid_size_x = int(self.controls[0] * self.window_size)
                grid_size_y = int(self.controls[1] * self.window_size)
                round_x = int(round(act_x / grid_size_x) * grid_size_x)
                round_y = int(round(act_y / grid_size_y) * grid_size_y)
                res_x = np.clip(act_x - round_x, -eps, eps)
                res_y = np.clip(act_y - round_y, -eps, eps)
                if abs(res_x) < eps or abs(res_y) < eps:
                    # Meaning it is already close to the grid
                    act = (act_x, act_y)
                else:
                    act = (round_x + res_x, round_y + res_y)
        return act

    def get_control_image(self):
        """Get control image."""
        control_image = np.zeros((self.render_size, self.render_size, 3), dtype=np.float32)
        if self.is_control and self.controls is not None:
            if self.control_type == "repulse":
                for point in self.controls:
                    point_coord = (point / 512 * self.render_size).astype(np.int32)
                    radius = int(self.control_params[self.control_type]["radius"] / 512 * self.render_size)
                    cv2.circle(control_image, tuple(point_coord), radius, (255, 0, 0), -1)
            elif self.control_type == "region":
                cv2.rectangle(control_image, tuple((self.controls[0] / 512 * self.render_size).astype(np.int32)), tuple((self.controls[1] / 512 * self.render_size).astype(np.int32)), (0, 255, 0), -1)
            elif self.control_type == "follow":
                # Follow the grid
                # Draw grid that the agent should follow
                grid_size_x = int(self.controls[0] * self.render_size)
                grid_size_y = int(self.controls[1] * self.render_size)
                for i in range(0, self.render_size, grid_size_x):
                    cv2.line(control_image, (i, 0), (i, self.render_size), (0, 0, 255), 1)
                for i in range(0, self.render_size, grid_size_y):
                    cv2.line(control_image, (0, i), (self.render_size, i), (0, 0, 255), 1)
            self.control_counter += 1
        return control_image.astype(np.uint8)

    def _draw_control_signal(self):
        temp_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))  # Make the surface transparent
        if self.is_control and self.controls is not None:
            if self.control_type == "repulse":
                for point in self.controls:
                    point_coord = (point / 512 * self.window_size).astype(np.int32)
                    radius = int(self.control_params[self.control_type]["radius"])
                    color = (255, 0, 0, 128)
                    pygame.draw.circle(temp_surface, color, point_coord, radius)
            elif self.control_type == "follow":
                # Draw grid that the agent should follow
                grid_size_x = int(self.controls[0] * self.window_size)
                grid_size_y = int(self.controls[1] * self.window_size)
                for i in range(0, self.window_size, grid_size_x):
                    pygame.draw.line(temp_surface, (0, 0, 255, 128), (i, 0), (i, self.window_size), 5)
                for i in range(0, self.window_size, grid_size_y):
                    pygame.draw.line(temp_surface, (0, 0, 255, 128), (0, i), (self.window_size, i), 5)
            elif self.control_type == "region":
                min_pos = self.controls[0]
                max_pos = self.controls[1]
                min_pos = (min_pos / 512 * self.window_size).astype(np.int32)
                max_pos = (max_pos / 512 * self.window_size).astype(np.int32)
                pygame.draw.rect(temp_surface, (0, 255, 0, 128), (min_pos[0], min_pos[1], max_pos[0] - min_pos[0], max_pos[1] - min_pos[1]))
        self.window.blit(temp_surface, temp_surface.get_rect())


class PushTControlImageEnv(PushTControlEnv):
    """Compared with control env, this env is used for evaluation."""

    def __init__(self, control_type="region", default_control=True, legacy=False, block_cog=None, damping=None, render_size=96, render_action=False):
        super().__init__(control_type, default_control, legacy, block_cog, damping, render_size, render_action)

    def _get_obs(self):
        img = super()._render_frame(mode="rgb_array")
        control_img = self.get_control_image()
        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        control_obs = np.moveaxis(control_img.astype(np.float32) / 255, -1, 0)
        obs = {"image": img_obs, "control": control_obs, "agent_pos": agent_pos}
        return obs

    def render(self, mode):
        assert mode == "rgb_array"
        image = super().render(mode)
        control_image = self.get_control_image()
        image = image * 0.5 + control_image * 0.5
        return image.astype(np.uint8)
