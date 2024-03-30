"""Control Interface for PushT Environment."""

import numpy as np
import pygame
import cv2
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv


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
    - control_type: str, control type, e.g. "contact", "repulse", "follow"
    """

    def __init__(self, control_type="contact", legacy=False, block_cog=None, damping=None, render_size=96, render_action=False):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping, render_size=render_size, render_action=render_action)
        ws = self.window_size
        self.control_type = control_type
        self.controls = None
        self.control_params = {"contact": {"radius": 60}, "repulse": {"radius": 40}, "follow": {"grid": 10, "eps": 0.01}}
        self.control_counter = 0
        self.control_update_freq = 10000
        self.is_control = True
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32),
                "control": spaces.Box(low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32),
                "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
            }
        )

    def set_control(self, flag):
        self.is_control = flag

    def reset(self):
        obs = super().reset()
        self.controls = None
        return obs

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
                act = (round_x + res_x, round_y + res_y)
        return act

    def get_control_image(self):
        """Get control image."""
        control_image = np.zeros((self.render_size, self.render_size, 3), dtype=np.float32)
        if self.is_control:
            if self.control_type == "contact" or self.control_type == "repulse":
                # Sample a point on block
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
                        points += sample_points_on_shape(shape, 10)  # Sample one point
                if self.controls is None or self.control_counter % self.control_update_freq == 0:
                    self.control_random_vals = np.random.choice(len(points), 1)
                    print(f"control_random_vals: {self.control_random_vals}")
                self.controls = [np.array(points[i]) for i in self.control_random_vals]
                for point in self.controls:
                    point_coord = (point / 512 * self.render_size).astype(np.int32)
                    radius = int(self.control_params[self.control_type]["radius"] / 512 * self.render_size)
                    cv2.circle(control_image, tuple(point_coord), radius, (0, 255, 0) if self.control_type == "contact" else (255, 0, 0), -1)
            elif self.control_type == "follow":
                if self.controls is None or self.control_counter % self.control_update_freq == 0:
                    self.control_random_vals = np.random.uniform(0.05, 0.08, (2))
                # Draw grid that the agent should follow
                grid_size_x = int(self.control_random_vals[0] * self.render_size)
                grid_size_y = int(self.control_random_vals[1] * self.render_size)
                for i in range(0, self.render_size, grid_size_x):
                    cv2.line(control_image, (i, 0), (i, self.render_size), (0, 0, 255), 1)
                for i in range(0, self.render_size, grid_size_y):
                    cv2.line(control_image, (0, i), (self.render_size, i), (0, 0, 255), 1)
                self.controls = self.control_random_vals
            self.control_counter += 1
        return control_image.astype(np.uint8)

    def _draw_control_signal(self):
        temp_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))  # Make the surface transparent
        if self.control_type == "contact" or self.control_type == "repulse":
            if self.controls is not None:
                for point in self.controls:
                    point_coord = (point / 512 * self.window_size).astype(np.int32)
                    radius = int(self.control_params[self.control_type]["radius"])
                    color = (0, 255, 0, 128) if self.control_type == "contact" else (255, 0, 0, 128)
                    pygame.draw.circle(temp_surface, color, point_coord, radius)
        elif self.control_type == "follow":
            if self.controls is not None:
                # Draw grid that the agent should follow
                grid_size_x = int(self.controls[0] * self.window_size)
                grid_size_y = int(self.controls[1] * self.window_size)
                for i in range(0, self.window_size, grid_size_x):
                    pygame.draw.line(temp_surface, (0, 0, 255, 128), (i, 0), (i, self.window_size), 5)
                for i in range(0, self.window_size, grid_size_y):
                    pygame.draw.line(temp_surface, (0, 0, 255, 128), (0, i), (self.window_size, i), 5)
        self.window.blit(temp_surface, temp_surface.get_rect())
