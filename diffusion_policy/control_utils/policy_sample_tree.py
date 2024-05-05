"""Policy Sample Tree."""
import numpy as np
from diffusion_policy.policy.base_sa_policy import BaseSAPolicy
from collections import namedtuple

SANode = namedtuple("SANode", ["state", "action", "depth", "value", "children"])

class PolicySampleTree:

    def __init__(self, policy: BaseSAPolicy, k_sample: int, max_depth: int):
        self.policy = policy
        self.k_sample = k_sample
        self.max_depth = max_depth
        self.nodes = []
        self.frontiers = []
        self.n_act_steps = policy.n_act_steps
        self.n_obs_steps = policy.n_obs_steps

    def reset(self):
        self.nodes = []
        self.frontiers = []

    def expand_tree(self, obs_dict: dict):
        # Expand the tree from a initial state
        root_node = SANode(state=obs_dict["state"], action=None, depth=0, value=None, children=[])
        self.nodes.append(root_node)
        self.frontiers.append(0)
        while len(self.frontiers) > 0:
            node_id = self.frontiers.pop(0)
            if self.nodes[node_id].depth < self.max_depth:
                self.expand_node(node_id)
        

    def expand_node(self, node_id: int):
        # Expand a node in the tree
        node = self.nodes[node_id]
        state = node.state
        # Repeat the state for k_sample times
        # repeat_state = np.repeat(state[-1:, ...], self.k_sample, axis=0)
        state = state[-1:, ...]
        pred = self.policy.predict_state_action({"state": state}, knn=self.k_sample)
        pred_states, pred_actions = pred["state"], pred["action"]
        # Convert to numpy
        pred_states = pred_states.cpu().numpy()
        pred_actions = pred_actions.cpu().numpy()
        # Overwrite the action of the node
        self.nodes[node_id] = SANode(state=node.state, action=pred_actions[:, :self.n_act_steps, :], depth=node.depth, value=None, children=node.children)
        for i in range(self.k_sample):
            new_node = SANode(state=pred_states[i, -self.n_obs_steps:, :], action=None, depth=node.depth + 1, value=None, children=[])
            self.nodes.append(new_node)
            self.nodes[node_id].children.append(len(self.nodes) - 1)
            self.frontiers.append(len(self.nodes) - 1)
        return True
    
    def export(self):
        # Export the state and action of the tree
        states = []
        actions = []
        depths = []
        childrens = []
        for node in self.nodes:
            states.append(node.state)
            actions.append(node.action)
            depths.append(node.depth)
            childrens.append(node.children)
        return states, actions, depths, childrens