"""Policy Sample Tree."""

import numpy as np
from diffusion_policy.policy.base_sa_policy import BaseSAPolicy
from collections import namedtuple

SANode = namedtuple("SANode", ["idx", "state", "action", "depth", "reward", "value", "children", "branch_idx"])


############################### Utility #################################
def extract_skeleton_from_tree(depths, childrens):
    # Compue all leaf nodes first
    leaf_nodes = []
    for i in range(len(childrens)):
        if len(childrens[i]) == 0:
            leaf_nodes.append(i)
    # Compute node parents
    parents = [-1] * len(childrens)
    for i in range(len(childrens)):
        for child in childrens[i]:
            parents[child] = i
    # Compute all tree skeleton by traversing back from leaf nodes
    tree_skeletons = []
    for leaf_node in leaf_nodes:
        node_id = leaf_node
        depth = depths[node_id]
        skeleton = []
        while depth >= 0:
            skeleton.append(node_id)
            node_id = parents[node_id]
            depth -= 1
        tree_skeletons.append(skeleton[::-1])
    return tree_skeletons


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

    def expand_tree(self, obs_dict: dict, reward_func=None):
        # Expand the tree from a initial state
        root_node = SANode(idx=0, state=obs_dict["state"], action=None, depth=0, reward=0, value=None, children=[], branch_idx=-1)
        self.nodes.append(root_node)
        self.frontiers.append(0)
        while len(self.frontiers) > 0:
            node_id = self.frontiers.pop(0)
            if self.nodes[node_id].depth < self.max_depth:
                self.expand_node(node_id, reward_func=reward_func)
        self.backtrack_value()

    def expand_node(self, node_id: int, reward_func=None):
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
        self.nodes[node_id] = SANode(
            idx=node_id, state=node.state, action=pred_actions[:, : self.n_act_steps, :], depth=node.depth, reward=node.reward, value=node.value, children=node.children, branch_idx=node.branch_idx
        )
        for i in range(self.k_sample):
            node_states = pred_states[i, -self.n_obs_steps :, :]
            reward = reward_func(node_states) if reward_func is not None else None
            new_node = SANode(idx=len(self.nodes), state=node_states, action=None, depth=node.depth + 1, reward=reward, value=None, children=[], branch_idx=i)
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
        values = []
        branch_idxs = []
        for node in self.nodes:
            states.append(node.state)
            actions.append(node.action)
            depths.append(node.depth)
            values.append(node.value)
            childrens.append(node.children)
            branch_idxs.append(node.branch_idx)
        skeletons = extract_skeleton_from_tree(depths, childrens)
        return states, actions, values, skeletons, branch_idxs

    def backtrack_value(self, gamma=0.99):
        """Backtrack the value of the nodes in the tree."""
        nodes_by_depth = {}
        rewards = []
        childrens = []
        for node in self.nodes:
            rewards.append(node.reward)
            childrens.append(node.children)
            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = []
            nodes_by_depth[node.depth].append(node.idx)

        depths = list(nodes_by_depth.keys())
        depths.sort(reverse=True)  # From the deepest to the shallowest
        for depth in depths:
            for node_id in nodes_by_depth[depth]:
                node = self.nodes[node_id]
                if len(node.children) == 0:
                    node_value = node.reward
                else:
                    children_value = 0
                    for child in node.children:
                        children_value += self.nodes[child].value if self.nodes[child].value is not None else 0
                    node_value = node.reward + gamma * children_value / len(node.children)
                # Overwrite the value of the node
                self.nodes[node_id] = SANode(
                    idx=node.idx, state=node.state, action=node.action, depth=node.depth, reward=node.reward, value=node_value, children=node.children, branch_idx=node.branch_idx
                )
