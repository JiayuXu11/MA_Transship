import torch
import torch.nn as nn
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP


class DeterministicMechanismPolicy(nn.Module):
    """Deterministic policy network for continuous action space.(Tailored for Mechanism)"""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args["num_agents"]
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape, hidden_sizes[0], activation_func
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]
        act_dim = action_space.shape[0]
        pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]
        self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(**self.tpdv)
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
        self.to(device)

    def forward(self, obs, **kwargs):
        # Return output from network scaled to action space limits.
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        outputs = self.pi(x)
        
        action_inputs = obs[:, :self.num_agents]
        pos_indices = action_inputs > 0
        neg_indices = action_inputs < 0
        pos_greater = action_inputs.sum(dim=1) > 0

        # 初始化 fulfill_ratio
        fulfill_ratio = torch.zeros_like(outputs)

        # 创建一个掩码，标记正负值同时存在的样本
        valid_mask = (pos_indices.sum(dim=1) > 0) & (neg_indices.sum(dim=1) > 0)

        # 对于正的情况
        pos_mask = valid_mask & pos_greater
        pos_outputs = outputs.clone()
        pos_outputs[~pos_indices] = float('-inf')  # 非正的位置标记为-inf，从而在softmax中被忽略
        pos_softmax = torch.softmax(pos_outputs, dim=1)
        fulfill_ratio[pos_mask] = pos_softmax[pos_mask]

        # 对于负的情况
        neg_mask = valid_mask & ~pos_greater
        neg_outputs = outputs.clone()
        neg_outputs[~neg_indices] = float('inf')  # 非负的位置标记为inf，从而在softmax中被忽略
        neg_softmax = torch.softmax(-neg_outputs, dim=1)
        fulfill_ratio[neg_mask] = neg_softmax[neg_mask]

        return fulfill_ratio
