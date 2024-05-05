from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseSAPolicy(ModuleAttrMixin):
    # base state/action policy
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()
    
    def predict_state_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Predict the action and resulting futer state given the current state.
        obs_dict:
            str: B,To,*
        return: B,Ta,Da+Do
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
