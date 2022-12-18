import string
from typing import Any, Dict, Optional, Tuple, Type
import random

import numpy as np
from cmath import inf
import gym
import torch
from torch import long, nn,Tensor,tensor, bool
from torch.nn import BatchNorm1d, BatchNorm2d,AvgPool1d, Conv1d, Conv2d
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, BasePolicy,BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution, 
)


class CNNPolicy(BasePolicy):
    def __init__(self, 
                observation_space: gym.spaces.Dict,
                action_space: gym.spaces.Discrete,
                lr_schedule: Schedule = Schedule,
                log_std_init: float = 0.0,
                use_sde: bool = False,
                squash_output: bool = False,
                ortho_init: bool = True,
                features_dim = 236,
                action_dim = 150,
                features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                policy_kwargs: Optional[Dict[str, Any]] = None,
                ):
        super(CNNPolicy, self).__init__(observation_space,
                                            action_space,
                                            features_extractor_kwargs,
                                            # features_dim,
                                            optimizer_class = optimizer_class,
                                            optimizer_kwargs = optimizer_kwargs,   
                                            squash_output = squash_output                                         
                                            )
        self.features_extractor = CNNFeatureExtractor(observation_space, policy_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space,use_sde=use_sde)
        self.log_std_init = log_std_init
        self.action_dim = action_dim

    def _predict(self, observation: Tensor, deterministic: bool = True) -> Tensor:
            actions, values, log_prob = self.forward(observation, deterministic=deterministic)
            if isinstance(actions, Tensor):
                return actions
            return tensor([actions])

    def _build(self):
        pass

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            distribution, values = self.get_distribution(obs)
            log_prob = distribution.log_prob(actions)

            return values, log_prob, distribution.entropy()

    def forward(self, obs, deterministic = False):

        distribution,values = self.get_distribution(obs)
        # print('distribution',distribution)
        # print('values',values)
        # entropy = random.random()
        # if entropy < 0.05:
        #     deterministic = False
        # else:
        #     deterministic = True
        deterministic = True
        actions = distribution.get_actions(deterministic=deterministic)
        # print('actions',actions)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def predict_values(self,obs):
        _, values = self.get_distribution(obs)
        return values

    def get_distribution(self, obs):

        latent_sde, values, mean_actions = self.extract_features(obs)

        # print('value',values)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.action_dim, log_std_init=self.log_std_init
            )
        
        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std.to(device="cuda"))
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        return distribution, values

class CNNFeatureExtractor(nn.Module):
    def __init__(self, observation_space, policy_kwargs): #This custom GNN receives the obs dict for the action log-probabilities
        super(CNNFeatureExtractor, self).__init__()

        if not policy_kwargs:
            import yaml
            with open('policy_kwargs.yaml', 'r') as file:
                policy_kwargs = yaml.safe_load(file)
                policy_kwargs = policy_kwargs["policy_kwargs"]

        activations = {"relu" : nn.ReLU(), "elu" : nn.ELU(), "leaky" : nn.LeakyReLU(), "tanh" : nn.Tanh()}
        print(policy_kwargs)
        cnn_activ = activations[policy_kwargs["policy"]["cnn"]["activ"]]
        value_activ = activations[policy_kwargs["value"]["activ"]]
        policy_activ = activations[policy_kwargs["policy"]["activ"]]
        feature_activ = activations[policy_kwargs["policy"]["features"]["activ"]]
        
        # linear params
        value_hidden = policy_kwargs["value"]["hidden"]

        #Feature params
        feature_input = self.feature_size = np.prod(observation_space["features"].shape)
        feature_hidden = policy_kwargs["policy"]["features"]["hidden"]
        feature_output = policy_kwargs["policy"]["features"]["output"]

        # CNN params
        input_channels = observation_space["matrix"].shape[0]
        output_channels = policy_kwargs["policy"]["cnn"]["output_channels"]
        kernel_size = policy_kwargs["policy"]["cnn"]["conv_kernel"]
        stride = policy_kwargs["policy"]["cnn"]["conv_stride"]

        pool_kernel = policy_kwargs["policy"]["cnn"]["pool_kernel"]
        pool_stride = policy_kwargs["policy"]["cnn"]["pool_stride"]
        cnn_output = policy_kwargs["policy"]["cnn"]["output"]

        # Embedded network params
        input_dim = cnn_output + feature_output #length of feature vector for MLP
        latent_pi_dim = 150 # Output action dimensions
        latent_vf_dim = 1 # Output value dimensions

        #Biases
        feature_bias = False
        policy_bias = False
        value_bias = False

        self.feature_network = nn.Sequential(
            nn.Linear(feature_input, feature_hidden, bias=feature_bias),
            feature_activ,
            nn.Linear(feature_hidden, feature_hidden, bias=feature_bias),
            feature_activ,
            nn.Linear(feature_hidden, feature_output, bias=feature_bias),
            feature_activ,
        )

        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            cnn_activ,
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.network(
                torch.as_tensor(observation_space["matrix"].sample()[None]).float()
            ).shape[1] 
 
        self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_output, bias=policy_bias), policy_activ)
        self.value_network = nn.Sequential(nn.Linear(input_dim, value_hidden, bias=value_bias), 
                                            value_activ, 
                                            nn.Linear(value_hidden, value_hidden, bias=value_bias),
                                            value_activ,
                                            nn.Linear(value_hidden, latent_vf_dim),
                                            )
        self.policy_network = nn.Sequential(nn.Linear(input_dim, latent_pi_dim, bias=policy_bias), nn.LogSoftmax(dim=-1))

    def forward(self, data):
        # print('data from PPO', data)
        matrix = data["matrix"].float().cuda()
        features = data["features"].float().cuda()
        # mask = data['mask']
        

        D_old_embed = self.feature_network(features.reshape(-1, self.feature_size))

        cnn_embed = self.linear(self.network(matrix))

        final_embedding = torch.concat((D_old_embed, cnn_embed), dim=1)

        value = self.value_network(final_embedding)

        # output = self.mask_log_softmax(self.policy_network(new_obs), mask)
        output = self.policy_network(final_embedding)

        return final_embedding, value, output

    def mask_log_softmax(self, vector, mask):
        """
        from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303:
        ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        In the case that the input vector is completely masked, the return value of this function is
        arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
        of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
        that we deal with this case relies on having single-precision floats; mixing half-precision
        floats with fully-masked vectors will likely give you ``nans``.
        If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
        lower), the way we handle masking here could mess you up.  But if you've got logit values that
        extreme, you've got bigger problems than this.
        """
        if mask is not None:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
            # results in nans when the whole vector is masked.  We need a very small value instead of a
            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
            # becomes 0 - this is just the smallest value we can actually use.

            vector = vector + (mask + 1e-45).log()

        return torch.nn.functional.log_softmax(vector, dim=-1)
