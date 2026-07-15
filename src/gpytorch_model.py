"""
GPyTorch Multi-Task Sparse Variational Gaussian Process for Cryptocurrency Volatility

This module implements a Multi-Task SVGP that can work alongside or instead of
the LSTM model for volatility prediction with uncertainty quantification.

author: @fjaraavila
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    MultitaskVariationalStrategy,
    VariationalStrategy,
)

from src.mean_functions import LSTMMean_function


class LatentSVGP(ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        base_kernel: Kernel,
        device: torch.device = torch.device("cpu"),
        learn_inducing_locations: bool = True,
    ):
        """
        This is the inferential system that uses the features extracted from the lstm.
        In this specific case this is considered the hidden states

        Args:
            inducing_points (torch.Tensor): Embeddings that are considered hidden states
            base_kernel (Kernel): Usage Kernel. Preferred RBF or Matern52
            device (torch.device, optional): _description_. Defaults to torch.device('cpu').
            learn_inducing_locations (bool, optional): Whether inducing points are learnable parameters.
            if not True, provide good inducing points.
        """
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = ScaleKernel(base_kernel)
        self.to(device)

    def forward(self, z):
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_z, covar_z)

class DeepKernelSVGP(gpytorch.Module):
    def __init__(self, feature_extractor: nn.Module, inferential_model: LatentSVGP):
        super(DeepKernelSVGP, self).__init__()
        self.feature_extractor = feature_extractor
        self.inferential_process = inferential_model

    def forward(self, x_seq) -> gpytorch.distributions.MultivariateNormal:
        output = self.feature_extractor(x_seq)
        print(f"Features extracted by LSTM: {output.shape}")
        output = self.inferential_process(output)
        print(f"Output from SVGP: {output.mean.shape}, {output.covariance_matrix.shape}")
        return output


class Univariate_SVGP(ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_shape: tuple[int],
        hidden_size: list[int],
        dropout: list[float],
        output_dimension: int,
        device: torch.device = torch.device("cpu"),
        learn_inducing_locations: bool = True,
    ):

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = LSTMMean_function(
            input_shape, hidden_size, dropout, output_dimension, device
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
"""
class MultiTask_SVGP_same_size(ApproximateGP):
    
    def __init__(self, 
                 inducing_points: torch.Tensor,
                 num_tasks: int,
                 input_shape: tuple[int],
                 hidden_size: list[int],
                 dropout: list[float],
                 output_dimension: int,
                 device: torch.device = torch.device('cpu'),
                 learn_inducing_locations: bool = True,
                 pre_trained_lstm: Optional[nn.Module] = None
                 ):
        
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )
        multitask_variational_strategy = MultitaskVariationalStrategy(
            variational_strategy, num_tasks=num_tasks
        )
        super().__init__(multitask_variational_strategy)

        if pre_trained_lstm is not None:
            self.mean_module = pre_trained_lstm
        else:
            self.mean_module = LSTMMean_function(
                input_shape,
                hidden_size,
                dropout,
                output_dimension,
                device
            )
        self.covar_module = LSTMDeepKernel_same_size_layers(
            base_kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            ),
            lstm_kernel_parameters={
                "input_size": input_shape[0],
                "hidden_size": hidden_size[0],
                "num_layers": len(hidden_size),
                "output_size": output_dimension
            },
            device=device
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
"""