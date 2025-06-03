import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

try:
    from infosedd.model import utils as mutils
    from infosedd.utils import statistics_batch
    from infosedd.sampling import get_mutinfo_step_fn
    from infosedd import noise_lib
except:
    from model import utils as mutils
    from utils import statistics_batch
    from sampling import get_mutinfo_step_fn

torch.autograd.set_detect_anomaly(True)

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)

    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_loss_fn(configs, noise, graph, train, sampling_eps=1e-3):

    def loss_fn(model, batch):
        """
        Batch shape: [B, L] int. D given from graph
        """

        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
        sigma, dsigma = noise(t)

        if configs.is_parametric_marginal and configs.variant == "j":
            marginal_flag = np.random.randint(0, 3)
            if marginal_flag == 0:
                absorb_indices = configs.x_indices
            elif marginal_flag == 1:
                absorb_indices = configs.y_indices
            else:
                absorb_indices = None
        elif configs.is_parametric_marginal and configs.variant == "c":
            marginal_flag = np.random.randint(0, 2)
            absorb_indices = configs.x_indices
        else:
            absorb_indices = None
            marginal_flag = None
        
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])
        if configs.is_parametric_marginal and configs.variant == "j":
            if absorb_indices is not None:
                perturbed_batch[:, absorb_indices] = graph.dim - 1
        elif configs.is_parametric_marginal and configs.variant == "c":
            if marginal_flag == 0:
                perturbed_batch[:, absorb_indices] = batch[:, absorb_indices]
            elif marginal_flag == 1:
                perturbed_batch[:, absorb_indices] = graph.dim - 1
            else:
                raise ValueError(f"Invalid marginal_flag: {marginal_flag}")

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False, marginal_flag=marginal_flag)
        log_score = log_score_fn(perturbed_batch, sigma)

        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        if configs.is_parametric_marginal:
            if configs.variant == "j":
                if marginal_flag == 0:
                    loss = loss[:, configs.y_indices]
                elif marginal_flag == 1:
                    loss = loss[:, configs.x_indices]
            elif configs.variant == "c":
                loss = loss[:, configs.y_indices]

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss
    
    return loss_fn

def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer