import torch
import torch.nn.functional as F
import numpy as np

from itertools import cycle

def get_model_fn(model, train=False, marginal_flag=None):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """

        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma, marginal_flag=marginal_flag)

    return model_fn

def get_score_fn(model, train=False, sampling=False, marginal_flag=None):
    model_fn = get_model_fn(model, train=train, marginal_flag=marginal_flag)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        def score_fn(x, sigma):
            
            score = model_fn(x, sigma)
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn