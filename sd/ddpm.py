import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class DDPMSampler:

  def __init__(self, generator = torch.Generator, num_training_steps = 1000, beta_start = 0.00085, beta_end = 0.0120):
    self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype = torch.float32) ** 2
    self.alphas = 1.0 - self.betas
    #need to find sum of all alphas from 0 to t, use cumprod
    self.alpha_cumprod = torch.cumprod(self.alphas, dim = 0)
    self.one = torch.tensor(1.0)

    self.generator = generator
    self.num_training_steps = num_training_steps
    #want to go from 1000 to 0 (final to start)
    self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

  def set_inference_timesteps(self, num_inference_steps = 50):
    self.num_inference_steps = num_inference_steps
    #starts from 999 , 998, ... to 0 (last number excluded)
    #but u want 999, 999-20, 999-40,... (1000/50 = 20)
    step_ratio = self.num_training_steps // self.num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) *step_ratio).round()[::-1].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps)

  def _get_previous_timestep(self, timestep):
    prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
    return prev_t

  def _get_variance(self, timestep):
    prev_t =  self._get_previous_timestep(timestep)

    alpha_prod_t = self.alpha_cumprod[timestep]
    alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    variance = torch.clamp(variance, min= 1e-20)

    return variance

  def set_strength(self, strength= 1):
    start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
    self.timesteps = self.timesteps[start_step:]
    self.start_step = start_step

  #modeloutput is predicted noise
  def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
    t = timestep
    prev_t = self._get_previous_timestep(t)

    alpha_prod_t = self.alpha_cumprod[timestep]
    alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    #compute x0 predicted image of original
    pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    #compute coefficient for pre_original_sample and current sample x_t
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
    current_sample_coeff = (alpha_prod_t ** 0.5 * beta_prod_t_prev) / beta_prod_t

    #compute predicted prev sample mean
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

    variance = 0
    if t > 0:
      device = model_output.device
      noise = torch.randn(model_output.shape, generator = self.generator, device = device, dtype = model_output.dtype)
      #same formula as bfr
      variance = (self._get_variance(t) ** 0.5) * noise

    pred_prev_sample = pred_prev_sample + variance
    return pred_prev_sample

  #noisyfying an image
  def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor):
    alpha_cumprod = self.alpha_cumprod.to(device = original_samples.device,dtype = original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
    self_alpha_prod = sqrt_alpha_prod.flatten()
    #need the shape to be the same so keep unsqueezing till its the same
    #its for broadcasting
    while len(self_alpha_prod) < len(original_samples.shape):
      self_alpha_prod = self_alpha_prod.unsqueeze(-1)

    #u sqrt bcus u want the stdev not the variance
    sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod) < len(original_samples.shape):
      sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noise = torch.randn(original_samples.shape, generator= self.generator, device = original_samples.device, dtype = original_samples.dtype)
    noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise
    return noisy_samples