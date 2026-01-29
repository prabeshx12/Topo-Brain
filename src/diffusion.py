"""
Diffusion Model Logic for Topo-Brain.
Implements Blueprint Section 7 (Training Strategy) and Q1-Level requirements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.losses import PerceptualLoss, TopologyLoss

# Helper to extract values at specific timesteps
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    t = t.to(a.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process with multi-task support.
    """
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l1",
        objective="pred_noise", # or pred_x0
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.objective = objective
        
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")
            
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # register buffer helper function that casts float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Losses (Lazy Init or Explicit)
        self.perceptual_loss = None 
        self.topology_loss = TopologyLoss() # Logic is stateless/simple instance

    def get_perceptual_loss(self):
        if self.perceptual_loss is None:
            self.perceptual_loss = PerceptualLoss().to(self.betas.device)
        return self.perceptual_loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, t, conditioning, t_index):
        # conditioning: 3T input
        model_output = self.model(x, t, conditioning)['prediction']
        
        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t, model_output)
        else:
            x_start = model_output
            
        model_mean, model_variance = self.q_posterior(x_start, x, t)
        
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(extract(self.posterior_variance, t, x.shape)) * noise

    @torch.no_grad()
    def p_sample_loop(self, conditioning, shape):
        device = self.betas.device
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        imgs = []
        for i in reversed(range(0, len(self.betas))):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, conditioning, i)
            # imgs.append(img.cpu()) # Optional: save intermediates
            
        return img
        
    def forward(self, x_start, conditioning, seg_target=None, 
                lambda_pixel=1.0, lambda_percep=0.1, lambda_topo=0.1):
        """
        Compute total loss.
        """
        b, c, d, h, w = x_start.shape
        t = torch.randint(0, len(self.betas), (b,), device=x_start.device).long()
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Model forward
        outputs = self.model(x_noisy, t, conditioning)
        noise_pred = outputs['prediction']
        seg_pred = outputs['segmentation']
        
        # 1. Diffusion Loss (MSE on noise)
        if self.loss_type == 'l1':
            loss_diff = F.l1_loss(noise_pred, noise)
        else:
            loss_diff = F.mse_loss(noise_pred, noise)
            
        # 2. Auxiliary L1 Loss (on predicted Img)
        # We need to predict x_0 first
        x_recon = self.predict_start_from_noise(x_noisy, t, noise_pred)
        loss_pixel = F.l1_loss(x_recon, x_start)
        
        # 3. Perceptual Loss (VGG)
        if lambda_percep > 0:
            loss_vgg = self.get_perceptual_loss()(x_recon, x_start)
        else:
            loss_vgg = torch.tensor(0.0, device=x_start.device)
            
        # 4. Topology Loss (Segmentation)
        if seg_target is not None and lambda_topo > 0:
            loss_topo = self.topology_loss(seg_pred, seg_target)
        else:
            loss_topo = torch.tensor(0.0, device=x_start.device)
            
        # Total Weighted Loss
        loss_total = loss_diff + lambda_pixel * loss_pixel + lambda_percep * loss_vgg + lambda_topo * loss_topo
        
        return {
            "loss": loss_total,
            "loss_diff": loss_diff,
            "loss_pixel": loss_pixel,
            "loss_vgg": loss_vgg,
            "loss_topo": loss_topo
        }
