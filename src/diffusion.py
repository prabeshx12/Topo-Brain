"""
Diffusion Model Logic for Topo-Brain.
Implements Blueprint Section 7 (Training Strategy) and Q1-Level requirements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Warning: torchvision not found. Perceptual loss will be disabled/dummy.")

# Import advanced topology loss
try:
    from .topology_loss import create_topology_loss
    HAS_TOPOLOGY_LOSS = True
except ImportError:
    HAS_TOPOLOGY_LOSS = False
    print("Warning: Advanced topology loss not available. Using standard cross-entropy.")

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

class PerceptualLoss(nn.Module):
    """
    Simple VGG-based Perceptual Loss (Feature Matching).
    Extracts features from VGG16 (frozen) and computes MSE.
    """
    def __init__(self, max_loss=10.0):
        super().__init__()
        self.max_loss = max_loss  # Clamp to prevent explosion
        if not HAS_TORCHVISION:
            raise RuntimeError("PerceptualLoss requires 'torchvision' library. Please install it or set lambda_percep=0.")
            
        vgg = models.vgg16(pretrained=True)
        # Use first few layers for texture/structure
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
            
        # Input x, y are [B, 1, D, H, W] (3D)
        # VGG expects [B, 3, H, W] (2D RGB)
        # We process slice-by-slice or average over depth to save memory/compute
        # Or simple reshaping: treat Depth as Batch dimension for 2D VGG
        
        b, c, d, h, w = x.shape

        # Sample every 8th depth slice to avoid OOM (B*D=128 VGG passes is too many)
        stride = max(1, d // 8)
        x_2d = x[:, :, ::stride, :, :].permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        y_2d = y[:, :, ::stride, :, :].permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        
        # Convert 1 channel to 3 channels (repeat)
        x_2d = x_2d.repeat(1, 3, 1, 1)
        y_2d = y_2d.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet mean/std (approx)
        # Assuming input is [-1, 1], map to [0, 1] then normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        # Clamp to [0,1] — x_recon can be in [-2,2] from predict_start_from_noise
        x_2d = torch.clamp((x_2d + 1) * 0.5, 0.0, 1.0)
        y_2d = torch.clamp((y_2d + 1) * 0.5, 0.0, 1.0)
        
        x_2d = (x_2d - mean) / std
        y_2d = (y_2d - mean) / std
        
        # Extract features
        x_feat = self.feature_extractor(x_2d)
        y_feat = self.feature_extractor(y_2d)
        
        loss = F.mse_loss(x_feat, y_feat)
        
        return torch.clamp(loss, max=self.max_loss)

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
        topo_kwargs=None,       # R1.3: component flags for the topology loss
        seg_at_t0=True,         # A3: compute seg/topo loss on a t=0 forward (matches inference)
        normalize_pixel_loss=True,  # A2: divide out sqrt_recipm1(t) so the pixel term is SNR-flat
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.objective = objective
        # Audit fixes (see revision/AUDIT.md). Both default ON; set False to reproduce the
        # original (buggy) published behaviour for the ablation table.
        self.seg_at_t0 = seg_at_t0
        self.normalize_pixel_loss = normalize_pixel_loss
        # Empty/None == published behaviour. See topology_loss.create_topology_loss
        # for the R1.3 ablation variants (no-edge-weighting / no-boundary-dice /
        # single-scale).
        self.topo_kwargs = topo_kwargs or {}
        
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

        # Perceptual Loss
        # We instantiate lazily or here? Need to handle device placement.
        # Ideally, passed in or handled in training loop, but class encapsulation is nice.
        self.perceptual_loss = None # Initialize in training or verify device later
        
        # Advanced Topology Loss helper initialization
        self._topology_loss_module = None

    def get_perceptual_loss(self):
        if self.perceptual_loss is None:
            self.perceptual_loss = PerceptualLoss().to(self.betas.device)
        return self.perceptual_loss

    def _compute_topology_loss(self, seg_pred, seg_target, mask=None):
        """Compute topology loss with advanced edge-aware features.

        Component flags (reviewer R1.3 -- isolate the Sobel and Dice terms) are
        read from self.topo_kwargs, set at construction. Empty dict == published
        behaviour (edge-weighted CE + 0.5*boundary-Dice, multi-scale).
        """
        if self._topology_loss_module is None:
            if HAS_TOPOLOGY_LOSS:
                try:
                    from .topology_loss import create_topology_loss
                    tk = dict(getattr(self, "topo_kwargs", {}) or {})
                    use_ms = tk.pop("use_multiscale", True)
                    self._topology_loss_module = create_topology_loss(
                        num_classes=seg_pred.shape[1],
                        use_multiscale=use_ms,
                        **tk,
                    ).to(seg_pred.device)
                    print(f"✓ Activated Topology Loss (multiscale={use_ms}, flags={tk})")
                except Exception as e:
                    print(f"Warning: Could not init advanced topology loss: {e}")
                    self._topology_loss_module = "standard"
            else:
                self._topology_loss_module = "standard"
        
        if self._topology_loss_module != "standard":
            loss_dict = self._topology_loss_module(seg_pred, seg_target, mask=mask)
            return loss_dict['loss']
        else:
            # Fallback to class-weighted Cross-Entropy (dynamic weights based on num_classes)
            nc = seg_pred.shape[1]
            if nc == 2:
                weights = torch.tensor([0.5, 1.5], device=seg_pred.device)
            elif nc == 3:
                weights = torch.tensor([0.5, 2.0, 1.5], device=seg_pred.device)
            elif nc == 4:
                weights = torch.tensor([0.5, 2.0, 1.5, 1.0], device=seg_pred.device)
            else:
                weights = torch.ones(nc, device=seg_pred.device)
            loss_ce = F.cross_entropy(seg_pred, seg_target, weight=weights, reduction='none')
            if mask is not None:
                loss_ce = loss_ce * mask.view(-1, 1, 1, 1)
            return loss_ce.mean()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise, clip_denoised: bool = False):
        """Recover x0-hat from x_t and the predicted noise.

        Args:
            clip_denoised: if True, clamp x0-hat to the DATA range [-1, 1]. This is what
                Ho et al. (2020) require during SAMPLING. During TRAINING we keep the wider
                [-2, 2] so gradients still flow at the boundary.

        BUGFIX (B2): this method is shared by training and sampling, and previously always
        clamped to [-2, 2]. That is correct for training but WRONG for sampling: the data
        range is exactly [-1, 1], so every reverse step fed out-of-range mass into
        q_posterior, and it accumulated over ~185 steps before being hard-clipped once at
        the very end -- costing dynamic range and PSNR. Note ddim_sample already clipped to
        [-1, 1] correctly, while p_sample did not -- and evaluate_full_volume DEFAULTS to
        `--sampler ddpm`, i.e. the buggy path.
        """
        # Add epsilon for numerical stability in division
        sqrt_recip = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        # NOTE: with max_safe_timestep = len(betas)-1 these clamps are provably inert over
        # the sampled range (max sqrt_recip = 128 at t=198, and only t=199 is degenerate).
        # They are raised from 10.0 (which was also never reached) to a value that cannot
        # silently distort x0-hat near the top of the chain.
        sqrt_recip = torch.clamp(sqrt_recip, max=200.0)
        sqrt_recipm1 = torch.clamp(sqrt_recipm1, max=200.0)

        x_0 = sqrt_recip * x_t - sqrt_recipm1 * noise

        if clip_denoised:
            x_0 = torch.clamp(x_0, min=-1.0, max=1.0)   # sampling: true data range
        else:
            x_0 = torch.clamp(x_0, min=-2.0, max=2.0)   # training: margin for gradient flow

        return x_0

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
        out = self.model(x, t, conditioning)
        model_output = out['prediction']
        seg_output = out.get('segmentation')
        
        if self.objective == 'pred_noise':
            # B2: clip x0-hat to the data range [-1,1] during sampling (Ho et al. 2020).
            x_start = self.predict_start_from_noise(x, t, model_output, clip_denoised=True)
        else:
            x_start = torch.clamp(model_output, -1.0, 1.0)

        model_mean, model_variance = self.q_posterior(x_start, x, t)
        
        if t_index == 0:
            return model_mean, seg_output
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(extract(self.posterior_variance, t, x.shape)) * noise, seg_output

    @torch.no_grad()
    def p_sample_loop(self, conditioning, shape, return_all=False):
        device = self.betas.device
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # B1: must MATCH the training range (len(betas)-1). Previously len(betas)-15, which
        # started the reverse chain at t=184 where alphas_cumprod=0.0136 -- i.e. the model was
        # always trained to expect ~12% signal there, but we hand it PURE noise (0% signal).
        # That off-manifold start biased the first several reverse steps toward the mean.
        max_safe_timestep = len(self.betas) - 1
        final_seg = None
        for i in reversed(range(0, max_safe_timestep)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img, final_seg = self.p_sample(img, t, conditioning, i)

        # Clamp final output to training data range [-1, 1].
        # predict_start_from_noise allows [-2, 2] for gradient flow during
        # training, but the final sample must respect the data range.
        # Without this, out-of-range voxels cause large MSE → low PSNR.
        img = torch.clamp(img, -1.0, 1.0)

        if return_all:
            return img, final_seg
        return img

    @torch.no_grad()
    def ddim_sample(self, conditioning, shape, ddim_steps=50, eta=0.0):
        """
        DDIM sampling — deterministic (eta=0) or semi-stochastic (eta>0).
        Faster and typically higher PSNR than full DDPM chain.
        """
        device = self.betas.device
        b = shape[0]
        # B1: match the training range (len(betas)-1); see p_sample_loop.
        max_safe = len(self.betas) - 1

        # Uniform subsequence of timesteps
        times = np.linspace(0, max_safe - 1, ddim_steps, dtype=int)
        times = list(reversed(times))

        img = torch.randn(shape, device=device)
        final_seg = None

        alphas_cumprod = self.alphas_cumprod

        for i, t_cur in enumerate(times):
            t = torch.full((b,), t_cur, device=device, dtype=torch.long)

            out = self.model(img, t, conditioning)
            model_out = out['prediction']
            final_seg = out.get('segmentation')

            alpha_t = extract(alphas_cumprod, t, img.shape)

            # Recover x0 and eps depending on training objective
            if self.objective == 'pred_noise':
                eps_pred = model_out
                x0_pred = (img - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            else:  # pred_x0
                x0_pred = model_out
                eps_pred = (img - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t)

            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i < len(times) - 1:
                t_next = times[i + 1]
                alpha_next = alphas_cumprod[t_next]
                sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
                dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * eps_pred
                noise = torch.randn_like(img) if eta > 0 else 0
                img = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * noise
            else:
                img = x0_pred

        img = torch.clamp(img, -1.0, 1.0)
        return img, final_seg
        
    def forward(self, x_start, conditioning, seg_target=None, 
                lambda_pixel=1.0, lambda_percep=0.1, lambda_topo=0.1):
        """
        Compute total loss.
        """
        b, c, d, h, w = x_start.shape
        
        # BUGFIX (B1): the old code excluded the last 15 timesteps, claiming the cosine
        # schedule "produces coefficients > 4000x at t > 185". That premise is FALSE.
        # Verified numerically for T=200 cosine:
        #     max sqrt_recip   over t in [0,184] = 8.575   (the clamp is 10.0 -> never fires)
        #     the 4058x blow-up occurs at t=199 ONLY; t=198 is already fine.
        # It was a 15x overcorrection for a 1-step problem, and it had a real cost: it left
        # alphas_cumprod[184]=0.0136 (sqrt=0.117), so at the top of the chain the model always
        # saw ~12% signal -- while the sampler starts from PURE noise (0% signal). That is an
        # off-manifold initialisation. Dropping only the single degenerate step t=T-1 restores
        # a valid pure-noise start (alphas_cumprod[198] = 6.07e-05).
        max_safe_timestep = len(self.betas) - 1
        t = torch.randint(0, max_safe_timestep, (b,), device=x_start.device).long()

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Model forward
        outputs = self.model(x_noisy, t, conditioning)
        noise_pred = outputs['prediction']
        seg_pred = outputs['segmentation']

        # BUGFIX (A3): the segmentation head is trained here on x_noisy at a UNIFORM RANDOM t,
        # but at inference the segmentation is only ever read at t=0 (from a near-clean image).
        # Only 0.54% of training steps land at t=0 and only ~21% have alpha_bar > 0.9, so the
        # head received a ~5x weaker signal in the ONLY regime it is evaluated in -- and the
        # topology loss inherited the same dilution. (The regression baseline's head sees the
        # test-time distribution on 100% of steps, which is why it beat us on Dice AND topology.)
        # Fix: compute the seg/topology loss on a dedicated t=0 forward pass, matching inference.
        if seg_target is not None and lambda_topo > 0 and self.seg_at_t0:
            t0 = torch.zeros_like(t)
            seg_pred = self.model(x_start, t0, conditioning)['segmentation']
        
        # Clamp noise predictions to prevent numerical instability
        # Noise should theoretically be N(0,1), so clip to [-10, 10] for safety
        # (wider range to preserve gradient flow for extreme predictions)
        noise_pred = torch.clamp(noise_pred, min=-10.0, max=10.0)
        
        # 1. Diffusion Loss (MSE on noise) - ALWAYS ACTIVE
        # Force float32 for mean accumulation to prevent AMP overflow (max 65,504)
        # on 64x64x64 patches (262,144 voxels).
        if self.loss_type == 'l1':
            loss_diff = F.l1_loss(noise_pred.float(), noise.float())
        else:
            loss_diff = F.mse_loss(noise_pred.float(), noise.float())
            
        # 2. Auxiliary L1 Loss (on predicted Img)
        x_recon = self.predict_start_from_noise(x_noisy, t, noise_pred)

        # Calculate pixel loss in float32 for stability
        loss_pixel = F.l1_loss(x_recon.float(), x_start.float())

        # BUGFIX (A2): this term is NOT an independent loss. Since
        #     x_recon - x_start = -sqrt_recipm1(t) * (noise_pred - noise)   (exactly),
        # we have  loss_pixel(t) == sqrt_recipm1(t) * L1(noise_pred, noise),
        # i.e. loss_pixel is loss_diff scaled by a t-dependent factor. Effective weight:
        #     t=0 -> 1.02   t=100 -> 2.03   t=184 -> 9.52
        # 86% of the weight mass lands on the noisy half of the chain. That is the EXACT
        # INVERSE of Min-SNR / P2 weighting: the model trains ~9x harder on the pure-noise
        # regime (global structure) than on the low-noise regime -- which is precisely where
        # PSNR/SSIM live. The regression baseline has no such term, and beat us on image
        # quality. Fix: divide out sqrt_recipm1(t) so the pixel term is an unbiased
        # (SNR-flat) image-space loss rather than a reweighting of loss_diff.
        if self.normalize_pixel_loss:
            snr_scale = extract(self.sqrt_recipm1_alphas_cumprod, t, x_start.shape)
            per_voxel = (x_recon.float() - x_start.float()).abs() / snr_scale.clamp(min=1e-3)
            loss_pixel = per_voxel.mean()

        # Cap extreme loss spikes to prevent gradient explosion
        loss_pixel = torch.clamp(loss_pixel, max=5.0)

        
        # 3. Perceptual Loss (VGG)
        if lambda_percep > 0:
            loss_vgg = self.get_perceptual_loss()(x_recon.float(), x_start.float())
        else:
            loss_vgg = torch.tensor(0.0, device=x_start.device)
            
        # 4. Topology Loss (Segmentation)
        if seg_target is not None and lambda_topo > 0:
            # Force strict FP32 for topology math to avoid AMP overflow in edge ops.
            with torch.autocast(device_type=seg_pred.device.type, enabled=False):
                loss_topo = self._compute_topology_loss(
                    seg_pred.float(),
                    seg_target.long(),
                    mask=None,
                )
        else:
            loss_topo = torch.tensor(0.0, device=x_start.device)
        
        # NaN Guard: Replace any NaN loss with 0.0 to prevent poisoning
        # This is a safety net — the root cause (AMP overflow) is fixed in topology_loss.py
        def _safe(loss, name=""):
            # Handle scalar or tensor losses robustly under AMP.
            if not torch.isfinite(loss).all():
                print(f"WARNING: {name} loss is non-finite (value={loss.item():.6f}), zeroing gradient for this step")
                safe = torch.zeros((), device=loss.device, dtype=loss.dtype)
                safe.requires_grad_(True)
                return safe
            return loss
        
        loss_diff = _safe(loss_diff, "diff")
        loss_pixel = _safe(loss_pixel, "pixel")
        loss_vgg = _safe(loss_vgg, "percep")
        loss_topo = _safe(loss_topo, "topo")
            
        # Total Weighted Loss
        loss_total = loss_diff + lambda_pixel * loss_pixel + lambda_percep * loss_vgg + lambda_topo * loss_topo
        
        return {
            "loss": loss_total,
            "loss_diff": loss_diff,
            "loss_pixel": loss_pixel,
            "loss_vgg": loss_vgg,
            "loss_topo": loss_topo
        }
