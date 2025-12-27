"""
3D GAN models for 3T → 7T MRI super-resolution.
"""
from .generator_unet3d import UNet3DGenerator
from .discriminator_patchgan3d import PatchGANDiscriminator3D

__all__ = [
    'UNet3DGenerator',
    'PatchGANDiscriminator3D',
]
