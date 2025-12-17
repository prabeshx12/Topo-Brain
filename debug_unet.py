"""Debug script to trace U-Net feature dimensions."""
import torch
from models.generator_unet3d import UNet3DGenerator

# Create model
model = UNet3DGenerator(
    in_channels=1,
    out_channels=1,
    base_features=32,
    num_levels=4,
)

print("Model created with:")
print(f"  base_features={32}")
print(f"  num_levels={4}")
print()

# Check encoder blocks
print("Encoder blocks:")
for i, block in enumerate(model.encoder_blocks):
    print(f"  Encoder {i}: {block.conv1.block[0].in_channels} → {block.conv2.block[0].out_channels}")

print()
print(f"Bottleneck: {model.bottleneck[0].block[0].in_channels} → {model.bottleneck[0].block[0].out_channels}")
print()

# Check decoder blocks
print("Decoder blocks:")
for i, block in enumerate(model.decoder_blocks):
    up_in = block.upsample.in_channels
    conv_in = block.conv1.block[0].in_channels
    conv_out = block.conv1.block[0].out_channels
    skip_ch = conv_in - up_in
    print(f"  Decoder {i}: in={up_in}, skip={skip_ch}, concat={conv_in} → {conv_out}")

print()
print(f"Output conv: {model.output_conv.in_channels} → {model.output_conv.out_channels}")
print()

# Test forward
x = torch.randn(1, 1, 64, 64, 64)
print(f"Input: {x.shape}")

# Trace forward
x = model.init_conv(x)
print(f"After init_conv: {x.shape}")

skips = []
for i, encoder in enumerate(model.encoder_blocks):
    x, skip = encoder(x)
    skips.append(skip)
    print(f"After encoder {i}: x={x.shape}, skip={skip.shape}")

x = model.bottleneck(x)
print(f"After bottleneck: {x.shape}")

for i, (decoder, skip) in enumerate(zip(model.decoder_blocks, reversed(skips))):
    print(f"Decoder {i}: x={x.shape}, skip={skip.shape}")
    x = decoder(x, skip)
    print(f"  After decoder {i}: {x.shape}")

x = model.output_conv(x)
print(f"Output: {x.shape}")
