"""
Quick integration script for advanced topology loss.
Run this to automatically update diffusion.py with the edge-aware loss.
"""
import re
from pathlib import Path

def integrate_advanced_topology_loss():
    """Integrate the advanced topology loss into diffusion.py."""
    
    diffusion_path = Path("src/diffusion.py")
    
    if not diffusion_path.exists():
        print(f"❌ Error: {diffusion_path} not found!")
        return False
    
    # Read current content
    with open(diffusion_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already integrated
    if "self._topology_loss" in content:
        print("✓ Advanced topology loss already integrated!")
        return True
    
    # Find the topology loss section
    pattern = r"(# 4\. Topology Loss.*?\n\s+if seg_target is not None and lambda_topo > 0:.*?\n\s+loss_topo = )F\.cross_entropy\(seg_pred, seg_target\)"
    
    replacement = r"""\1self._compute_topology_loss(seg_pred, seg_target)"""
    
    # Add the helper method
    helper_method = '''
    def _compute_topology_loss(self, seg_pred, seg_target):
        """Compute topology loss with optional advanced features."""
        # Initialize advanced topology loss if available
        if not hasattr(self, '_topology_loss'):
            self._topology_loss = None
            if HAS_TOPOLOGY_LOSS:
                try:
                    self._topology_loss = create_topology_loss(
                        num_classes=seg_pred.shape[1],
                        use_multiscale=True
                    ).to(seg_pred.device)
                    print("✓ Using advanced edge-aware topology loss")
                except Exception as e:
                    print(f"⚠ Could not initialize advanced topology loss: {e}")
                    print("  Falling back to standard cross-entropy")
        
        # Use advanced loss if available, otherwise standard CE
        if self._topology_loss is not None:
            loss_dict = self._topology_loss(seg_pred, seg_target)
            return loss_dict['loss']
        else:
            return F.cross_entropy(seg_pred, seg_target)
'''
    
    # Apply replacement
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        print("❌ Could not find topology loss section to replace")
        print("   Manual integration required - see IMPLEMENTATION_SUMMARY.md")
        return False
    
    # Insert helper method before the forward method
    forward_pattern = r"(    def forward\(self, x_start, conditioning)"
    new_content = re.sub(forward_pattern, helper_method + r"\n\1", new_content)
    
    # Write back
    with open(diffusion_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✓ Successfully integrated advanced topology loss!")
    print("  - Edge-aware boundary weighting (2x)")
    print("  - Multi-scale consistency (3 scales)")
    print("  - Automatic fallback to standard CE if unavailable")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Topology Loss Integration")
    print("=" * 60)
    print()
    
    success = integrate_advanced_topology_loss()
    
    print()
    if success:
        print("✅ Integration complete!")
        print()
        print("Next steps:")
        print("1. Commit and push changes")
        print("2. Pull on CERNbox")
        print("3. Resume training from 100k checkpoint")
        print()
        print("The advanced loss will automatically activate when topology")
        print("loss weight (lambda_topo) becomes non-zero at step 100k+")
    else:
        print("⚠ Integration failed - manual steps required")
        print("See IMPLEMENTATION_SUMMARY.md Section A for manual integration")
    
    print("=" * 60)
