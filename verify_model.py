import time
import torch
import torch.nn as nn
from models import DF_CMNet
from losses import CompoundForgeryLoss, compute_morphological_boundary

def verify_pipeline():
    print("=" * 60)
    print("          DF-CMNet Model Verification & Profiling")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # 1. Model Instantiation
    print("\n[1] Initializing DF-CMNet...")
    model = DF_CMNet(pretrained=False).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [+] Total Parameters:     {total_params / 1e6:.2f} M")
    print(f"  [+] Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # 2. Forward Pass Test
    B, C, H, W = 2, 3, 256, 256
    print(f"\n[2] Testing Forward Pass with dummy batch [{B}, {C}, {H}, {W}]...")
    dummy_input = torch.randn(B, C, H, W, device=device)
    dummy_mask = torch.randint(0, 2, (B, 1, H, W), device=device, dtype=torch.float32)

    outputs = model(dummy_input)
    assert 'mask_logits' in outputs, "Missing mask_logits in output"
    assert 'edge_logits' in outputs, "Missing edge_logits in output"
    assert 'mask_pred' in outputs, "Missing mask_pred in output"
    assert 'edge_pred' in outputs, "Missing edge_pred in output"

    print(f"  [+] mask_logits shape: {outputs['mask_logits'].shape} (Expected: [{B}, 1, {H}, {W}])")
    print(f"  [+] edge_logits shape: {outputs['edge_logits'].shape} (Expected: [{B}, 1, {H}, {W}])")
    print(f"  [+] mask_pred range:   [{outputs['mask_pred'].min():.3f}, {outputs['mask_pred'].max():.3f}]")
    print(f"  [+] edge_pred range:   [{outputs['edge_pred'].min():.3f}, {outputs['edge_pred'].max():.3f}]")

    # 3. Dynamic Boundary Generation Test
    print("\n[3] Testing Morphological Boundary Extraction...")
    edge_gt = compute_morphological_boundary(dummy_mask, kernel_size=7)
    print(f"  [+] edge_gt shape: {edge_gt.shape}, sum: {edge_gt.sum().item():.0f}")

    # 4. Backward Pass & Gradient Flow
    print("\n[4] Testing Loss Computation & Backward Pass...")
    criterion = CompoundForgeryLoss(lambda_foc=1.0, lambda_dice=1.0, lambda_bound=0.5)
    loss, loss_dict = criterion(outputs, dummy_mask)
    print(f"  [+] Total Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"      - {k}: {v:.4f}")

    model.zero_grad()
    loss.backward()

    # Verify gradients in both branches
    bayar_grad = model.bayar.weight.grad
    spatial_stem_grad = model.spatial_encoder.stem[0].weight.grad
    assert bayar_grad is not None, "Gradient failed to flow to Bayar Conv (Frequency stream)!"
    assert spatial_stem_grad is not None, "Gradient failed to flow to Spatial stem!"
    print("  [+] Gradients successfully flowed to BOTH Spatial and Frequency streams!")
    print(f"      - Spatial stem grad norm: {spatial_stem_grad.norm().item():.6f}")
    print(f"      - Bayar conv grad norm:   {bayar_grad.norm().item():.6f}")

    # 5. Latency & Throughput Benchmark
    print("\n[5] Measuring Inference Latency (Batch Size = 1)...")
    single_input = torch.randn(1, 3, 256, 256, device=device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(single_input)

    # Timing
    iterations = 30
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(single_input)
    total_time = time.time() - start
    avg_latency_ms = (total_time / iterations) * 1000.0
    fps = iterations / total_time
    print(f"  [+] Average Latency: {avg_latency_ms:.2f} ms")
    print(f"  [+] Throughput:      {fps:.2f} FPS")

    print("\n" + "=" * 60)
    print("     ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    verify_pipeline()
