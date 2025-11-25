import os
import torch
import torch.nn as nn
import time
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def load_model(model_path="best_model.pth"):
    print("Loading model...")
    model = SimpleNN()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded\n")
    return model


def export_xnnpack(model, example_input, output_dir="executorch_models"):
    """Export with XNNPACK (CPU) - Works on all STM32"""
    print("Exporting XNNPACK (CPU)...")
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    exported = export(model, (example_input,))

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()]
    )

    exec_prog = edge.to_executorch()
    filepath = os.path.join(output_dir, "model_xnnpack_cpu.pte")

    with open(filepath, "wb") as f:
        exec_prog.write_to_file(f)

    elapsed = time.time() - start
    size_kb = os.path.getsize(filepath) / 1024

    print(f"✓ model_xnnpack_cpu.pte: {size_kb:.2f} KB ({elapsed:.2f}s)")
    return filepath


def export_arm_ethosu(model, example_input, output_dir="executorch_models"):
    """Export with ARM Ethos-U (NPU) - Optional, only on specialized ARM devices"""
    print("Checking ARM Ethos-U (NPU) support...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        from executorch.backends.arm.arm_partitioner import ArmPartitioner

        print("  Exporting ARM Ethos-U (NPU)...")
        start = time.time()

        exported = export(model, (example_input,))

        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[ArmPartitioner()]
        )

        exec_prog = edge.to_executorch()
        filepath = os.path.join(output_dir, "model_arm_ethosu_npu.pte")

        with open(filepath, "wb") as f:
            exec_prog.write_to_file(f)

        elapsed = time.time() - start
        size_kb = os.path.getsize(filepath) / 1024

        print(f"  ✓ model_arm_ethosu_npu.pte: {size_kb:.2f} KB ({elapsed:.2f}s)")
        return filepath
    except ImportError:
        print("  ℹ️  ARM Ethos-U not available (requires Cortex-M85 or STM32H7 with NPU)")
        print("     → Using XNNPACK CPU models instead (works on ALL STM32s)")
        return None
    except Exception as e:
        print(f"  ⚠️  ARM Ethos-U export failed: {e}")
        print("     → This is OK! Use XNNPACK models instead")
        return None


def export_xnnpack_int8(model, example_input, output_dir="executorch_models"):
    """Export with XNNPACK + INT8 quantization (CPU) - Smaller size"""
    print("Exporting XNNPACK INT8 (CPU + Quantized)...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # FIX: Import from torchao, not torch.ao
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config
        )

        start = time.time()

        # Export the model using standard export
        exported = export(model, (example_input,))

        # Prepare quantizer configuration
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)

        # Get the actual model from exported program
        m = exported.module()

        # Prepare the model for quantization
        m = prepare_pt2e(m, quantizer)
        m(example_input)

        # Convert to quantized model
        m = convert_pt2e(m)

        # Export the quantized model
        exported_quantized = export(m, (example_input,))

        # Lower to edge and compile
        edge = to_edge_transform_and_lower(
            exported_quantized,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackPartitioner()]
        )

        exec_prog = edge.to_executorch()
        filepath = os.path.join(output_dir, "model_xnnpack_int8.pte")

        with open(filepath, "wb") as f:
            exec_prog.write_to_file(f)

        elapsed = time.time() - start
        size_kb = os.path.getsize(filepath) / 1024

        print(f"✓ model_xnnpack_int8.pte: {size_kb:.2f} KB ({elapsed:.2f}s)")
        return filepath
    except ImportError as e:
        print(f"✗ Quantization library not available: {e}")
        print("  Install with: pip install torchao --upgrade")
        return None
    except Exception as e:
        print(f"✗ INT8 Quantization Error: {e}")
        return None


def main():
    print("=" * 60)
    print("ExecuTorch Export - CPU + ARM Backends")
    print("=" * 60)
    print()

    model = load_model("best_model.pth")
    example_input = torch.randn(1, 4)

    results = []

    # XNNPACK (CPU) - always try
    try:
        path = export_xnnpack(model, example_input)
        results.append(("XNNPACK CPU", path))
    except Exception as e:
        print(f"✗ XNNPACK failed: {e}\n")

    # XNNPACK INT8 (CPU + Quantized) - smaller
    try:
        path = export_xnnpack_int8(model, example_input)
        if path:
            results.append(("XNNPACK INT8", path))
    except Exception as e:
        print(f"✗ XNNPACK INT8 failed: {e}\n")

    # ARM Ethos-U (NPU) - only if available
    try:
        path = export_arm_ethosu(model, example_input)
        if path:
            results.append(("ARM Ethos-U", path))
    except Exception as e:
        print(f"✗ ARM Ethos-U failed: {e}\n")

    print("\n" + "=" * 60)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 60)
    print()

    if len(results) >= 2:
        print("✓ SUCCESS - You have 2 compatible models:")
        print()
        print("🏆 RECOMMENDED: XNNPACK INT8 (smallest)")
        print("   └─ Works on ALL STM32 devices")
        print("   └─ Best performance/size tradeoff")
        print()
        print("📦 ALTERNATIVE: XNNPACK CPU (fallback)")
        print("   └─ Works on ALL STM32 devices")
        print("   └─ Higher precision (FP32)")
        print()
        print("ℹ️  ARM Ethos-U: Requires Cortex-M85 (not available)")
        print()
    elif len(results) == 1:
        print("✓ XNNPACK CPU model available")
        print("   └─ Deploy to STM32: executorch_models/model_xnnpack_cpu.pte")
        print()

    print("=" * 60)

    for name, path in results:
        size = os.path.getsize(path) / 1024
        print(f"\n✓ {name}: {size:.2f} KB")
        print(f"  Location: {path}")


if __name__ == "__main__":
    main()