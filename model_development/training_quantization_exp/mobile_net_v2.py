import os
import ssl
import time

import torch
from executorch import version
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

# Disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context


class BackendExporter:
    """Export models for different backends and configurations"""

    def __init__(self, output_dir="model_v2"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.example_inputs = (torch.randn(1, 3, 224, 224),)

        print("=" * 70)
        print(f"PyTorch version: {torch.__version__}")
        print(f"ExecuTorch version: {version.__version__}")
        print(f"Output directory: {self.output_dir}")
        print()

    def load_base_model(self):
        """Load fresh MobileNetV2 model"""
        print("📥 Loading MobileNetV2...")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model = model.eval()
        return model

    def export_baseline_portable(self):
        """Export 1: Portable (CPU baseline - no optimization)"""
        print("\n" + "=" * 70)
        print("1️⃣  PORTABLE (CPU Baseline - No Delegation)")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv2_portable_fp32.pte"

        start_time = time.time()
        exported = torch.export.export(model, self.example_inputs)
        executorch_program = to_edge_transform_and_lower(
            exported,
            partitioner=[]  # No delegation
        ).to_executorch()

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            executorch_program.write_to_file(f)

        export_time = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)

        print(f"✅ Exported: {filename}")
        print(f"   Export time: {export_time:.2f}s")
        print(f"   File size: {file_size:.2f} MB")

        return {
            'name': 'Portable',
            'backend': 'CPU (baseline)',
            'quantization': 'FP32',
            'filename': filename,
            'export_time': export_time,
            'file_size_mb': file_size
        }

    def export_xnnpack_fp32(self):
        """Export 2: XNNPACK FP32"""
        print("\n" + "=" * 70)
        print("2️⃣  XNNPACK (CPU Optimized) - FP32")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv2_xnnpack_fp32.pte"

        start_time = time.time()
        exported = torch.export.export(model, self.example_inputs)
        executorch_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            executorch_program.write_to_file(f)

        export_time = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)

        print(f"✅ Exported: {filename}")
        print(f"   Export time: {export_time:.2f}s")
        print(f"   File size: {file_size:.2f} MB")

        return {
            'name': 'XNNPACK',
            'backend': 'CPU (optimized)',
            'quantization': 'FP32',
            'filename': filename,
            'export_time': export_time,
            'file_size_mb': file_size
        }

    def export_xnnpack_int8_pt2e(self):
        """Export 3: XNNPACK with 8-bit quantization (PT2E)"""
        print("\n" + "=" * 70)
        print("3️⃣  XNNPACK + PT2E 8-bit Quantization")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv2_xnnpack_int8_pt2e.pte"

        start_time = time.time()

        # PT2E quantization flow
        print("   Applying PT2E quantization...")
        qparams = get_symmetric_quantization_config(is_per_channel=True)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(qparams)

        exported = torch.export.export(model, self.example_inputs)
        prepared_model = prepare_pt2e(exported.module(), quantizer)

        # Calibration (using random data here, it would be good with real data, for MLOps propose we think that's okay)
        print("   Calibrating...")
        for _ in range(10):
            prepared_model(torch.randn(1, 3, 224, 224))

        quantized_model = convert_pt2e(prepared_model)

        # Export quantized model
        print("   Exporting quantized model...")
        exported_quantized = torch.export.export(quantized_model, self.example_inputs)
        executorch_program = to_edge_transform_and_lower(
            exported_quantized,
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            executorch_program.write_to_file(f)

        export_time = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)

        print(f"✅ Exported: {filename}")
        print(f"   Export time: {export_time:.2f}s")
        print(f"   File size: {file_size:.2f} MB")

        return {
            'name': 'XNNPACK+INT8',
            'backend': 'CPU (optimized)',
            'quantization': 'INT8 (PT2E)',
            'filename': filename,
            'export_time': export_time,
            'file_size_mb': file_size
        }

    def export_vulkan_fp32(self):
        """Export 4: Vulkan GPU FP32"""
        print("\n" + "=" * 70)
        print("4️⃣  VULKAN (GPU) - FP32")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv2_vulkan_fp32.pte"

        start_time = time.time()
        exported = torch.export.export(model, self.example_inputs)
        executorch_program = to_edge_transform_and_lower(
            exported,
            partitioner=[VulkanPartitioner()]
        ).to_executorch()

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            executorch_program.write_to_file(f)

        export_time = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)

        print(f"✅ Exported: {filename}")
        print(f"   Export time: {export_time:.2f}s")
        print(f"   File size: {file_size:.2f} MB")

        return {
            'name': 'Vulkan',
            'backend': 'GPU',
            'quantization': 'FP32',
            'filename': filename,
            'export_time': export_time,
            'file_size_mb': file_size
        }

    def export_vulkan_fp16(self):
        """Export 5: Vulkan GPU FP16 """
        print("\n" + "=" * 70)
        print("5️⃣  VULKAN (GPU) - FP16")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv2_vulkan_fp16.pte"

        # Convert model to FP16
        print("   Converting model to FP16...")
        model = model.half()
        class FP16ModelWrapper(torch.nn.Module):
            """
            Wrapper that:
            - Accepts FP32 inputs (from Android)
            - Converts to FP16 internally
            - Runs FP16 computation (GPU accelerated)
            - Converts output back to FP32 (for Android)
            """
            def __init__(self, fp16_model):
                super().__init__()
                self.model = fp16_model
            
            def forward(self, x):
                # FP32 input → FP16 → FP16 compute → FP32 output
                return self.model(x.half()).float()
        wrapped_model = FP16ModelWrapper(model)
        
        start_time = time.time()
        # Export with FP32 inputs AND FP32 outputs
        exported = torch.export.export(wrapped_model, self.example_inputs)
        executorch_program = to_edge_transform_and_lower(
            exported,
            partitioner=[VulkanPartitioner()]
        ).to_executorch()

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            executorch_program.write_to_file(f)

        export_time = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)

        print(f"✅ Exported: {filename}")
        print(f"   Export time: {export_time:.2f}s")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   NOTE: FP32 I/O, FP16 internal compute")

        return {
            'name': 'Vulkan+FP16',
            'backend': 'GPU',
            'quantization': 'FP16',
            'filename': filename,
            'export_time': export_time,
            'file_size_mb': file_size
        }

    def run_all_exports(self):
        """Run all exports and generate summary"""
        results = []

        # Export all configurations
        results.append(self.export_baseline_portable())
        results.append(self.export_xnnpack_fp32())
        results.append(self.export_xnnpack_int8_pt2e())
        results.append(self.export_vulkan_fp32())
        results.append(self.export_vulkan_fp16())

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print export summary table"""
        print("\n" + "=" * 70)
        print("📊 EXPORT SUMMARY - MobileNetV2 Ready for S22 Ultra Testing")
        print("=" * 70)
        print(f"{'Configuration':<25} {'Backend':<20} {'Quant':<15} {'Size (MB)':<12} {'Time (s)'}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<25} {r['backend']:<20} {r['quantization']:<15} "
                  f"{r['file_size_mb']:<12.2f} {r['export_time']:.2f}")

        print("=" * 70)
        print(f"\n✅ All models exported to: {self.output_dir}/")

def main():
    """Main execution"""
    exporter = BackendExporter()
    results = exporter.run_all_exports()

    # Save results to JSON for reference
    import json
    with open(os.path.join(exporter.output_dir, "export_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("📄 Export summary saved to: export_summary.json\n")


if __name__ == "__main__":
    main()
