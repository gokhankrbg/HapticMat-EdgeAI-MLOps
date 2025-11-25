import os
import ssl
import time
import json

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
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

# Disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context


class BackendExporter:
    """Export models for different backends and configurations"""

    def __init__(self, output_dir="models_mv3"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Standard input size for MobileNet models
        self.example_inputs = (torch.randn(1, 3, 224, 224),)

        print(f"PyTorch version: {torch.__version__}")
        print(f"ExecuTorch version: {version.__version__}")
        print(f"Output directory: {self.output_dir}")
        print()

    def load_base_model(self):
        """Load fresh MobileNetV3 Large model"""
        print("📥 Loading MobileNetV3 Large...")
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model = model.eval()
        return model

    def export_baseline_portable(self):
        """Export 1: Portable (CPU baseline - no optimization)"""
        print("\n" + "=" * 70)
        print("1️⃣  PORTABLE (CPU Baseline - No Delegation)")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv3_portable_fp32.pte"  # Updated filename

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
        filename = "mv3_xnnpack_fp32.pte"  # Updated filename

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
        filename = "mv3_xnnpack_int8_pt2e.pte"  # Updated filename

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
        filename = "mv3_vulkan_fp32.pte"  # Updated filename

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
        """Export 5: Vulkan GPU FP16"""
        print("\n" + "=" * 70)
        print("5️⃣  VULKAN (GPU) - FP16")
        print("=" * 70)

        model = self.load_base_model()
        filename = "mv3_vulkan_fp16.pte"  # Updated filename

        # Convert model to FP16
        print("   Converting model to FP16...")
        model = model.half()

        # But I/O stays FP32 for Android compatibility
        print("   Creating FP32 I/O wrapper (FP16 compute internally)...")

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
        # Export with FP32 inputs AND FP32 outputs (what Android needs)
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

    def export_qnn_snapdragon(self):
        """Export 6: QNN (Qualcomm Snapdragon NPU) - FP32"""
        print("\n" + "=" * 70)
        print("6️⃣  QNN (Qualcomm Snapdragon NPU) - FP32")
        print("=" * 70)

        try:
            from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

            model = self.load_base_model()
            filename = "mv3_qnn_fp32.pte"  # Updated filename

            start_time = time.time()
            exported = torch.export.export(model, self.example_inputs)
            executorch_program = to_edge_transform_and_lower(
                exported,
                partitioner=[QnnPartitioner()]
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
                'name': 'QNN-FP32',
                'backend': 'NPU (Snapdragon)',
                'quantization': 'FP32',
                'filename': filename,
                'export_time': export_time,
                'file_size_mb': file_size
            }
        except ImportError as e:
            print(f"⚠️  QNN SDK not installed - Skipping: {e}")
            print("   Install: Follow Qualcomm AI Engine Direct SDK setup")
            return None

    def export_qnn_int8(self):
        """Export 7: QNN with INT8 Quantization"""
        print("\n" + "=" * 70)
        print("7️⃣  QNN (Qualcomm Snapdragon NPU) - INT8 Quantized")
        print("=" * 70)

        try:
            from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
            from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
            from executorch.backends.qualcomm.quantizer.custom_annotation import (
                get_default_16a8w_qnn_ptq_config
            )

            model = self.load_base_model()
            filename = "mv3_qnn_int8.pte"  # Updated filename

            start_time = time.time()

            # Step 1: Export the model
            print("   Exporting model...")
            exported = torch.export.export(model, self.example_inputs)

            # Step 2: Quantization
            print("   Applying QNN INT8 quantization...")
            quantizer = QnnQuantizer()

            # Use 16-bit activation, 8-bit weight quantization (recommended for QNN)
            quant_config = get_default_16a8w_qnn_ptq_config()
            quantizer.set_per_channel_weight_dtype(quant_config)

            # Prepare for quantization
            prepared_model = prepare_pt2e(exported.module(), quantizer)

            # Calibration (using random data here, it would be good with real data, for MLOps propose we think that's okay)
            print("   Calibrating with sample data...")
            for _ in range(100):  # More calibration samples for better accuracy
                prepared_model(torch.randn(1, 3, 224, 224))

            # Convert to quantized model
            quantized_model = convert_pt2e(prepared_model)

            # Step 3: Export with QNN partitioner
            print("   Lowering to QNN backend...")
            exported_quantized = torch.export.export(quantized_model, self.example_inputs)
            executorch_program = to_edge_transform_and_lower(
                exported_quantized,
                partitioner=[QnnPartitioner()]
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
                'name': 'QNN-INT8',
                'backend': 'NPU (Snapdragon)',
                'quantization': 'INT8 (16a8w)',
                'filename': filename,
                'export_time': export_time,
                'file_size_mb': file_size
            }
        except ImportError as e:
            print(f"⚠️  QNN SDK not installed - Skipping: {e}")
            print("   Install: Follow Qualcomm AI Engine Direct SDK setup")
            return None
        except Exception as e:
            print(f"❌ Error during QNN INT8 export: {e}")
            return None

    def export_qnn_fp16(self):
        """Export 8: QNN with FP16"""
        print("\n" + "=" * 70)
        print("8️⃣  QNN (Qualcomm Snapdragon NPU)")
        print("=" * 70)

        try:
            from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

            model = self.load_base_model()
            filename = "mv3_qnn_fp16.pte"  # Updated filename

            # Convert model to FP16
            print("   Converting model to FP16...")
            model = model.half()

            class FP16ModelWrapper(torch.nn.Module):
                """
                Wrapper that:
                - Accepts FP32 inputs (from Android)
                - Converts to FP16 internally
                - Runs FP16 computation (NPU accelerated)
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
            # Export with FP32 inputs AND FP32 outputs (what Android needs)
            exported = torch.export.export(wrapped_model, self.example_inputs)
            executorch_program = to_edge_transform_and_lower(
                exported,
                partitioner=[QnnPartitioner()]
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
                'name': 'QNN-FP16',
                'backend': 'NPU (Snapdragon)',
                'quantization': 'FP16',
                'filename': filename,
                'export_time': export_time,
                'file_size_mb': file_size
            }
        except ImportError as e:
            print(f"⚠️  QNN SDK not installed - Skipping: {e}")
            return None
        except Exception as e:
            print(f"❌ Error during QNN FP16 export: {e}")
            return None

    def run_all_exports(self):
        """Run all exports and generate summary"""
        results = []

        # Export all configurations
        results.append(self.export_baseline_portable())
        results.append(self.export_xnnpack_fp32())
        results.append(self.export_xnnpack_int8_pt2e())
        results.append(self.export_vulkan_fp32())
        results.append(self.export_vulkan_fp16())

        qnn_fp32 = self.export_qnn_snapdragon()
        if qnn_fp32:
            results.append(qnn_fp32)

        qnn_int8 = self.export_qnn_int8()
        if qnn_int8:
            results.append(qnn_int8)

        qnn_fp16 = self.export_qnn_fp16()
        if qnn_fp16:
            results.append(qnn_fp16)

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print export summary table"""
        print("\n" + "=" * 70)
        print("📊 EXPORT SUMMARY - MobileNetV3 Ready for S22 Ultra Testing")
        print("=" * 70)
        print(f"{'Configuration':<25} {'Backend':<20} {'Quant':<15} {'Size (MB)':<12} {'Time (s)'}")
        print("-" * 70)

        for r in results:
            print(f"{r['name']:<25} {r['backend']:<20} {r['quantization']:<15} "
                  f"{r['file_size_mb']:<12.2f} {r['export_time']:.2f}")


def main():
    """Main execution"""
    exporter = BackendExporter()
    results = exporter.run_all_exports()

    # Save results to JSON for reference
    with open(os.path.join(exporter.output_dir, "export_summary_mv3.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("📄 Export summary saved to: export_summary_mv3.json\n")


if __name__ == "__main__":
    main()