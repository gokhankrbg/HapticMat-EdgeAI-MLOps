# -*- coding: utf-8 -*-
"""Demo for OxfordML

ExecuTorch demonstration showing how to export a MobileNetV2 model
and run inference using the ExecuTorch runtime with XNNPACK backend.
"""

import torch
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch import version
from executorch.runtime import Runtime
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
import ssl
import urllib.request

def main():
    """Main demonstration function."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"ExecuTorch version: {version.__version__}")

    # Initialize ExecuTorch runtime
    runtime = Runtime.get()
    operator_names = runtime.operator_registry.operator_names
    print(f"Number of registered operators: {len(operator_names)}")

    # Load MobileNetV2 model
    print("Loading MobileNetV2 model...")
    mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model = mv2.eval()  # Set to evaluation mode

    # Define example inputs for model export
    example_inputs = (torch.randn((1, 3, 224, 224)),)

    # Export the model to Core ATen graph
    print("Exporting model to Core ATen graph...")
    exported_graph = torch.export.export(model, example_inputs)
    print("Exported graph:", exported_graph)

    # Transform and lower the graph with XNNPACK partitioner
    print("Transforming and lowering with XNNPACK partitioner...")
    executorch_program = to_edge_transform_and_lower(
        exported_graph,
        partitioner=[XnnpackPartitioner()]  # For accelerated CPU inference
    ).to_executorch()

    # Serialize to .pte file
    pte_path = "mv2_xnnpack.pte"
    print(f"Serializing model to {pte_path}...")
    with open(pte_path, "wb") as file:
        executorch_program.write_to_file(file)

    # Load and run the serialized program
    print("Loading and testing the serialized program...")
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    # Test inference
    test_input = torch.randn((1, 3, 224, 224))
    output = method.execute([test_input])

    # Validate output
    assert len(output) == 1, f"Unexpected output length {len(output)}"
    assert output[0].size() == torch.Size([1, 1000]), f"Unexpected output size {output[0].size()}"

    print("✅ PASS - Model export and inference successful!")
    print(f"Output shape: {output[0].size()}")


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context

    main()
