package com.uzi.lemon.core

/**
 * Supported ExecuTorch backends
 */
enum class BackendType {
    CPU,
    XNNPACK,
    QUALCOMM_HTP,
    MEDIATEK_NPU,
    VULKAN
}
