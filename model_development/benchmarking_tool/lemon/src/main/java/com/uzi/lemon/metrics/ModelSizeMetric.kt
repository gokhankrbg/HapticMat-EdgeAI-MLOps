package com.uzi.lemon.metrics

import com.uzi.lemon.core.EvaluableModule
import com.uzi.lemon.core.Metric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import org.pytorch.executorch.EValue
import java.io.File

/**
 * Result of model size measurement
 */
@Serializable
data class ModelSizeResult(
    val sizeBytes: Long,             // Model size in bytes
    val sizeMB: Double,              // Model size in megabytes
    val sizeKB: Double,              // Model size in kilobytes
    val filePath: String,            // Path to the model file
    val isValid: Boolean,            // Whether the file exists and is readable
    override val unit: String = "MB"
) : Metric {
    override val name: String = "ModelSize"
    
    override fun toString(): String {
        return """
            Model Size Statistics:
              Size: ${"%.2f".format(sizeMB)} MB ($sizeBytes bytes)
              File: $filePath
              Valid: ${if (isValid) "✅ Yes" else "❌ No"}
        """.trimIndent()
    }
    
    /**
     * Get human-readable size string
     */
    fun getReadableSize(): String {
        return when {
            sizeBytes < 1024 -> "$sizeBytes B"
            sizeBytes < 1024 * 1024 -> "${"%.2f".format(sizeKB)} KB"
            sizeBytes < 1024 * 1024 * 1024 -> "${"%.2f".format(sizeMB)} MB"
            else -> "${"%.2f".format(sizeBytes / (1024.0 * 1024.0 * 1024.0))} GB"
        }
    }
}

/**
 * Measures model file size
 * 
 * This metric provides detailed information about the model file size,
 * which is useful for understanding storage requirements and compression ratios.
 */
class ModelSizeMetric {
    
    /**
     * Measure model size for the given module
     * 
     * Note: inputs parameter is not used but kept for API consistency
     * 
     * @param module The evaluable module to measure
     * @param inputs Not used (kept for API consistency)
     * @return ModelSizeResult containing size information
     */
    suspend fun measure(
        module: EvaluableModule,
        inputs: List<Array<EValue>>
    ): ModelSizeResult = withContext(Dispatchers.IO) {
        measureFromModule(module)
    }
    
    /**
     * Measure model size directly from a file path
     * 
     * @param filePath Absolute path to the model file
     * @return ModelSizeResult containing size information
     */
    suspend fun measureFromPath(filePath: String): ModelSizeResult = withContext(Dispatchers.IO) {
        val file = File(filePath)
        
        if (!file.exists() || !file.isFile) {
            return@withContext ModelSizeResult(
                sizeBytes = 0L,
                sizeMB = 0.0,
                sizeKB = 0.0,
                filePath = filePath,
                isValid = false
            )
        }
        
        val sizeBytes = file.length()
        
        ModelSizeResult(
            sizeBytes = sizeBytes,
            sizeMB = sizeBytes / (1024.0 * 1024.0),
            sizeKB = sizeBytes / 1024.0,
            filePath = filePath,
            isValid = true
        )
    }
    
    /**
     * Measure model size from an EvaluableModule
     * 
     * @param module The evaluable module
     * @return ModelSizeResult containing size information
     */
    private fun measureFromModule(module: EvaluableModule): ModelSizeResult {
        val sizeBytes = module.getModelSize()
        
        return if (sizeBytes > 0) {
            ModelSizeResult(
                sizeBytes = sizeBytes,
                sizeMB = sizeBytes / (1024.0 * 1024.0),
                sizeKB = sizeBytes / 1024.0,
                filePath = "From module",
                isValid = true
            )
        } else {
            ModelSizeResult(
                sizeBytes = 0L,
                sizeMB = 0.0,
                sizeKB = 0.0,
                filePath = "Unknown",
                isValid = false
            )
        }
    }
    
    companion object {
        /**
         * Compare two model sizes and calculate compression ratio
         * 
         * @param original Original model size result
         * @param compressed Compressed model size result
         * @return Compression ratio (e.g., 4.0 means compressed is 4x smaller)
         */
        fun calculateCompressionRatio(
            original: ModelSizeResult,
            compressed: ModelSizeResult
        ): Double {
            return if (compressed.sizeBytes > 0) {
                original.sizeBytes.toDouble() / compressed.sizeBytes.toDouble()
            } else {
                1.0
            }
        }
        
        /**
         * Calculate size reduction percentage
         * 
         * @param original Original model size result
         * @param compressed Compressed model size result
         * @return Percentage reduction (e.g., 75.0 means 75% smaller)
         */
        fun calculateSizeReduction(
            original: ModelSizeResult,
            compressed: ModelSizeResult
        ): Double {
            return if (original.sizeBytes > 0) {
                ((original.sizeBytes - compressed.sizeBytes).toDouble() / original.sizeBytes.toDouble()) * 100.0
            } else {
                0.0
            }
        }
    }
}
