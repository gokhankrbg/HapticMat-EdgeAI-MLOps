package com.uzi.lemon.util

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import kotlin.math.sqrt

/**
 * Utility functions for tensor operations and analysis
 * 
 * Provides helpful methods for working with ExecuTorch tensors,
 * including statistics, validation, and data transformation.
 */
object TensorUtils {
    
    /**
     * Calculate statistics for a float array
     * 
     * @param data Input float array
     * @return TensorStatistics object
     */
    fun calculateStatistics(data: FloatArray): TensorStatistics {
        if (data.isEmpty()) {
            return TensorStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        }
        
        val min = data.minOrNull() ?: 0f
        val max = data.maxOrNull() ?: 0f
        val mean = data.average()
        val variance = data.map { (it - mean) * (it - mean) }.average()
        val stdDev = sqrt(variance)
        val median = data.sorted()[data.size / 2]
        
        return TensorStatistics(
            min = min.toDouble(),
            max = max.toDouble(),
            mean = mean,
            median = median.toDouble(),
            stdDev = stdDev,
            variance = variance,
            count = data.size
        )
    }
    
    /**
     * Calculate statistics from a tensor
     * 
     * @param tensor Input tensor
     * @return TensorStatistics object
     */
    fun calculateStatistics(tensor: Tensor): TensorStatistics {
        val data = tensor.dataAsFloatArray
        return calculateStatistics(data)
    }
    
    /**
     * Validate tensor shape matches expected shape
     * 
     * @param tensor Tensor to validate
     * @param expectedShape Expected shape
     * @return True if shapes match
     */
    fun validateShape(tensor: Tensor, expectedShape: LongArray): Boolean {
        val actualShape = tensor.shape()
        if (actualShape.size != expectedShape.size) return false
        return actualShape.contentEquals(expectedShape)
    }
    
    /**
     * Get human-readable string representation of tensor shape
     * 
     * @param tensor Input tensor
     * @return Shape as string (e.g., "[1, 3, 224, 224]")
     */
    fun shapeToString(tensor: Tensor): String {
        return tensor.shape().joinToString(prefix = "[", postfix = "]")
    }
    
    /**
     * Get human-readable string representation of shape array
     * 
     * @param shape Shape array
     * @return Shape as string
     */
    fun shapeToString(shape: LongArray): String {
        return shape.joinToString(prefix = "[", postfix = "]")
    }
    
    /**
     * Calculate total number of elements in a tensor
     * 
     * @param shape Tensor shape
     * @return Total number of elements
     */
    fun calculateTotalElements(shape: LongArray): Long {
        return shape.fold(1L) { acc, dim -> acc * dim }
    }
    
    /**
     * Normalize float array to range [0, 1]
     * 
     * @param data Input data
     * @return Normalized data
     */
    fun normalizeToZeroOne(data: FloatArray): FloatArray {
        val min = data.minOrNull() ?: 0f
        val max = data.maxOrNull() ?: 1f
        val range = max - min
        
        return if (range > 0) {
            data.map { (it - min) / range }.toFloatArray()
        } else {
            data
        }
    }
    
    /**
     * Normalize float array to range [-1, 1]
     * 
     * @param data Input data
     * @return Normalized data
     */
    fun normalizeToMinusOneOne(data: FloatArray): FloatArray {
        val min = data.minOrNull() ?: -1f
        val max = data.maxOrNull() ?: 1f
        val range = max - min
        
        return if (range > 0) {
            data.map { 2 * ((it - min) / range) - 1 }.toFloatArray()
        } else {
            data
        }
    }
    
    /**
     * Apply softmax to logits
     * 
     * @param logits Input logits
     * @return Probabilities (sum to 1)
     */
    fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return floatArrayOf()
        
        // Subtract max for numerical stability
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
        val sum = expValues.sum()
        
        return if (sum > 0) {
            expValues.map { it / sum }.toFloatArray()
        } else {
            FloatArray(logits.size) { 1f / logits.size }
        }
    }
    
    /**
     * Get top K indices with highest values
     * 
     * @param data Input data
     * @param k Number of top indices to return
     * @return List of (index, value) pairs
     */
    fun topK(data: FloatArray, k: Int): List<Pair<Int, Float>> {
        return data.mapIndexed { index, value -> index to value }
            .sortedByDescending { it.second }
            .take(k)
    }
    
    /**
     * Check if tensor contains NaN or Inf values
     * 
     * @param tensor Input tensor
     * @return True if tensor contains invalid values
     */
    fun hasInvalidValues(tensor: Tensor): Boolean {
        val data = tensor.dataAsFloatArray
        return data.any { it.isNaN() || it.isInfinite() }
    }
    
    /**
     * Check if tensor contains NaN or Inf values
     * 
     * @param data Input data
     * @return True if data contains invalid values
     */
    fun hasInvalidValues(data: FloatArray): Boolean {
        return data.any { it.isNaN() || it.isInfinite() }
    }
    
    /**
     * Compare two tensors element-wise
     * 
     * @param tensor1 First tensor
     * @param tensor2 Second tensor
     * @param tolerance Absolute tolerance for floating point comparison
     * @return True if tensors are equal within tolerance
     */
    fun areEqual(tensor1: Tensor, tensor2: Tensor, tolerance: Float = 1e-5f): Boolean {
        if (!tensor1.shape().contentEquals(tensor2.shape())) return false
        
        val data1 = tensor1.dataAsFloatArray
        val data2 = tensor2.dataAsFloatArray
        
        return data1.zip(data2).all { (a, b) -> kotlin.math.abs(a - b) <= tolerance }
    }
    
    /**
     * Calculate mean squared error between two tensors
     * 
     * @param tensor1 First tensor (e.g., ground truth)
     * @param tensor2 Second tensor (e.g., prediction)
     * @return MSE value
     */
    fun calculateMSE(tensor1: Tensor, tensor2: Tensor): Double {
        require(tensor1.shape().contentEquals(tensor2.shape())) {
            "Tensors must have same shape"
        }
        
        val data1 = tensor1.dataAsFloatArray
        val data2 = tensor2.dataAsFloatArray
        
        return data1.zip(data2).map { (a, b) ->
            val diff = a - b
            (diff * diff).toDouble()
        }.average()
    }
    
    /**
     * Convert EValue to Tensor
     * 
     * @param eValue Input EValue
     * @return Tensor if EValue contains tensor, null otherwise
     */
    fun toTensor(eValue: EValue): Tensor? {
        return try {
            eValue.toTensor()
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Create a zero tensor with given shape
     * 
     * @param shape Desired tensor shape
     * @return Tensor filled with zeros
     */
    fun zeros(shape: LongArray): Tensor {
        val totalSize = calculateTotalElements(shape).toInt()
        val data = FloatArray(totalSize) { 0f }
        return Tensor.fromBlob(data, shape)
    }
    
    /**
     * Create a ones tensor with given shape
     * 
     * @param shape Desired tensor shape
     * @return Tensor filled with ones
     */
    fun ones(shape: LongArray): Tensor {
        val totalSize = calculateTotalElements(shape).toInt()
        val data = FloatArray(totalSize) { 1f }
        return Tensor.fromBlob(data, shape)
    }
    
    /**
     * Create a tensor filled with a specific value
     * 
     * @param shape Desired tensor shape
     * @param value Fill value
     * @return Tensor filled with the specified value
     */
    fun full(shape: LongArray, value: Float): Tensor {
        val totalSize = calculateTotalElements(shape).toInt()
        val data = FloatArray(totalSize) { value }
        return Tensor.fromBlob(data, shape)
    }
    
    /**
     * Statistics for a tensor
     */
    data class TensorStatistics(
        val min: Double,
        val max: Double,
        val mean: Double,
        val median: Double,
        val stdDev: Double,
        val variance: Double,
        val count: Int
    ) {
        override fun toString(): String {
            return """
                Tensor Statistics:
                  Count: $count
                  Min: ${"%.4f".format(min)}
                  Max: ${"%.4f".format(max)}
                  Mean: ${"%.4f".format(mean)}
                  Median: ${"%.4f".format(median)}
                  Std Dev: ${"%.4f".format(stdDev)}
                  Variance: ${"%.4f".format(variance)}
            """.trimIndent()
        }
    }
}
