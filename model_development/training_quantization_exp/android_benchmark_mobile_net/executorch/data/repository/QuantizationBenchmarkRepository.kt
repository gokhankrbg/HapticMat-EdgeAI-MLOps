package com.uzi.executorch.data.repository

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.uzi.executorch.ImageNetClasses
import com.uzi.executorch.data.model.*
import com.uzi.executorch.presentation.viewmodel.ModelConfig
import com.uzi.executorch.presentation.viewmodel.ModelType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.random.Random

/**
 * Repository for benchmarking different quantization strategies
 * Focuses on comparing various quantized models available in assets
 */
class QuantizationBenchmarkRepository(private val context: Context) {
    
    companion object {
        private const val TAG = "QuantizationBenchmark"
    }
    
    // Define available models (excluding the two specified files)
    private val modelConfigs = listOf(
        ModelConfig("mv2_xnnpack_fp32.pte", "MobileNetV2 XNNPACK FP32", ModelType.XNNPACK_FP32),
        ModelConfig("mv2_xnnpack_int8_pt2e.pte", "MobileNetV2 XNNPACK INT8", ModelType.XNNPACK_INT8),
        ModelConfig("mv2_vulkan_fp32.pte", "MobileNetV2 Vulkan FP32", ModelType.VULKAN_FP32),
        ModelConfig("mv2_vulkan_fp16.pte", "MobileNetV2 Vulkan FP16", ModelType.VULKAN_FP16),
        ModelConfig("mv2_portable_fp32.pte", "MobileNetV2 Portable FP32", ModelType.PORTABLE_FP32)
    )
    
    /**
     * Get list of available models
     */
    fun getAvailableModels(): List<ModelConfig> = modelConfigs
    
    /**
     * Run complete benchmark on all available models
     */
    suspend fun runCompleteBenchmark(bitmap: Bitmap? = null): List<BenchmarkResult> = withContext(Dispatchers.IO) {
        Log.d(TAG, "Starting complete benchmark on ${modelConfigs.size} models...")
        
        val inputData = if (bitmap != null) {
            Log.d(TAG, "Using image input for benchmark")
            preprocessImageExact(bitmap)
        } else {
            Log.d(TAG, "Using random input for benchmark")
            createRandomInput()
        }
        
        val inputType = if (bitmap != null) InputType.IMAGE else InputType.RANDOM
        val results = mutableListOf<BenchmarkResult>()
        
        modelConfigs.forEach { config ->
            try {
                Log.d(TAG, "Benchmarking ${config.displayName}...")
                val result = benchmarkModel(config, inputData, inputType)
                results.add(result)
                Log.d(TAG, "✅ ${config.displayName}: ${result.inferenceTimeMs}ms")
            } catch (e: Exception) {
                Log.e(TAG, "❌ Failed to benchmark ${config.displayName}: ${e.message}", e)
            }
        }
        
        Log.d(TAG, "Benchmark completed. Successfully ran ${results.size}/${modelConfigs.size} models")
        results
    }
    
    /**
     * Benchmark a single model
     */
    private suspend fun benchmarkModel(
        config: ModelConfig,
        inputData: FloatArray,
        inputType: InputType
    ): BenchmarkResult = withContext(Dispatchers.IO) {
        val filePath = getAssetFilePath(config.fileName)
        val module = Module.load(filePath)
        
        try {
            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, 224, 224))
            val inputEValue = EValue.from(inputTensor)
            
            // Measure memory before inference
            val runtime = Runtime.getRuntime()
            runtime.gc() // Suggest garbage collection
            Thread.sleep(100) // Give GC time to work
            val memoryBefore = runtime.totalMemory() - runtime.freeMemory()
            
            val startTime = System.currentTimeMillis()
            val outputs = module.forward(inputEValue)
            val endTime = System.currentTimeMillis()
            
            // Measure memory after inference
            val memoryAfter = runtime.totalMemory() - runtime.freeMemory()
            val memoryUsedBytes = (memoryAfter - memoryBefore).coerceAtLeast(0)
            val memoryUsedMb = memoryUsedBytes / (1024.0 * 1024.0)
            
            val inferenceTime = endTime - startTime
            
            // Calculate energy efficiency score (lower is better)
            // This is a simplified metric: time * memory usage
            val energyEfficiencyScore = inferenceTime * memoryUsedMb
            
            Log.d(TAG, "${config.displayName} - Time: ${inferenceTime}ms, Memory: ${"%.2f".format(memoryUsedMb)}MB, Energy Score: ${"%.2f".format(energyEfficiencyScore)}")
            
            if (outputs.isEmpty()) {
                throw Exception("No outputs received from model")
            }
            
            val outputTensor = outputs[0].toTensor()
            val outputShape = outputTensor.shape()
            val outputData = outputTensor.dataAsFloatArray
            
            // Apply softmax to get probabilities
            val probabilities = applySoftmax(outputData)
            
            // Create predictions
            val predictions = probabilities.mapIndexed { index, confidence ->
                ClassificationPrediction(
                    classIndex = index,
                    className = try {
                        if (index < 1000) {
                            ImageNetClasses.getFormattedClassName(index)
                        } else {
                            "Class $index"
                        }
                    } catch (e: Exception) {
                        "Class $index"
                    },
                    confidence = confidence
                )
            }.sortedByDescending { it.confidence }.take(5)
            
            // Calculate statistics
            val statistics = ClassificationStatistics(
                minConfidence = probabilities.minOrNull() ?: 0f,
                maxConfidence = probabilities.maxOrNull() ?: 0f,
                meanConfidence = probabilities.average().toFloat(),
                topConfidence = probabilities.maxOrNull() ?: 0f
            )
            
            val classificationResult = ClassificationResult(
                predictions = predictions,
                inferenceTimeMs = inferenceTime,
                inputType = "${inputType.displayName} (${config.displayName})",
                outputShape = outputShape,
                isShapeCorrect = outputShape.contentEquals(longArrayOf(1, outputData.size.toLong())),
                statistics = statistics,
                modelType = com.uzi.executorch.data.model.ModelType.ORIGINAL // Using as generic type
            )
            
            BenchmarkResult(
                modelName = config.displayName,
                modelConfig = config,
                inferenceTimeMs = inferenceTime,
                classificationResult = classificationResult,
                memoryUsedMb = memoryUsedMb,
                energyEfficiencyScore = energyEfficiencyScore
            )
            
        } finally {
            // Module cleanup is handled by ExecuTorch
        }
    }
    
    /**
     * Create random input tensor (normalized for consistency)
     */
    private fun createRandomInput(): FloatArray {
        val inputSize = 1 * 3 * 224 * 224
        // Generate random values in the typical range after ImageNet normalization
        return FloatArray(inputSize) { 
            // Generate Gaussian-like distribution using Box-Muller transform
            val u1 = Random.nextFloat()
            val u2 = Random.nextFloat()
            val gaussian = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1.toDouble())) * 
                          kotlin.math.cos(2.0 * kotlin.math.PI * u2.toDouble())
            gaussian.toFloat()
        }
    }
    
    /**
     * EXACT preprocessing to match Python calibration
     */
    private fun preprocessImageExact(bitmap: Bitmap): FloatArray {
        Log.d(TAG, "Applying EXACT Android preprocessing to match Python calibration...")
        
        // Step 1: Resize image to 224x224 using bilinear interpolation
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        
        // Step 2: Extract pixels
        val pixels = IntArray(224 * 224)
        resizedBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        
        // Step 3: EXACT ImageNet normalization values
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f) // R, G, B
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)   // R, G, B
        
        // Step 4: Process pixels with EXACT same logic as Python
        val inputSize = 1 * 3 * 224 * 224
        val input = FloatArray(inputSize)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            
            // Extract RGB values and normalize to [0, 1] range
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // Apply ImageNet normalization and arrange in CHW format
            input[i] = (r - mean[0]) / std[0]                      // R channel
            input[224 * 224 + i] = (g - mean[1]) / std[1]          // G channel
            input[2 * 224 * 224 + i] = (b - mean[2]) / std[2]      // B channel
        }
        
        return input
    }
    
    /**
     * Apply softmax with improved numerical stability
     */
    private fun applySoftmax(logits: FloatArray): FloatArray {
        return try {
            if (logits.isEmpty()) {
                return floatArrayOf()
            }
            
            val maxLogit = logits.maxOrNull() ?: 0f
            
            // Subtract max for numerical stability
            val expValues = logits.map { 
                kotlin.math.exp((it - maxLogit).toDouble()).toFloat() 
            }
            
            val sum = expValues.sum()
            
            if (sum <= 0f || !sum.isFinite()) {
                return FloatArray(logits.size) { 1f / logits.size }
            }
            
            expValues.map { it / sum }.toFloatArray()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error applying softmax: ${e.message}", e)
            FloatArray(logits.size) { 1f / logits.size }
        }
    }
    
    /**
     * Get file path from assets
     */
    @Throws(IOException::class)
    private fun getAssetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        
        // Check if file already exists and is valid
        if (file.exists() && file.length() > 0) {
            Log.d(TAG, "Using cached model file: ${file.absolutePath}")
            return file.absolutePath
        }
        
        Log.d(TAG, "Copying model from assets: $assetName")
        
        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(8 * 1024)
                    var read: Int
                    var totalBytes = 0L
                    
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                        totalBytes += read
                    }
                    outputStream.flush()
                    
                    Log.d(TAG, "Successfully copied $assetName ($totalBytes bytes)")
                }
            }
        } catch (e: Exception) {
            if (file.exists()) {
                file.delete()
            }
            throw IOException("Failed to copy asset $assetName: ${e.message}", e)
        }
        
        return file.absolutePath
    }
}
