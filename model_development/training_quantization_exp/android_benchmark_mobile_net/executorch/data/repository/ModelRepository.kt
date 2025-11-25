package com.uzi.executorch.data.repository

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.uzi.executorch.data.model.*
import com.uzi.executorch.ImageNetClasses
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
 * Enhanced Repository for managing dual ExecuTorch model operations with benchmarking
 */
class ModelRepository(private val context: Context) {
    
    companion object {
        private const val TAG = "ModelRepository"
        private const val ORIGINAL_MODEL_NAME = "mobilenet_v2_original_executorch.pte"
        private const val QUANTIZED_MODEL_NAME = "mobilenet_v2_quantized_executorch.pte"
    }
    
    private var originalModule: Module? = null
    private var quantizedModule: Module? = null
    
    /**
     * Load both models from assets
     */
    suspend fun loadModels(): Result<Pair<Boolean, Boolean>> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Loading both models...")
            
            var originalLoaded = false
            var quantizedLoaded = false
            
            // Load original model
            try {
                val originalFile = getAssetFilePath(ORIGINAL_MODEL_NAME)
                originalModule = Module.load(originalFile)
                originalLoaded = true
                Log.d(TAG, "✅ Original model loaded successfully")
            } catch (e: Exception) {
                Log.e(TAG, "❌ Failed to load original model: ${e.message}")
            }
            
            // Load quantized model
            try {
                val quantizedFile = getAssetFilePath(QUANTIZED_MODEL_NAME)
                quantizedModule = Module.load(quantizedFile)
                quantizedLoaded = true
                Log.d(TAG, "✅ Quantized model loaded successfully")
            } catch (e: Exception) {
                Log.e(TAG, "❌ Failed to load quantized model: ${e.message}")
            }
            
            if (!originalLoaded && !quantizedLoaded) {
                Result.failure(Exception("Failed to load both models"))
            } else {
                Result.success(Pair(originalLoaded, quantizedLoaded))
            }
            
        } catch (e: Exception) {
            val error = "Failed to load models: ${e.message}"
            Log.e(TAG, error, e)
            Result.failure(Exception(error))
        }
    }
    
    /**
     * Run benchmark comparison between both models
     */
    suspend fun runBenchmarkComparison(bitmap: Bitmap? = null): Result<BenchmarkComparisonResult> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Running benchmark comparison...")
            
            val inputData = if (bitmap != null) {
                preprocessImageExact(bitmap)
            } else {
                createRandomInput()
            }
            
            val inputType = if (bitmap != null) InputType.IMAGE else InputType.RANDOM
            
            val originalResult = if (originalModule != null) {
                runInferenceOnModel(originalModule!!, inputData, ModelType.ORIGINAL, inputType)
            } else null
            
            val quantizedResult = if (quantizedModule != null) {
                runInferenceOnModel(quantizedModule!!, inputData, ModelType.QUANTIZED, inputType)
            } else null
            
            if (originalResult == null && quantizedResult == null) {
                return@withContext Result.failure(Exception("No models available for comparison"))
            }
            
            val comparison = BenchmarkComparisonResult(
                originalResult = originalResult,
                quantizedResult = quantizedResult,
                inputType = inputType,
                speedupRatio = if (originalResult != null && quantizedResult != null && 
                    originalResult.isSuccess && quantizedResult.isSuccess) {
                    val origTime = originalResult.getOrNull()?.inferenceTimeMs?.toDouble()
                    val quantTime = quantizedResult.getOrNull()?.inferenceTimeMs?.toDouble()
                    if (origTime != null && quantTime != null && quantTime > 0) {
                        origTime / quantTime
                    } else null
                } else null,
                predictionMatch = if (originalResult != null && quantizedResult != null && 
                    originalResult.isSuccess && quantizedResult.isSuccess) {
                    val origPred = originalResult.getOrNull()?.predictions?.firstOrNull()?.classIndex
                    val quantPred = quantizedResult.getOrNull()?.predictions?.firstOrNull()?.classIndex
                    origPred != null && quantPred != null && origPred == quantPred
                } else null
            )
            
            Log.d(TAG, "Benchmark comparison completed successfully")
            Result.success(comparison)
            
        } catch (e: Exception) {
            val error = "Benchmark comparison failed: ${e.message}"
            Log.e(TAG, error, e)
            Result.failure(Exception(error))
        }
    }
    
    /**
     * Run inference with random input data on original model
     */
    suspend fun runRandomInference(): Result<ClassificationResult> = withContext(Dispatchers.IO) {
        val randomInput = createRandomInput()
        val module = originalModule ?: quantizedModule 
            ?: return@withContext Result.failure(Exception("No model loaded"))
        runInferenceOnModel(module, randomInput, ModelType.ORIGINAL, InputType.RANDOM)
    }
    
    /**
     * Run inference with image input on both models if available
     */
    suspend fun runImageInference(bitmap: Bitmap): Result<ClassificationResult> = withContext(Dispatchers.IO) {
        val imageInput = preprocessImageExact(bitmap)
        
        // If both models are available, run benchmark comparison and return the original result
        if (originalModule != null && quantizedModule != null) {
            Log.d(TAG, "Both models available, running comparison for image classification...")
            
            val originalResult = runInferenceOnModel(originalModule!!, imageInput, ModelType.ORIGINAL, InputType.IMAGE)
            val quantizedResult = runInferenceOnModel(quantizedModule!!, imageInput, ModelType.QUANTIZED, InputType.IMAGE)
            
            // Log comparison for debugging
            if (originalResult.isSuccess && quantizedResult.isSuccess) {
                val origConf = originalResult.getOrNull()?.predictions?.firstOrNull()?.confidencePercentage
                val quantConf = quantizedResult.getOrNull()?.predictions?.firstOrNull()?.confidencePercentage
                val origClass = originalResult.getOrNull()?.predictions?.firstOrNull()?.className
                val quantClass = quantizedResult.getOrNull()?.predictions?.firstOrNull()?.className
                
                Log.d(TAG, "Image Classification Comparison:")
                Log.d(TAG, "  Original: $origClass (${"%.1f".format(origConf ?: 0f)}%)")
                Log.d(TAG, "  Quantized: $quantClass (${"%.1f".format(quantConf ?: 0f)}%)")
                Log.d(TAG, "  Match: ${origClass == quantClass}")
            }
            
            // Return the original result, but the comparison will be logged
            return@withContext originalResult
        }
        
        // Fallback to single model if only one is available
        val module = originalModule ?: quantizedModule 
            ?: return@withContext Result.failure(Exception("No model loaded"))
        val modelType = if (module == originalModule) ModelType.ORIGINAL else ModelType.QUANTIZED
        runInferenceOnModel(module, imageInput, modelType, InputType.IMAGE)
    }
    
    /**
     * Check if any model is loaded
     */
    fun isModelLoaded(): Boolean = originalModule != null || quantizedModule != null
    
    /**
     * Get model status
     */
    fun getModelStatus(): Pair<Boolean, Boolean> = Pair(
        originalModule != null,
        quantizedModule != null
    )
    
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
     * EXACT preprocessing to match Python calibration - CRITICAL for quantized model accuracy
     */
    private fun preprocessImageExact(bitmap: Bitmap): FloatArray {
        Log.d(TAG, "Applying EXACT Android preprocessing to match Python calibration...")
        
        // Step 1: Resize image to 224x224 using bilinear interpolation (matches Android default)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        
        // Step 2: Extract pixels
        val pixels = IntArray(224 * 224)
        resizedBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        
        // Step 3: EXACT ImageNet normalization values (critical!)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f) // R, G, B
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)   // R, G, B
        
        // Step 4: Process pixels with EXACT same logic as Python
        val inputSize = 1 * 3 * 224 * 224
        val input = FloatArray(inputSize)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            
            // Extract RGB values and normalize to [0, 1] range
            // This MUST match the Python preprocessing exactly
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // Apply ImageNet normalization and arrange in CHW format
            // Layout: [R_pixels, G_pixels, B_pixels]
            input[i] = (r - mean[0]) / std[0]                      // R channel
            input[224 * 224 + i] = (g - mean[1]) / std[1]          // G channel
            input[2 * 224 * 224 + i] = (b - mean[2]) / std[2]      // B channel
        }
        
        // Log preprocessing statistics for debugging
        val rChannelMean = input.slice(0 until 224*224).average()
        val gChannelMean = input.slice(224*224 until 2*224*224).average()
        val bChannelMean = input.slice(2*224*224 until 3*224*224).average()
        
        Log.d(TAG, "Preprocessing stats - R: %.3f, G: %.3f, B: %.3f".format(rChannelMean, gChannelMean, bChannelMean))
        Log.d(TAG, "Value range: [%.3f, %.3f]".format(input.minOrNull(), input.maxOrNull()))
        
        return input
    }
    
    /**
     * Run inference on specific model with detailed logging
     */
    private fun runInferenceOnModel(
        module: Module, 
        inputData: FloatArray, 
        modelType: ModelType, 
        inputType: InputType
    ): Result<ClassificationResult> {
        return try {
            Log.d(TAG, "Running inference on ${modelType.name} model with ${inputType.displayName}...")
            
            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, 224, 224))
            val inputEValue = EValue.from(inputTensor)
            
            val startTime = System.currentTimeMillis()
            val outputs = module.forward(inputEValue)
            val endTime = System.currentTimeMillis()
            
            val inferenceTime = endTime - startTime
            Log.d(TAG, "${modelType.name} inference completed in ${inferenceTime}ms")
            
            if (outputs.isEmpty()) {
                return Result.failure(Exception("No outputs received from ${modelType.name} model"))
            }
            
            val outputTensor = outputs[0].toTensor()
            val outputShape = outputTensor.shape()
            val outputData = outputTensor.dataAsFloatArray
            
            Log.d(TAG, "${modelType.name} output shape: ${outputShape.contentToString()}")
            Log.d(TAG, "${modelType.name} output data length: ${outputData.size}")
            
            // Log raw output statistics for comparison
            Log.d(TAG, "${modelType.name} raw output range: [%.6f, %.6f]".format(
                outputData.minOrNull(), outputData.maxOrNull()))
            
            // Apply softmax to get probabilities
            val probabilities = applySoftmax(outputData)
            
            // Log top raw predictions before softmax
            val topRawIndices = outputData.mapIndexed { index, value -> index to value }
                .sortedByDescending { it.second }.take(3)
            Log.d(TAG, "${modelType.name} top 3 raw logits: ${topRawIndices}")
            
            // Shape validation
            val actualClasses = if (outputShape.size >= 2) outputShape[1].toInt() else outputData.size
            val expectedShape = longArrayOf(1, actualClasses.toLong())
            val isShapeCorrect = outputShape.contentEquals(expectedShape)
            
            // Create predictions with bounds checking
            val predictions = probabilities.mapIndexed { index, confidence ->
                ClassificationPrediction(
                    classIndex = index,
                    className = try {
                        if (index < 1000 && actualClasses <= 1000) {
                            ImageNetClasses.getFormattedClassName(index)
                        } else {
                            "Class $index (Custom Model)"
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to get class name for index $index: ${e.message}")
                        "Class $index"
                    },
                    confidence = confidence
                )
            }.sortedByDescending { it.confidence }.take(5)
            
            // Log top predictions for comparison
            Log.d(TAG, "${modelType.name} top prediction: ${predictions.firstOrNull()?.className} " +
                    "(${predictions.firstOrNull()?.confidencePercentage?.let { "%.2f".format(it) }}%)")
            
            // Calculate statistics
            val statistics = ClassificationStatistics(
                minConfidence = probabilities.minOrNull() ?: 0f,
                maxConfidence = probabilities.maxOrNull() ?: 0f,
                meanConfidence = probabilities.average().toFloat(),
                topConfidence = probabilities.maxOrNull() ?: 0f
            )
            
            val result = ClassificationResult(
                predictions = predictions,
                inferenceTimeMs = inferenceTime,
                inputType = "${inputType.displayName} (${modelType.name})",
                outputShape = outputShape,
                isShapeCorrect = isShapeCorrect,
                statistics = statistics,
                modelType = modelType
            )
            
            Log.d(TAG, "${modelType.name} classification completed successfully")
            Result.success(result)
            
        } catch (e: Exception) {
            val error = "${modelType.name} inference failed: ${e.message}"
            Log.e(TAG, error, e)
            Result.failure(Exception(error))
        }
    }
    
    /**
     * Apply softmax with improved numerical stability
     */
    private fun applySoftmax(logits: FloatArray): FloatArray {
        return try {
            if (logits.isEmpty()) {
                Log.w(TAG, "Empty logits array, returning empty probabilities")
                return floatArrayOf()
            }
            
            val maxLogit = logits.maxOrNull() ?: 0f
            Log.d(TAG, "Applying softmax - input size: ${logits.size}, max logit: %.6f".format(maxLogit))
            
            // Subtract max for numerical stability
            val expValues = logits.map { 
                kotlin.math.exp((it - maxLogit).toDouble()).toFloat() 
            }
            
            val sum = expValues.sum()
            
            if (sum <= 0f || !sum.isFinite()) {
                Log.w(TAG, "Invalid softmax sum: $sum, returning uniform distribution")
                return FloatArray(logits.size) { 1f / logits.size }
            }
            
            val result = expValues.map { it / sum }.toFloatArray()
            val resultSum = result.sum()
            
            Log.d(TAG, "Softmax applied successfully - output sum: %.6f, max prob: %.6f".format(
                resultSum, result.maxOrNull()))
            
            result
        } catch (e: Exception) {
            Log.e(TAG, "Error applying softmax: ${e.message}", e)
            // Return uniform distribution as fallback
            FloatArray(logits.size) { 1f / logits.size }
        }
    }
    
    /**
     * Get file path from assets with improved error handling
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
                    val buffer = ByteArray(8 * 1024) // 8KB buffer for better performance
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
            // Clean up partial file
            if (file.exists()) {
                file.delete()
            }
            throw IOException("Failed to copy asset $assetName: ${e.message}", e)
        }
        
        return file.absolutePath
    }
}