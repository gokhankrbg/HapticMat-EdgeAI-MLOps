package com.uzi.lemon.util

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import kotlin.random.Random

/**
 * Utility class for loading and preparing test data for model evaluation
 * 
 * Provides convenient methods to generate various types of input data
 * for testing ExecuTorch models.
 */
class DataLoader(private val context: Context) {
    
    /**
     * Generate random normalized input tensors
     * 
     * @param count Number of input tensors to generate
     * @param shape Shape of each tensor (default: [1, 3, 224, 224] for MobileNet)
     * @param normalization Normalization method
     * @return List of input arrays ready for evaluation
     */
    fun generateRandomInputs(
        count: Int,
        shape: LongArray = longArrayOf(1, 3, 224, 224),
        normalization: NormalizationType = NormalizationType.STANDARD
    ): List<Array<EValue>> {
        val totalSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
        
        return List(count) {
            val data = when (normalization) {
                NormalizationType.STANDARD -> {
                    // Standard normal distribution (-1 to 1)
                    FloatArray(totalSize) { (Random.nextFloat() * 2) - 1 }
                }
                NormalizationType.ZERO_ONE -> {
                    // Uniform distribution (0 to 1)
                    FloatArray(totalSize) { Random.nextFloat() }
                }
                NormalizationType.IMAGENET -> {
                    // ImageNet normalization (approximate)
                    FloatArray(totalSize) {
                        val mean = if (it % 3 == 0) 0.485f else if (it % 3 == 1) 0.456f else 0.406f
                        val std = if (it % 3 == 0) 0.229f else if (it % 3 == 1) 0.224f else 0.225f
                        ((Random.nextFloat() - mean) / std)
                    }
                }
            }
            
            val tensor = Tensor.fromBlob(data, shape)
            arrayOf(EValue.from(tensor))
        }
    }
    
    /**
     * Generate fixed input tensors (useful for deterministic testing)
     * 
     * @param count Number of input tensors to generate
     * @param shape Shape of each tensor
     * @param value Fixed value to fill (default: 0.5f)
     * @return List of input arrays
     */
    fun generateFixedInputs(
        count: Int,
        shape: LongArray = longArrayOf(1, 3, 224, 224),
        value: Float = 0.5f
    ): List<Array<EValue>> {
        val totalSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
        
        return List(count) {
            val data = FloatArray(totalSize) { value }
            val tensor = Tensor.fromBlob(data, shape)
            arrayOf(EValue.from(tensor))
        }
    }
    
    /**
     * Convert bitmap to tensor with ImageNet preprocessing
     * 
     * @param bitmap Input bitmap image
     * @param targetSize Target size for resizing (default: 224)
     * @return Preprocessed tensor
     */
    fun bitmapToTensor(
        bitmap: Bitmap,
        targetSize: Int = 224
    ): Array<EValue> {
        // Resize bitmap
        val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
        
        // Extract pixels
        val pixels = IntArray(targetSize * targetSize)
        resized.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)
        
        // ImageNet normalization
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        // Convert to CHW format with normalization
        val input = FloatArray(3 * targetSize * targetSize)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            
            // Extract RGB and normalize to [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // Apply ImageNet normalization and arrange in CHW format
            input[i] = (r - mean[0]) / std[0]
            input[targetSize * targetSize + i] = (g - mean[1]) / std[1]
            input[2 * targetSize * targetSize + i] = (b - mean[2]) / std[2]
        }
        
        val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong()))
        return arrayOf(EValue.from(tensor))
    }
    
    /**
     * Generate a batch of repeated inputs (useful for throughput testing)
     * 
     * @param singleInput Single input to repeat
     * @param batchSize Number of times to repeat
     * @return List of repeated inputs
     */
    fun repeatInput(singleInput: Array<EValue>, batchSize: Int): List<Array<EValue>> {
        return List(batchSize) { singleInput }
    }
    
    /**
     * Generate inputs with progressive difficulty (e.g., increasing noise)
     * 
     * @param count Number of inputs to generate
     * @param shape Tensor shape
     * @return List of inputs with varying characteristics
     */
    fun generateProgressiveInputs(
        count: Int,
        shape: LongArray = longArrayOf(1, 3, 224, 224)
    ): List<Array<EValue>> {
        val totalSize = shape.fold(1L) { acc, dim -> acc * dim }.toInt()
        
        return List(count) { index ->
            // Increase noise level progressively
            val noiseLevel = (index.toFloat() / count) * 2f
            val data = FloatArray(totalSize) {
                ((Random.nextFloat() * 2) - 1) * noiseLevel
            }
            
            val tensor = Tensor.fromBlob(data, shape)
            arrayOf(EValue.from(tensor))
        }
    }
    
    /**
     * Normalization types for input data
     */
    enum class NormalizationType {
        /** Standard normalization: values in range [-1, 1] */
        STANDARD,
        
        /** Zero-one normalization: values in range [0, 1] */
        ZERO_ONE,
        
        /** ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] */
        IMAGENET
    }
    
    companion object {
        /**
         * Common input shapes for popular models
         */
        object CommonShapes {
            /** MobileNet V2 input shape */
            val MOBILENET = longArrayOf(1, 3, 224, 224)
            
            /** ResNet input shape */
            val RESNET = longArrayOf(1, 3, 224, 224)
            
            /** EfficientNet input shape */
            val EFFICIENTNET = longArrayOf(1, 3, 224, 224)
            
            /** SqueezeNet input shape */
            val SQUEEZENET = longArrayOf(1, 3, 227, 227)
        }
    }
}
