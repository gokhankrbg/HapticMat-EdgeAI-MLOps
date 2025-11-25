package com.uzi.executorch.data.model

/**
 * Sealed class representing the state of model loading
 */
sealed class ModelLoadState {
    object NotLoaded : ModelLoadState()
    object Loading : ModelLoadState()
    object Loaded : ModelLoadState()
    data class Error(val message: String) : ModelLoadState()
}

/**
 * Sealed class representing the state of inference
 */
sealed class InferenceState {
    object Idle : InferenceState()
    object Processing : InferenceState()
    data class Success(val result: ClassificationResult) : InferenceState()
    data class Error(val message: String) : InferenceState()
}

/**
 * Data class for UI state
 */
data class UiState(
    val isModelSectionMinimized: Boolean = false,
    val selectedImagePath: String? = null
)

/**
 * Enum for input types
 */
enum class InputType(val displayName: String) {
    RANDOM("Random Test Data"),
    IMAGE("Real Image")
}

/**
 * Enum for model types
 */
enum class ModelType(val displayName: String) {
    ORIGINAL("Original FP32"),
    QUANTIZED("Quantized INT8")
}

/**
 * Data class for individual classification prediction
 */
data class ClassificationPrediction(
    val classIndex: Int,
    val className: String,
    val confidence: Float
) {
    val confidencePercentage: Float
        get() = confidence * 100f
}

/**
 * Data class for classification statistics
 */
data class ClassificationStatistics(
    val minConfidence: Float,
    val maxConfidence: Float,
    val meanConfidence: Float,
    val topConfidence: Float
)

/**
 * Data class for classification results
 */
data class ClassificationResult(
    val predictions: List<ClassificationPrediction>,
    val inferenceTimeMs: Long,
    val inputType: String,
    val outputShape: LongArray,
    val isShapeCorrect: Boolean,
    val statistics: ClassificationStatistics,
    val modelType: ModelType = ModelType.ORIGINAL
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ClassificationResult

        if (predictions != other.predictions) return false
        if (inferenceTimeMs != other.inferenceTimeMs) return false
        if (inputType != other.inputType) return false
        if (!outputShape.contentEquals(other.outputShape)) return false
        if (isShapeCorrect != other.isShapeCorrect) return false
        if (statistics != other.statistics) return false
        if (modelType != other.modelType) return false

        return true
    }

    override fun hashCode(): Int {
        var result = predictions.hashCode()
        result = 31 * result + inferenceTimeMs.hashCode()
        result = 31 * result + inputType.hashCode()
        result = 31 * result + outputShape.contentHashCode()
        result = 31 * result + isShapeCorrect.hashCode()
        result = 31 * result + statistics.hashCode()
        result = 31 * result + modelType.hashCode()
        return result
    }
}

/**
 * Data class for benchmark comparison results
 */
data class BenchmarkComparisonResult(
    val originalResult: Result<ClassificationResult>?,
    val quantizedResult: Result<ClassificationResult>?,
    val inputType: InputType,
    val speedupRatio: Double? = null,
    val predictionMatch: Boolean? = null
) {
    val hasOriginal: Boolean get() = originalResult?.isSuccess == true
    val hasQuantized: Boolean get() = quantizedResult?.isSuccess == true
    val hasBoth: Boolean get() = hasOriginal && hasQuantized
    
    fun getOriginalResult(): ClassificationResult? = originalResult?.getOrNull()
    fun getQuantizedResult(): ClassificationResult? = quantizedResult?.getOrNull()
    
    val performanceImprovement: Double?
        get() = speedupRatio?.let { (it - 1.0) * 100.0 }
}

/**
 * Data class for individual benchmark results
 */
data class BenchmarkResult(
    val modelName: String,
    val modelConfig: com.uzi.executorch.presentation.viewmodel.ModelConfig,
    val inferenceTimeMs: Long,
    val classificationResult: ClassificationResult,
    val memoryUsedMb: Double = 0.0,
    val energyEfficiencyScore: Double = 0.0 // Lower is better (ms * MB)
)