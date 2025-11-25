package com.uzi.executorch.data.model

/**
 * Data class representing a single classification prediction
 */
data class ClassificationPrediction(
    val classIndex: Int,
    val className: String,
    val confidence: Float,
    val confidencePercentage: Float = confidence * 100f
)

/**
 * Data class representing the complete classification result
 */
data class ClassificationResult(
    val predictions: List<ClassificationPrediction>,
    val inferenceTimeMs: Long,
    val inputType: String,
    val outputShape: LongArray,
    val isShapeCorrect: Boolean,
    val statistics: ClassificationStatistics
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

        return true
    }

    override fun hashCode(): Int {
        var result = predictions.hashCode()
        result = 31 * result + inferenceTimeMs.hashCode()
        result = 31 * result + inputType.hashCode()
        result = 31 * result + outputShape.contentHashCode()
        result = 31 * result + isShapeCorrect.hashCode()
        result = 31 * result + statistics.hashCode()
        return result
    }
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
