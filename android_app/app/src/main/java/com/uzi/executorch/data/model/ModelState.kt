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
