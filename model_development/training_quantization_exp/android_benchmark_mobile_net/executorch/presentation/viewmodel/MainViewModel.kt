package com.uzi.executorch.presentation.viewmodel

import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.uzi.executorch.data.model.*
import com.uzi.executorch.data.repository.ModelRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Enhanced ViewModel for the main screen with dual model benchmarking support
 */
class MainViewModel(
    private val modelRepository: ModelRepository
) : ViewModel() {
    
    // Model loading state
    private val _modelLoadState = MutableStateFlow<ModelLoadState>(ModelLoadState.NotLoaded)
    val modelLoadState: StateFlow<ModelLoadState> = _modelLoadState.asStateFlow()
    
    // Inference state
    private val _inferenceState = MutableStateFlow<InferenceState>(InferenceState.Idle)
    val inferenceState: StateFlow<InferenceState> = _inferenceState.asStateFlow()
    
    // Benchmark comparison state
    private val _benchmarkState = MutableStateFlow<BenchmarkState>(BenchmarkState.Idle)
    val benchmarkState: StateFlow<BenchmarkState> = _benchmarkState.asStateFlow()
    
    // UI state
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // Selected image bitmap
    private val _selectedImageBitmap = MutableStateFlow<Bitmap?>(null)
    val selectedImageBitmap: StateFlow<Bitmap?> = _selectedImageBitmap.asStateFlow()
    
    // Model status
    private val _modelStatus = MutableStateFlow(Pair(false, false)) // (original, quantized)
    val modelStatus: StateFlow<Pair<Boolean, Boolean>> = _modelStatus.asStateFlow()
    
    /**
     * Load both ExecuTorch models
     */
    fun loadModels() {
        viewModelScope.launch {
            _modelLoadState.value = ModelLoadState.Loading
            
            val result = modelRepository.loadModels()
            if (result.isSuccess) {
                val (originalLoaded, quantizedLoaded) = result.getOrThrow()
                _modelStatus.value = Pair(originalLoaded, quantizedLoaded)
                
                if (originalLoaded || quantizedLoaded) {
                    _modelLoadState.value = ModelLoadState.Loaded
                } else {
                    _modelLoadState.value = ModelLoadState.Error("Failed to load any models")
                }
            } else {
                _modelLoadState.value = ModelLoadState.Error(
                    result.exceptionOrNull()?.message ?: "Unknown error"
                )
            }
        }
    }
    
    /**
     * Run benchmark comparison between both models
     */
    fun runBenchmarkComparison(useImage: Boolean = false) {
        if (!isAnyModelLoaded()) {
            _benchmarkState.value = BenchmarkState.Error("No models loaded. Please load models first.")
            return
        }
        
        viewModelScope.launch {
            _benchmarkState.value = BenchmarkState.Processing
            
            val bitmap = if (useImage) _selectedImageBitmap.value else null
            val result = modelRepository.runBenchmarkComparison(bitmap)
            
            _benchmarkState.value = if (result.isSuccess) {
                BenchmarkState.Success(result.getOrThrow())
            } else {
                BenchmarkState.Error(result.exceptionOrNull()?.message ?: "Unknown error")
            }
        }
    }
    
    /**
     * Run inference with random data
     */
    fun runRandomInference() {
        if (!modelRepository.isModelLoaded()) {
            _inferenceState.value = InferenceState.Error("Model not loaded yet. Please load models first.")
            return
        }
        
        viewModelScope.launch {
            _inferenceState.value = InferenceState.Processing
            
            val result = modelRepository.runRandomInference()
            _inferenceState.value = if (result.isSuccess) {
                InferenceState.Success(result.getOrThrow())
            } else {
                InferenceState.Error(result.exceptionOrNull()?.message ?: "Unknown error")
            }
        }
    }
    
    /**
     * Run dual model classification and show separate results for each model
     */
    fun runDualImageClassification(bitmap: Bitmap) {
        if (!modelRepository.isModelLoaded()) {
            _inferenceState.value = InferenceState.Error("Model not loaded yet. Please load models first.")
            return
        }

        viewModelScope.launch {
            _inferenceState.value = InferenceState.Processing

            // If both models are available, run on both and show individual results
            if (areBothModelsLoaded()) {
                val benchmarkResult = modelRepository.runBenchmarkComparison(bitmap)
                if (benchmarkResult.isSuccess) {
                    _benchmarkState.value = BenchmarkState.Success(benchmarkResult.getOrThrow())
                } else {
                    _benchmarkState.value = BenchmarkState.Error(
                        benchmarkResult.exceptionOrNull()?.message ?: "Benchmark failed"
                    )
                }
            } else {
                // Single model fallback
                val result = modelRepository.runImageInference(bitmap)
                _inferenceState.value = if (result.isSuccess) {
                    InferenceState.Success(result.getOrThrow())
                } else {
                    InferenceState.Error(result.exceptionOrNull()?.message ?: "Unknown error")
                }
            }
        }
    }
    
    /**
     * Set selected image bitmap
     */
    fun setSelectedImage(bitmap: Bitmap?) {
        _selectedImageBitmap.value = bitmap
    }
    
    /**
     * Toggle model section minimize state
     */
    fun toggleModelSectionMinimized() {
        _uiState.value = _uiState.value.copy(
            isModelSectionMinimized = !_uiState.value.isModelSectionMinimized
        )
    }
    
    /**
     * Clear inference result
     */
    fun clearInferenceResult() {
        _inferenceState.value = InferenceState.Idle
    }
    
    /**
     * Clear benchmark result
     */
    fun clearBenchmarkResult() {
        _benchmarkState.value = BenchmarkState.Idle
    }
    
    /**
     * Get model load status text
     */
    fun getModelLoadStatusText(): String {
        return when (val state = _modelLoadState.value) {
            is ModelLoadState.NotLoaded -> "Models not loaded yet"
            is ModelLoadState.Loading -> "Loading models..."
            is ModelLoadState.Loaded -> {
                val (original, quantized) = _modelStatus.value
                when {
                    original && quantized -> "✅ Both models loaded successfully!"
                    original -> "✅ Original model loaded"
                    quantized -> "✅ Quantized model loaded"
                    else -> "❌ No models available"
                }
            }
            is ModelLoadState.Error -> "❌ ${state.message}"
        }
    }
    
    /**
     * Get model load status for minimized view
     */
    fun getModelLoadStatusForMinimized(): Pair<String, String> {
        return when (val state = _modelLoadState.value) {
            is ModelLoadState.NotLoaded -> "⏳" to "Not Loaded"
            is ModelLoadState.Loading -> "⏳" to "Loading..."
            is ModelLoadState.Loaded -> {
                val (original, quantized) = _modelStatus.value
                when {
                    original && quantized -> "✅" to "Both Loaded"
                    original -> "🟡" to "Original Only"
                    quantized -> "🟡" to "Quantized Only"
                    else -> "❌" to "None Loaded"
                }
            }
            is ModelLoadState.Error -> "❌" to "Load Failed"
        }
    }
    
    /**
     * Check if any model is loaded
     */
    fun isAnyModelLoaded(): Boolean {
        return _modelLoadState.value is ModelLoadState.Loaded
    }
    
    /**
     * Check if both models are loaded
     */
    fun areBothModelsLoaded(): Boolean {
        val (original, quantized) = _modelStatus.value
        return _modelLoadState.value is ModelLoadState.Loaded && original && quantized
    }
    
    /**
     * Check if model is loaded (for backwards compatibility)
     */
    fun isModelLoaded(): Boolean = isAnyModelLoaded()
    
    /**
     * Check if processing
     */
    fun isProcessing(): Boolean {
        return _modelLoadState.value is ModelLoadState.Loading || 
               _inferenceState.value is InferenceState.Processing ||
               _benchmarkState.value is BenchmarkState.Processing
    }
    
    /**
     * Get available models description
     */
    fun getAvailableModelsDescription(): String {
        val (original, quantized) = _modelStatus.value
        return when {
            original && quantized -> "Original FP32 & Quantized INT8"
            original -> "Original FP32 only"
            quantized -> "Quantized INT8 only"
            else -> "No models loaded"
        }
    }
}

/**
 * Sealed class for benchmark comparison state
 */
sealed class BenchmarkState {
    object Idle : BenchmarkState()
    object Processing : BenchmarkState()
    data class Success(val result: BenchmarkComparisonResult) : BenchmarkState()
    data class Error(val message: String) : BenchmarkState()
}

/**
 * Factory for creating MainViewModel
 */
class MainViewModelFactory(private val context: Context) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            return MainViewModel(ModelRepository(context)) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}