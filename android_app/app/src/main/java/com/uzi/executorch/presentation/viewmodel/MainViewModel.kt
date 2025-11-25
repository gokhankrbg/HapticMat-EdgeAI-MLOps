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
 * ViewModel for the main screen
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
    
    // UI state
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // Selected image bitmap
    private val _selectedImageBitmap = MutableStateFlow<Bitmap?>(null)
    val selectedImageBitmap: StateFlow<Bitmap?> = _selectedImageBitmap.asStateFlow()
    
    /**
     * Load the ExecuTorch model
     */
    fun loadModel() {
        viewModelScope.launch {
            _modelLoadState.value = ModelLoadState.Loading
            
            val result = modelRepository.loadModel()
            _modelLoadState.value = if (result.isSuccess) {
                ModelLoadState.Loaded
            } else {
                ModelLoadState.Error(result.exceptionOrNull()?.message ?: "Unknown error")
            }
        }
    }
    
    /**
     * Run inference with random data
     */
    fun runRandomInference() {
        if (!modelRepository.isModelLoaded()) {
            _inferenceState.value = InferenceState.Error("Model not loaded yet. Please load model first.")
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
     * Run inference with image data
     */
    fun runImageInference(bitmap: Bitmap) {
        if (!modelRepository.isModelLoaded()) {
            _inferenceState.value = InferenceState.Error("Model not loaded yet. Please load model first.")
            return
        }
        
        viewModelScope.launch {
            _inferenceState.value = InferenceState.Processing
            
            val result = modelRepository.runImageInference(bitmap)
            _inferenceState.value = if (result.isSuccess) {
                InferenceState.Success(result.getOrThrow())
            } else {
                InferenceState.Error(result.exceptionOrNull()?.message ?: "Unknown error")
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
     * Get model load status text
     */
    fun getModelLoadStatusText(): String {
        return when (val state = _modelLoadState.value) {
            is ModelLoadState.NotLoaded -> "Model not loaded yet"
            is ModelLoadState.Loading -> "Loading model..."
            is ModelLoadState.Loaded -> "✅ Model loaded successfully!"
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
            is ModelLoadState.Loaded -> "✅" to "Model Loaded"
            is ModelLoadState.Error -> "❌" to "Load Failed"
        }
    }
    
    /**
     * Check if model is loaded
     */
    fun isModelLoaded(): Boolean {
        return _modelLoadState.value is ModelLoadState.Loaded
    }
    
    /**
     * Check if processing
     */
    fun isProcessing(): Boolean {
        return _modelLoadState.value is ModelLoadState.Loading || 
               _inferenceState.value is InferenceState.Processing
    }
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
