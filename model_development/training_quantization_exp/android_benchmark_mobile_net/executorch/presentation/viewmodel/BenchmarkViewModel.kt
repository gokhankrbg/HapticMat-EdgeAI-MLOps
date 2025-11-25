package com.uzi.executorch.presentation.viewmodel

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.uzi.executorch.data.model.BenchmarkResult
import com.uzi.executorch.data.repository.QuantizationBenchmarkRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for the Quantization Benchmark screen
 */
class BenchmarkViewModel(
    private val repository: QuantizationBenchmarkRepository
) : ViewModel() {
    
    // Benchmark state
    private val _benchmarkState = MutableStateFlow<MultiBenchmarkState>(MultiBenchmarkState.Idle)
    val benchmarkState: StateFlow<MultiBenchmarkState> = _benchmarkState.asStateFlow()
    
    // Selected image
    private val _selectedImageBitmap = MutableStateFlow<Bitmap?>(null)
    val selectedImageBitmap: StateFlow<Bitmap?> = _selectedImageBitmap.asStateFlow()
    
    // Processing state
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    // Available models
    private val _availableModels = MutableStateFlow(repository.getAvailableModels())
    val availableModels: StateFlow<List<ModelConfig>> = _availableModels.asStateFlow()
    
    /**
     * Run complete benchmark on all available models
     */
    fun runCompleteBenchmark(useImage: Boolean = false) {
        viewModelScope.launch {
            _isProcessing.value = true
            _benchmarkState.value = MultiBenchmarkState.Processing
            
            try {
                val bitmap = if (useImage) _selectedImageBitmap.value else null
                val results = repository.runCompleteBenchmark(bitmap)
                
                if (results.isEmpty()) {
                    _benchmarkState.value = MultiBenchmarkState.Error("No models available for benchmarking")
                } else {
                    _benchmarkState.value = MultiBenchmarkState.Success(results)
                }
            } catch (e: Exception) {
                _benchmarkState.value = MultiBenchmarkState.Error(e.message ?: "Benchmark failed")
            } finally {
                _isProcessing.value = false
            }
        }
    }
    
    /**
     * Set selected image
     */
    fun setSelectedImage(bitmap: Bitmap?) {
        _selectedImageBitmap.value = bitmap
    }
    
    /**
     * Clear benchmark results
     */
    fun clearResults() {
        _benchmarkState.value = MultiBenchmarkState.Idle
    }
}

/**
 * Sealed class for multi-model benchmark state
 */
sealed class MultiBenchmarkState {
    object Idle : MultiBenchmarkState()
    object Processing : MultiBenchmarkState()
    data class Success(val results: List<BenchmarkResult>) : MultiBenchmarkState()
    data class Error(val message: String) : MultiBenchmarkState()
}

/**
 * Model configuration
 */
data class ModelConfig(
    val fileName: String,
    val displayName: String,
    val type: ModelType
)

/**
 * Model type enumeration
 */
enum class ModelType(val displayName: String, val color: Color) {
    FP32("FP32 (Full Precision)", Color(0xFF4CAF50)),
    FP16("FP16 (Half Precision)", Color(0xFF2196F3)),
    INT8("INT8 (Quantized)", Color(0xFFFF9800)),
    XNNPACK_FP32("XNNPACK FP32", Color(0xFF9C27B0)),
    XNNPACK_INT8("XNNPACK INT8", Color(0xFFE91E63)),
    VULKAN_FP32("Vulkan FP32", Color(0xFF00BCD4)),
    VULKAN_FP16("Vulkan FP16", Color(0xFF009688)),
    PORTABLE_FP32("Portable FP32", Color(0xFFFF5722))
}

/**
 * Factory for creating BenchmarkViewModel
 */
class BenchmarkViewModelFactory(private val context: Context) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(BenchmarkViewModel::class.java)) {
            return BenchmarkViewModel(QuantizationBenchmarkRepository(context)) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
