package com.uzi.lemon.core

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import java.io.File

/**
 * Wrapper for ExecuTorch Module with evaluation capabilities
 * 
 * This class wraps the ExecuTorch Module and provides additional
 * functionality needed for performance evaluation.
 * 
 * @param context Android context
 * @param modelPath Path to the .pte model file
 */
class EvaluableModule(
    private val context: Context,
    private val modelPath: String
) {
    private var module: Module? = null
    private val TAG = "EvaluableModule"
    
    /**
     * Check if the module is loaded
     */
    val isLoaded: Boolean 
        get() = module != null
    
    /**
     * Load the ExecuTorch module
     * @return this instance for chaining
     */
    @Synchronized
    fun load(): EvaluableModule {
        // If already loaded, return
        if (module != null) {
            Log.d(TAG, "Module already loaded")
            return this
        }
        
        try {
            Log.d(TAG, "Loading model from: $modelPath")
            
            // Verify file exists
            val file = File(modelPath)
            if (!file.exists()) {
                throw IllegalStateException("Model file does not exist: $modelPath")
            }
            
            Log.d(TAG, "Model file size: ${file.length()} bytes")
            
            // Load the module
            module = Module.load(modelPath)
            Log.d(TAG, "Module loaded successfully")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model from $modelPath", e)
            throw IllegalStateException("Failed to load model from $modelPath: ${e.message}", e)
        }
        return this
    }
    
    /**
     * Execute forward pass
     * @param inputs Variable number of EValue inputs
     * @return Array of EValue outputs
     */
    @Synchronized
    fun forward(vararg inputs: EValue): Array<EValue> {
        val currentModule = module 
            ?: throw IllegalStateException("Module not loaded. Call load() first.")
        
        return try {
            currentModule.forward(*inputs)
        } catch (e: Exception) {
            Log.e(TAG, "Forward pass failed", e)
            throw IllegalStateException("Forward pass failed: ${e.message}", e)
        }
    }
    
    /**
     * Get the model file size in bytes
     * @return Model size in bytes, or -1 if unavailable
     */
    fun getModelSize(): Long {
        return try {
            val file = File(modelPath)
            if (file.exists()) {
                file.length()
            } else {
                // Try to get from assets
                try {
                    context.assets.openFd(modelPath).length
                } catch (e: Exception) {
                    -1L
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get model size", e)
            -1L
        }
    }
    
    /**
     * Release the module resources
     * This should be called when done with the module to free native memory
     */
    @Synchronized
    fun release() {
        try {
            module = null
            // Force garbage collection to clean up native resources
            System.gc()
            Log.d(TAG, "Module released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing module", e)
        }
    }
    
    /**
     * Ensure resources are released when the object is garbage collected
     */
    protected fun finalize() {
        release()
    }
}
