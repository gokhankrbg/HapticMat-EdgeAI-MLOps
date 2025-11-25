package com.uzi.lemon.core

import android.content.Context
import android.util.Log
import com.uzi.lemon.metrics.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue

/**
 * Main evaluator class for ExecuTorch models
 * 
 * This is the entry point for evaluating ExecuTorch models with Lemon.
 * It orchestrates the evaluation of multiple metrics based on the provided configuration.
 * Also captures system state for reproducibility.
 * 
 * Example usage:
 * ```
 * val config = PerformanceConfig.builder()
 *     .addMetric(MetricType.LATENCY)
 *     .addMetric(MetricType.MEMORY)
 *     .iterations(100)
 *     .build()
 * 
 * val evaluator = ModelEvaluator(context, config)
 * val result = evaluator.evaluate(modelPath, inputs)
 * result.printReport()
 * ```
 * 
 * @param context Android context
 * @param config Performance evaluation configuration
 */
class ModelEvaluator(
    private val context: Context,
    private val config: PerformanceConfig
) {
    private val TAG = "ModelEvaluator"
    private val systemStateCollector = SystemStateCollector(context)
    
    /**
     * Evaluate a model with the configured metrics
     * 
     * @param modelPath Path to the .pte model file
     * @param inputs List of input arrays for evaluation
     * @return EvaluationResult containing all measured metrics and system state
     */
    suspend fun evaluate(
        modelPath: String,
        inputs: List<Array<EValue>>
    ): EvaluationResult = withContext(Dispatchers.Default) {
        
        require(inputs.isNotEmpty()) { "Inputs list cannot be empty" }
        
        Log.d(TAG, "Starting evaluation for model: $modelPath")
        Log.d(TAG, "Configuration: ${config.metrics.size} metrics, ${config.iterations} iterations, ${config.warmupIterations} warmup")
        
        // Collect system state BEFORE benchmark
        val systemState = try {
            systemStateCollector.collect()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to collect system state", e)
            null
        }
        
        systemState?.let {
            Log.d(TAG, "System State:")
            Log.d(TAG, "  Device: ${it.device.model}")
            Log.d(TAG, "  Thermal: ${it.thermal.thermalState} (${it.thermal.temperature}°C)")
            Log.d(TAG, "  Battery: ${it.battery.level}% (${if (it.battery.isCharging) "Charging" else "Not Charging"})")
            Log.d(TAG, "  CPU Governor: ${it.cpu.governor}")
            Log.d(TAG, "  Available RAM: ${it.memory.availableRamMB} MB")
        }
        
        // Load the model
        val module = EvaluableModule(context, modelPath).load()
        
        try {
            // Measure each configured metric
            val results = mutableMapOf<MetricType, Any>()
            
            config.metrics.forEach { metricType ->
                Log.d(TAG, "Measuring metric: $metricType")
                
                try {
                    when (metricType) {
                        MetricType.LATENCY -> {
                            val metric = LatencyMetric(
                                iterations = config.iterations,
                                warmupIterations = config.warmupIterations
                            )
                            results[metricType] = metric.measure(module, inputs)
                            Log.d(TAG, "Latency measured: ${(results[metricType] as LatencyResult).mean} ms")
                        }
                        
                        MetricType.MEMORY -> {
                            val metric = MemoryMetric(context)
                            results[metricType] = metric.measure(module, inputs)
                            Log.d(TAG, "Memory measured: ${(results[metricType] as MemoryResult).peakPss} KB")
                        }
                        
                        MetricType.THROUGHPUT -> {
                            val metric = ThroughputMetric(iterations = config.iterations)
                            results[metricType] = metric.measure(module, inputs)
                            Log.d(TAG, "Throughput measured: ${(results[metricType] as ThroughputResult).fps} FPS")
                        }
                        
                        MetricType.MODEL_SIZE -> {
                            val metric = ModelSizeMetric()
                            results[metricType] = metric.measure(module, inputs)
                            Log.d(TAG, "Model size measured: ${(results[metricType] as ModelSizeResult).sizeMB} MB")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to measure metric $metricType", e)
                    // Continue with other metrics
                }
            }
            
            // Build evaluation result
            val result = EvaluationResult(
                modelPath = modelPath,
                backend = config.backend.name,
                latency = results[MetricType.LATENCY] as? LatencyResult,
                memory = results[MetricType.MEMORY] as? MemoryResult,
                throughput = results[MetricType.THROUGHPUT] as? ThroughputResult,
                modelSize = (results[MetricType.MODEL_SIZE] as? ModelSizeResult)?.sizeBytes,
                systemState = systemState
            )
            
            Log.d(TAG, "Evaluation completed successfully")
            result
            
        } catch (e: Exception) {
            Log.e(TAG, "Evaluation failed", e)
            throw e
        } finally {
            // Clean up resources
            try {
                module.release()
                Log.d(TAG, "Module resources released")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing module", e)
            }
        }
    }
    
    /**
     * Compare two evaluation results
     * 
     * @param baseline Baseline evaluation result
     * @param optimized Optimized evaluation result
     * @return ComparisonReport containing the comparison
     */
    fun compare(
        baseline: EvaluationResult,
        optimized: EvaluationResult
    ): ComparisonReport {
        return ComparisonReport(baseline, optimized)
    }
}
