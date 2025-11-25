package com.uzi.lemon

import android.content.Context
import com.uzi.lemon.core.*

/**
 * 🍋 Lemon - ExecuTorch Performance Evaluation Framework
 * 
 * Main entry point for the Lemon library.
 * Provides convenient factory methods for creating evaluators.
 * 
 * Example usage:
 * ```
 * val evaluator = Lemon.create(context)
 *     .latency()
 *     .memory()
 *     .throughput()
 *     .build()
 * 
 * val result = evaluator.evaluate(modelPath, inputs)
 * result.printReport()
 * ```
 */
object Lemon {
    
    /**
     * Library version
     */
    const val VERSION = "1.0.0"
    
    /**
     * Create a new evaluator builder
     * 
     * @param context Android context
     * @return LemonBuilder for fluent API
     */
    fun create(context: Context): LemonBuilder {
        return LemonBuilder(context)
    }
    
    /**
     * Create an evaluator with default configuration
     * Includes latency, throughput, and memory metrics
     * 
     * @param context Android context
     * @return ModelEvaluator with default configuration
     */
    fun createDefault(context: Context): ModelEvaluator {
        val config = PerformanceConfig.builder()
            .addMetrics(MetricType.LATENCY, MetricType.THROUGHPUT, MetricType.MEMORY)
            .iterations(100)
            .warmup(10)
            .build()
        
        return ModelEvaluator(context, config)
    }
    
    /**
     * Create an evaluator with all metrics enabled
     * 
     * @param context Android context
     * @return ModelEvaluator with all metrics
     */
    fun createComplete(context: Context): ModelEvaluator {
        val config = PerformanceConfig.builder()
            .addMetrics(
                MetricType.LATENCY,
                MetricType.THROUGHPUT,
                MetricType.MEMORY,
                MetricType.MODEL_SIZE
            )
            .iterations(100)
            .warmup(10)
            .build()
        
        return ModelEvaluator(context, config)
    }
    
    /**
     * Fluent builder for creating ModelEvaluator
     */
    class LemonBuilder(private val context: Context) {
        private val configBuilder = PerformanceConfig.builder()
        
        fun latency() = apply { configBuilder.addMetric(MetricType.LATENCY) }
        fun memory() = apply { configBuilder.addMetric(MetricType.MEMORY) }
        fun throughput() = apply { configBuilder.addMetric(MetricType.THROUGHPUT) }
        fun modelSize() = apply { configBuilder.addMetric(MetricType.MODEL_SIZE) }
        
        fun iterations(n: Int) = apply { configBuilder.iterations(n) }
        fun warmup(n: Int) = apply { configBuilder.warmup(n) }
        fun backend(type: BackendType) = apply { configBuilder.backend(type) }
        fun profiling(enabled: Boolean) = apply { configBuilder.profiling(enabled) }
        
        fun build(): ModelEvaluator {
            return ModelEvaluator(context, configBuilder.build())
        }
    }
}
