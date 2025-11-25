package com.uzi.lemon.core

/**
 * Configuration for performance evaluation
 * 
 * @param metrics List of metrics to evaluate
 * @param iterations Number of iterations for timing metrics
 * @param warmupIterations Number of warmup iterations before measurement
 * @param backend ExecuTorch backend to use
 * @param enableProfiling Enable detailed profiling (future feature)
 */
data class PerformanceConfig(
    val metrics: List<MetricType>,
    val iterations: Int = 100,
    val warmupIterations: Int = 10,
    val backend: BackendType = BackendType.CPU,
    val enableProfiling: Boolean = false
) {
    
    /**
     * Builder for PerformanceConfig using DSL-style API
     */
    class Builder {
        private val metrics = mutableListOf<MetricType>()
        private var iterations = 100
        private var warmupIterations = 10
        private var backend = BackendType.CPU
        private var enableProfiling = false
        
        /**
         * Add a metric to evaluate
         */
        fun addMetric(metric: MetricType) = apply { 
            metrics.add(metric) 
        }
        
        /**
         * Add multiple metrics to evaluate
         */
        fun addMetrics(vararg metricTypes: MetricType) = apply {
            metrics.addAll(metricTypes)
        }
        
        /**
         * Set number of iterations
         */
        fun iterations(n: Int) = apply { 
            require(n > 0) { "Iterations must be positive" }
            iterations = n 
        }
        
        /**
         * Set number of warmup iterations
         */
        fun warmup(n: Int) = apply { 
            require(n >= 0) { "Warmup iterations must be non-negative" }
            warmupIterations = n 
        }
        
        /**
         * Set backend type
         */
        fun backend(type: BackendType) = apply { 
            backend = type 
        }
        
        /**
         * Enable or disable profiling
         */
        fun profiling(enabled: Boolean) = apply { 
            enableProfiling = enabled 
        }
        
        /**
         * Build the configuration
         */
        fun build(): PerformanceConfig {
            require(metrics.isNotEmpty()) { "At least one metric must be specified" }
            return PerformanceConfig(
                metrics = metrics.toList(),
                iterations = iterations,
                warmupIterations = warmupIterations,
                backend = backend,
                enableProfiling = enableProfiling
            )
        }
    }
    
    companion object {
        /**
         * Create a new builder
         */
        fun builder() = Builder()
    }
}
