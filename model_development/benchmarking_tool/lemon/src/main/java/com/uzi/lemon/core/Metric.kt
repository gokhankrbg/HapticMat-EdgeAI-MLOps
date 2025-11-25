package com.uzi.lemon.core

/**
 * Base interface for all metrics in Lemon evaluation framework
 */
interface Metric {
    val name: String
    val unit: String
}

/**
 * Types of metrics available in Lemon
 * 
 * Note: ENERGY metric removed - not reliable for micro-benchmarks (<10 seconds)
 * Battery APIs don't have sufficient granularity for short inference measurements
 */
enum class MetricType {
    LATENCY,
    THROUGHPUT,
    MEMORY,
    MODEL_SIZE
}
