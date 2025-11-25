package com.uzi.lemon.metrics

import com.uzi.lemon.core.EvaluableModule
import com.uzi.lemon.core.Metric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import kotlinx.serialization.Serializable
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.abs

/**
 * Result of latency measurement with statistical validity
 */
@Serializable
data class LatencyResult(
    val mean: Double,
    val median: Double,
    val p50: Double,
    val p95: Double,
    val p99: Double,
    val min: Double,
    val max: Double,
    val stdDev: Double,
    val confidenceInterval95: ConfidenceInterval,
    val coefficientOfVariation: Double,
    val outliersRemoved: Int,
    val totalSamples: Int,
    override val unit: String = "ms"
) : Metric {
    override val name: String = "Latency"
    
    override fun toString(): String {
        return """
            Latency Statistics:
              Mean: ${"%.3f".format(mean)} $unit (95% CI: [${confidenceInterval95.lower.format()}, ${confidenceInterval95.upper.format()}])
              Median: ${"%.3f".format(median)} $unit
              P95: ${"%.3f".format(p95)} $unit
              P99: ${"%.3f".format(p99)} $unit
              Min: ${"%.3f".format(min)} $unit
              Max: ${"%.3f".format(max)} $unit
              StdDev: ${"%.3f".format(stdDev)} $unit
              CV: ${"%.2f".format(coefficientOfVariation)}%
              Outliers: $outliersRemoved / $totalSamples
        """.trimIndent()
    }
    
    private fun Double.format() = "%.3f".format(this)
}

/**
 * 95% Confidence Interval
 */
@Serializable
data class ConfidenceInterval(
    val lower: Double,
    val upper: Double,
    val marginOfError: Double
)

/**
 * Measures model inference latency with statistical validity
 * 
 * Features:
 * - 95% confidence intervals
 * - Outlier detection and removal (3-sigma rule)
 * - Coefficient of variation
 * - Interpolated percentiles
 * 
 * @param iterations Number of inference iterations to measure
 * @param warmupIterations Number of warmup iterations before measurement
 * @param removeOutliers Whether to remove statistical outliers (3-sigma rule)
 */
class LatencyMetric(
    private val iterations: Int = 100,
    private val warmupIterations: Int = 10,
    private val removeOutliers: Boolean = true
) {
    
    /**
     * Measure latency for the given module and inputs
     * 
     * @param module The evaluable module to measure
     * @param inputs List of input arrays to test with
     * @return LatencyResult containing statistics with confidence intervals
     */
    suspend fun measure(
        module: EvaluableModule,
        inputs: List<Array<EValue>>
    ): LatencyResult = withContext(Dispatchers.Default) {
        
        require(inputs.isNotEmpty()) { "Inputs list cannot be empty" }
        
        // Warmup phase
        repeat(warmupIterations) {
            val input = inputs.random()
            module.forward(*input)
        }
        
        // Actual measurement
        val latencies = mutableListOf<Long>()
        
        repeat(iterations) {
            val input = inputs[it % inputs.size]
            
            val startTime = System.nanoTime()
            module.forward(*input)
            val endTime = System.nanoTime()
            
            latencies.add(endTime - startTime)
        }
        
        // Calculate statistics with outlier removal
        calculateStatistics(latencies)
    }
    
    private fun calculateStatistics(rawLatencies: List<Long>): LatencyResult {
        val totalSamples = rawLatencies.size
        
        // Convert to milliseconds
        val latenciesMs = rawLatencies.map { it / 1_000_000.0 }
        
        // Step 1: Detect and remove outliers using 3-sigma rule
        val (cleanLatencies, outliersRemoved) = if (removeOutliers) {
            removeOutliers3Sigma(latenciesMs)
        } else {
            Pair(latenciesMs, 0)
        }
        
        val sorted = cleanLatencies.sorted()
        val mean = cleanLatencies.average()
        val stdDev = calculateStdDev(cleanLatencies, mean)
        
        // Step 2: Calculate 95% confidence interval
        val confidenceInterval = calculate95ConfidenceInterval(mean, stdDev, cleanLatencies.size)
        
        // Step 3: Calculate coefficient of variation
        val cv = (stdDev / mean) * 100.0
        
        // Step 4: Interpolated percentiles
        val p50 = interpolatePercentile(sorted, 0.50)
        val p95 = interpolatePercentile(sorted, 0.95)
        val p99 = interpolatePercentile(sorted, 0.99)
        
        return LatencyResult(
            mean = mean,
            median = p50,
            p50 = p50,
            p95 = p95,
            p99 = p99,
            min = sorted.first(),
            max = sorted.last(),
            stdDev = stdDev,
            confidenceInterval95 = confidenceInterval,
            coefficientOfVariation = cv,
            outliersRemoved = outliersRemoved,
            totalSamples = totalSamples
        )
    }
    
    /**
     * Remove outliers using 3-sigma rule
     * Data points beyond 3 standard deviations from mean are considered outliers
     */
    private fun removeOutliers3Sigma(data: List<Double>): Pair<List<Double>, Int> {
        if (data.size < 10) return Pair(data, 0) // Need enough samples
        
        val mean = data.average()
        val stdDev = calculateStdDev(data, mean)
        val threshold = 3.0 * stdDev
        
        val cleaned = data.filter { abs(it - mean) <= threshold }
        val removed = data.size - cleaned.size
        
        return Pair(cleaned, removed)
    }
    
    /**
     * Calculate 95% confidence interval using t-distribution
     * CI = mean ± (t * SE), where SE = stdDev / sqrt(n)
     */
    private fun calculate95ConfidenceInterval(
        mean: Double, 
        stdDev: Double, 
        n: Int
    ): ConfidenceInterval {
        // Standard error
        val se = stdDev / sqrt(n.toDouble())
        
        // t-value for 95% confidence (approximation for large n)
        // For n > 30, t ≈ 1.96 (z-value)
        // For smaller n, use conservative estimates
        val tValue = when {
            n >= 30 -> 1.96
            n >= 20 -> 2.086
            n >= 10 -> 2.262
            else -> 3.0 // Very conservative for small samples
        }
        
        val marginOfError = tValue * se
        
        return ConfidenceInterval(
            lower = mean - marginOfError,
            upper = mean + marginOfError,
            marginOfError = marginOfError
        )
    }
    
    /**
     * Interpolated percentile calculation
     * More accurate than nearest-neighbor approach
     */
    private fun interpolatePercentile(sorted: List<Double>, percentile: Double): Double {
        require(percentile in 0.0..1.0) { "Percentile must be between 0 and 1" }
        
        if (sorted.isEmpty()) return 0.0
        if (sorted.size == 1) return sorted[0]
        
        // Calculate position with linear interpolation
        val position = percentile * (sorted.size - 1)
        val lower = position.toInt()
        val upper = minOf(lower + 1, sorted.size - 1)
        val fraction = position - lower
        
        // Linear interpolation between two nearest values
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }
    
    private fun calculateStdDev(values: List<Double>, mean: Double): Double {
        if (values.size < 2) return 0.0
        val variance = values.map { (it - mean).pow(2) }.sum() / (values.size - 1)
        return sqrt(variance)
    }
}
