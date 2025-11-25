package com.uzi.lemon.metrics

import com.uzi.lemon.core.EvaluableModule
import com.uzi.lemon.core.Metric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import org.pytorch.executorch.EValue

/**
 * Result of throughput measurement
 */
@Serializable
data class ThroughputResult(
    val fps: Double,                    // Frames per second (real throughput)
    val samplesPerSecond: Double,       // Same as FPS
    val averageLatencyMs: Double,       // Average latency per inference
    val totalSamples: Int,              // Total samples processed
    val totalTimeMs: Double,            // Total time taken
    override val unit: String = "FPS"
) : Metric {
    override val name: String = "Throughput"
    
    override fun toString(): String {
        return """
            Throughput Statistics:
              FPS: ${"%.2f".format(fps)}
              Samples/sec: ${"%.2f".format(samplesPerSecond)}
              Avg Latency: ${"%.2f".format(averageLatencyMs)} ms
              Total Samples: $totalSamples
              Total Time: ${"%.2f".format(totalTimeMs)} ms
        """.trimIndent()
    }
}

/**
 * Measures model inference throughput (FPS / samples per second)
 * 
 * FIXED: Now calculates real FPS based on average latency per inference,
 * not batch processing time.
 * 
 * @param iterations Number of inference iterations to measure
 */
class ThroughputMetric(
    private val iterations: Int = 100
) {
    
    /**
     * Measure throughput for the given module and inputs
     * 
     * Calculates real FPS: 1000 / average_latency_ms
     * 
     * @param module The evaluable module to measure
     * @param inputs List of input arrays to test with
     * @return ThroughputResult containing real throughput statistics
     */
    suspend fun measure(
        module: EvaluableModule,
        inputs: List<Array<EValue>>
    ): ThroughputResult = withContext(Dispatchers.Default) {
        
        require(inputs.isNotEmpty()) { "Inputs list cannot be empty" }
        
        val latencies = mutableListOf<Long>()
        val overallStartTime = System.nanoTime()
        
        // Measure individual latencies
        repeat(iterations) {
            val input = inputs[it % inputs.size]
            
            val startTime = System.nanoTime()
            module.forward(*input)
            val endTime = System.nanoTime()
            
            latencies.add(endTime - startTime)
        }
        
        val overallEndTime = System.nanoTime()
        
        // Calculate statistics
        val totalTimeMs = (overallEndTime - overallStartTime) / 1_000_000.0
        val averageLatencyMs = latencies.average() / 1_000_000.0
        
        // Real FPS calculation: 1000 / average_latency
        val fps = 1000.0 / averageLatencyMs
        
        ThroughputResult(
            fps = fps,
            samplesPerSecond = fps,
            averageLatencyMs = averageLatencyMs,
            totalSamples = iterations,
            totalTimeMs = totalTimeMs
        )
    }
}
