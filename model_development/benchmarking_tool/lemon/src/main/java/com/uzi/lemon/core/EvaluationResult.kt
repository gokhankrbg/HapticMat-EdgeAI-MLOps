package com.uzi.lemon.core

import com.uzi.lemon.metrics.LatencyResult
import com.uzi.lemon.metrics.MemoryResult
import com.uzi.lemon.metrics.ThroughputResult
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

/**
 * Result of model evaluation containing all measured metrics and system state
 * 
 * @param modelPath Path to the evaluated model
 * @param backend Backend used for evaluation
 * @param latency Latency measurement result (optional)
 * @param throughput Throughput measurement result (optional)
 * @param memory Memory measurement result (optional)
 * @param modelSize Model file size in bytes (optional)
 * @param systemState System state during benchmark (optional)
 * @param timestamp Timestamp of evaluation
 */
@Serializable
data class EvaluationResult(
    val modelPath: String,
    val backend: String,
    val latency: LatencyResult? = null,
    val throughput: ThroughputResult? = null,
    val memory: MemoryResult? = null,
    val modelSize: Long? = null,
    val systemState: SystemState? = null,
    val timestamp: Long = System.currentTimeMillis()
) {
    
    /**
     * Export evaluation result to JSON file
     * 
     * @param outputPath Path to output JSON file
     */
    fun exportToJson(outputPath: String) {
        val json = Json { 
            prettyPrint = true
            ignoreUnknownKeys = true
        }
        File(outputPath).writeText(json.encodeToString(this))
    }
    
    /**
     * Print a formatted report to console
     */
    fun printReport() {
        println("""
            ====================================
            🍋 Lemon Evaluation Report
            ====================================
            Model: $modelPath
            Backend: $backend
            Timestamp: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(timestamp)}
            ------------------------------------
            ${latency?.let { "📊 $it\n" } ?: ""}${throughput?.let { "⚡ $it\n" } ?: ""}${memory?.let { "💾 ${it.toMB()}\n" } ?: ""}${modelSize?.let { "📦 Model Size: ${it / 1024 / 1024} MB\n" } ?: ""}${systemState?.let { "\n🖥️ $it\n" } ?: ""}====================================
        """.trimIndent())
    }
    
    /**
     * Get a summary string of the evaluation
     */
    fun getSummary(): String {
        return buildString {
            append("Model: $modelPath | Backend: $backend")
            latency?.let { append(" | Latency: ${"%.2f".format(it.mean)}ms (CV: ${"%.1f".format(it.coefficientOfVariation)}%)") }
            throughput?.let { append(" | FPS: ${"%.1f".format(it.fps)}") }
            memory?.let { append(" | Memory: ${it.peakPss / 1024}MB") }
            systemState?.let { 
                append(" | Thermal: ${it.thermal.thermalState}")
                append(" | Battery: ${it.battery.level}%")
            }
        }
    }
}
