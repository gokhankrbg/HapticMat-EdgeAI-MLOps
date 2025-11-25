package com.uzi.lemon.reporter

import com.uzi.lemon.core.ComparisonReport
import com.uzi.lemon.core.EvaluationResult
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Generates various report formats from evaluation results
 * 
 * This class provides utilities to export evaluation results and comparisons
 * in different formats suitable for analysis, documentation, and publication.
 */
class ReportGenerator {
    
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
    }
    
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    
    /**
     * Export single evaluation result to JSON
     * 
     * @param result Evaluation result to export
     * @param outputPath Path to output JSON file
     */
    fun exportToJson(result: EvaluationResult, outputPath: String) {
        val jsonString = json.encodeToString(result)
        File(outputPath).writeText(jsonString)
    }
    
    /**
     * Export multiple evaluation results to JSON array
     * 
     * @param results List of evaluation results
     * @param outputPath Path to output JSON file
     */
    fun exportMultipleToJson(results: List<EvaluationResult>, outputPath: String) {
        val jsonString = json.encodeToString(results)
        File(outputPath).writeText(jsonString)
    }
    
    /**
     * Export evaluation result to CSV
     * 
     * @param result Evaluation result to export
     * @param outputPath Path to output CSV file
     * @param includeHeader Whether to include CSV header row
     */
    fun exportToCsv(result: EvaluationResult, outputPath: String, includeHeader: Boolean = true) {
        val csv = buildString {
            if (includeHeader) {
                appendLine("Timestamp,Model,Backend,LatencyMean,LatencyMedian,LatencyP95,LatencyP99,ThroughputSPS,MemoryPeakPSS,MemoryNativeHeap,ModelSize")
            }
            
            append(result.timestamp).append(",")
            append(result.modelPath).append(",")
            append(result.backend).append(",")
            append(result.latency?.mean ?: "").append(",")
            append(result.latency?.median ?: "").append(",")
            append(result.latency?.p95 ?: "").append(",")
            append(result.latency?.p99 ?: "").append(",")
            append(result.throughput?.samplesPerSecond ?: "").append(",")
            append(result.memory?.peakPss ?: "").append(",")
            append(result.memory?.nativeHeap ?: "").append(",")
            appendLine(result.modelSize ?: "")
        }
        
        File(outputPath).writeText(csv)
    }
    
    /**
     * Export multiple results to CSV
     * 
     * @param results List of evaluation results
     * @param outputPath Path to output CSV file
     */
    fun exportMultipleToCsv(results: List<EvaluationResult>, outputPath: String) {
        val csv = buildString {
            // Header
            appendLine("Timestamp,Model,Backend,LatencyMean,LatencyMedian,LatencyP95,LatencyP99,ThroughputSPS,MemoryPeakPSS,MemoryNativeHeap,ModelSize")
            
            // Data rows
            results.forEach { result ->
                append(result.timestamp).append(",")
                append(result.modelPath).append(",")
                append(result.backend).append(",")
                append(result.latency?.mean ?: "").append(",")
                append(result.latency?.median ?: "").append(",")
                append(result.latency?.p95 ?: "").append(",")
                append(result.latency?.p99 ?: "").append(",")
                append(result.throughput?.samplesPerSecond ?: "").append(",")
                append(result.memory?.peakPss ?: "").append(",")
                append(result.memory?.nativeHeap ?: "").append(",")
                appendLine(result.modelSize ?: "")
            }
        }
        
        File(outputPath).writeText(csv)
    }
    
    /**
     * Generate HTML report from evaluation result
     * 
     * @param result Evaluation result
     * @param outputPath Path to output HTML file
     */
    fun exportToHtml(result: EvaluationResult, outputPath: String) {
        val html = buildString {
            appendLine("<!DOCTYPE html>")
            appendLine("<html>")
            appendLine("<head>")
            appendLine("    <meta charset='UTF-8'>")
            appendLine("    <title>🍋 Lemon Evaluation Report</title>")
            appendLine("    <style>")
            appendLine("        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }")
            appendLine("        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }")
            appendLine("        h1 { color: #CDDC39; }")
            appendLine("        h2 { color: #333; border-bottom: 2px solid #CDDC39; padding-bottom: 10px; }")
            appendLine("        .metric { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }")
            appendLine("        .metric-title { font-weight: bold; color: #555; margin-bottom: 10px; }")
            appendLine("        .metric-value { font-size: 24px; color: #CDDC39; font-weight: bold; }")
            appendLine("        .metric-detail { color: #777; margin-top: 5px; }")
            appendLine("        .info { color: #666; font-size: 14px; }")
            appendLine("    </style>")
            appendLine("</head>")
            appendLine("<body>")
            appendLine("    <div class='container'>")
            appendLine("        <h1>🍋 Lemon Evaluation Report</h1>")
            appendLine("        <p class='info'>Generated: ${dateFormat.format(Date(result.timestamp))}</p>")
            appendLine("        <p class='info'>Model: ${result.modelPath}</p>")
            appendLine("        <p class='info'>Backend: ${result.backend}</p>")
            
            // Latency section
            result.latency?.let { latency ->
                appendLine("        <h2>⚡ Latency</h2>")
                appendLine("        <div class='metric'>")
                appendLine("            <div class='metric-title'>Mean Latency</div>")
                appendLine("            <div class='metric-value'>${"%.2f".format(latency.mean)} ms</div>")
                appendLine("            <div class='metric-detail'>Median: ${"%.2f".format(latency.median)} ms | P95: ${"%.2f".format(latency.p95)} ms | P99: ${"%.2f".format(latency.p99)} ms</div>")
                appendLine("        </div>")
            }
            
            // Throughput section
            result.throughput?.let { throughput ->
                appendLine("        <h2>🚀 Throughput</h2>")
                appendLine("        <div class='metric'>")
                appendLine("            <div class='metric-title'>Samples Per Second</div>")
                appendLine("            <div class='metric-value'>${"%.1f".format(throughput.samplesPerSecond)}</div>")
                appendLine("        </div>")
            }
            
            // Memory section
            result.memory?.let { memory ->
                appendLine("        <h2>💾 Memory</h2>")
                appendLine("        <div class='metric'>")
                appendLine("            <div class='metric-title'>Peak Memory Usage</div>")
                appendLine("            <div class='metric-value'>${memory.peakPss / 1024} MB</div>")
                appendLine("            <div class='metric-detail'>Native Heap: ${memory.nativeHeap / 1024 / 1024} MB | Java Heap: ${memory.javaHeap / 1024 / 1024} MB</div>")
                appendLine("        </div>")
            }
            
            // Model size section
            result.modelSize?.let { size ->
                appendLine("        <h2>📦 Model Size</h2>")
                appendLine("        <div class='metric'>")
                appendLine("            <div class='metric-value'>${size / 1024 / 1024} MB</div>")
                appendLine("        </div>")
            }
            
            appendLine("    </div>")
            appendLine("</body>")
            appendLine("</html>")
        }
        
        File(outputPath).writeText(html)
    }
    
    /**
     * Generate a summary text report
     * 
     * @param result Evaluation result
     * @return Formatted text summary
     */
    fun generateTextSummary(result: EvaluationResult): String {
        return buildString {
            appendLine("=" .repeat(50))
            appendLine("🍋 LEMON EVALUATION SUMMARY")
            appendLine("=" .repeat(50))
            appendLine()
            appendLine("Model: ${result.modelPath}")
            appendLine("Backend: ${result.backend}")
            appendLine("Timestamp: ${dateFormat.format(Date(result.timestamp))}")
            appendLine()
            
            result.latency?.let {
                appendLine("⚡ LATENCY")
                appendLine("  Mean: ${"%.2f".format(it.mean)} ms")
                appendLine("  P95: ${"%.2f".format(it.p95)} ms")
                appendLine()
            }
            
            result.throughput?.let {
                appendLine("🚀 THROUGHPUT")
                appendLine("  ${"%.1f".format(it.samplesPerSecond)} samples/sec")
                appendLine()
            }
            
            result.memory?.let {
                appendLine("💾 MEMORY")
                appendLine("  Peak: ${it.peakPss / 1024} MB")
                appendLine()
            }
            
            result.modelSize?.let {
                appendLine("📦 MODEL SIZE")
                appendLine("  ${it / 1024 / 1024} MB")
                appendLine()
            }
            
            appendLine("=" .repeat(50))
        }
    }
    
    /**
     * Generate a comparison report suitable for academic papers
     * 
     * @param comparison Comparison report
     * @return LaTeX-formatted table
     */
    fun generateLatexTable(comparison: ComparisonReport): String {
        return buildString {
            appendLine("\\begin{table}[h]")
            appendLine("\\centering")
            appendLine("\\caption{Performance Comparison: ${comparison.baseline.backend} vs ${comparison.optimized.backend}}")
            appendLine("\\begin{tabular}{lrrc}")
            appendLine("\\hline")
            appendLine("\\textbf{Metric} & \\textbf{Baseline} & \\textbf{Optimized} & \\textbf{Improvement} \\\\")
            appendLine("\\hline")
            
            comparison.baseline.latency?.let { baseLatency ->
                comparison.optimized.latency?.let { optLatency ->
                    val speedup = comparison.latencySpeedup ?: 1.0
                    appendLine("Latency (ms) & ${"%.2f".format(baseLatency.mean)} & ${"%.2f".format(optLatency.mean)} & ${"%.2f".format(speedup)}$\\times$ \\\\")
                }
            }
            
            comparison.baseline.memory?.let { baseMem ->
                comparison.optimized.memory?.let { optMem ->
                    val savings = comparison.memorySavings ?: 0.0
                    appendLine("Memory (MB) & ${baseMem.peakPss / 1024} & ${optMem.peakPss / 1024} & ${"%.1f".format(savings)}\\% \\\\")
                }
            }
            
            comparison.baseline.modelSize?.let { baseSize ->
                comparison.optimized.modelSize?.let { optSize ->
                    val reduction = comparison.modelSizeReduction ?: 0.0
                    appendLine("Model Size (MB) & ${baseSize / 1024 / 1024} & ${optSize / 1024 / 1024} & ${"%.1f".format(reduction)}\\% \\\\")
                }
            }
            
            appendLine("\\hline")
            appendLine("\\end{tabular}")
            appendLine("\\label{tab:performance_comparison}")
            appendLine("\\end{table}")
        }
    }
}
