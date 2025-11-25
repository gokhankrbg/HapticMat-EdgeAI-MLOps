package com.uzi.lemon.core

import java.io.File

/**
 * Comparison report between two evaluation results
 * 
 * This class provides detailed comparison between a baseline model
 * and an optimized model, calculating improvement percentages and speedups.
 * 
 * @param baseline Baseline evaluation result
 * @param optimized Optimized evaluation result
 */
data class ComparisonReport(
    val baseline: EvaluationResult,
    val optimized: EvaluationResult
) {
    
    /**
     * Calculate latency speedup (how many times faster)
     */
    val latencySpeedup: Double? = baseline.latency?.mean?.let { base ->
        optimized.latency?.mean?.let { opt ->
            base / opt
        }
    }
    
    /**
     * Calculate memory savings percentage
     */
    val memorySavings: Double? = baseline.memory?.peakPss?.let { base ->
        optimized.memory?.peakPss?.let { opt ->
            ((base - opt).toDouble() / base) * 100
        }
    }
    
    /**
     * Calculate model size reduction percentage
     */
    val modelSizeReduction: Double? = baseline.modelSize?.let { base ->
        optimized.modelSize?.let { opt ->
            ((base - opt).toDouble() / base) * 100
        }
    }
    
    /**
     * Calculate throughput improvement percentage
     */
    val throughputImprovement: Double? = baseline.throughput?.samplesPerSecond?.let { base ->
        optimized.throughput?.samplesPerSecond?.let { opt ->
            ((opt - base) / base) * 100
        }
    }
    
    /**
     * Print a formatted comparison summary
     */
    fun printSummary() {
        println("""
            ====================================
            🍋 Lemon Comparison Report
            ====================================
            Baseline:  ${baseline.modelPath} (${baseline.backend})
            Optimized: ${optimized.modelPath} (${optimized.backend})
            ------------------------------------
            
            📈 Performance Improvements:
            ${latencySpeedup?.let { "  ⚡ Latency: ${"%.2f".format(it)}x faster" } ?: ""}
            ${throughputImprovement?.let { "  🚀 Throughput: ${"%+.1f".format(it)}% improvement" } ?: ""}
            ${memorySavings?.let { "  💾 Memory: ${"%.1f".format(it)}% reduction" } ?: ""}
            ${modelSizeReduction?.let { "  📦 Model Size: ${"%.1f".format(it)}% smaller" } ?: ""}
            
            ------------------------------------
            
            📊 Detailed Metrics:
            
            Baseline:
              ${baseline.latency?.let { "Latency: ${"%.2f".format(it.mean)} ms" } ?: ""}
              ${baseline.throughput?.let { "Throughput: ${"%.1f".format(it.samplesPerSecond)} samples/s" } ?: ""}
              ${baseline.memory?.let { "Memory: ${it.peakPss / 1024} MB" } ?: ""}
              ${baseline.modelSize?.let { "Model Size: ${it / 1024 / 1024} MB" } ?: ""}
            
            Optimized:
              ${optimized.latency?.let { "Latency: ${"%.2f".format(it.mean)} ms" } ?: ""}
              ${optimized.throughput?.let { "Throughput: ${"%.1f".format(it.samplesPerSecond)} samples/s" } ?: ""}
              ${optimized.memory?.let { "Memory: ${it.peakPss / 1024} MB" } ?: ""}
              ${optimized.modelSize?.let { "Model Size: ${it / 1024 / 1024} MB" } ?: ""}
            
            ====================================
        """.trimIndent())
    }
    
    /**
     * Export comparison report to Markdown file
     * 
     * @param outputPath Path to output markdown file
     */
    fun exportToMarkdown(outputPath: String) {
        val markdown = buildString {
            appendLine("# 🍋 Lemon Model Performance Comparison")
            appendLine()
            appendLine("## Configuration")
            appendLine("- **Baseline Model**: ${baseline.modelPath}")
            appendLine("- **Baseline Backend**: ${baseline.backend}")
            appendLine("- **Optimized Model**: ${optimized.modelPath}")
            appendLine("- **Optimized Backend**: ${optimized.backend}")
            appendLine()
            appendLine("## Performance Summary")
            appendLine()
            appendLine("| Metric | Baseline | Optimized | Improvement |")
            appendLine("|--------|----------|-----------|-------------|")
            
            // Latency row
            baseline.latency?.let { baseLatency ->
                optimized.latency?.let { optLatency ->
                    appendLine("| Latency (mean) | ${"%.2f".format(baseLatency.mean)} ms | ${"%.2f".format(optLatency.mean)} ms | ${"%.2f".format(latencySpeedup ?: 0.0)}x faster |")
                }
            }
            
            // Throughput row
            baseline.throughput?.let { baseThroughput ->
                optimized.throughput?.let { optThroughput ->
                    appendLine("| Throughput | ${"%.1f".format(baseThroughput.samplesPerSecond)} samples/s | ${"%.1f".format(optThroughput.samplesPerSecond)} samples/s | ${"%+.1f".format(throughputImprovement ?: 0.0)}% |")
                }
            }
            
            // Memory row
            baseline.memory?.let { baseMem ->
                optimized.memory?.let { optMem ->
                    appendLine("| Memory (peak) | ${baseMem.peakPss / 1024} MB | ${optMem.peakPss / 1024} MB | ${"%.1f".format(memorySavings ?: 0.0)}% reduction |")
                }
            }
            
            // Model size row
            baseline.modelSize?.let { baseSize ->
                optimized.modelSize?.let { optSize ->
                    appendLine("| Model Size | ${baseSize / 1024 / 1024} MB | ${optSize / 1024 / 1024} MB | ${"%.1f".format(modelSizeReduction ?: 0.0)}% smaller |")
                }
            }
            
            appendLine()
            appendLine("---")
            appendLine()
            appendLine("*Generated by Lemon 🍋 - ExecuTorch Performance Evaluation Framework*")
        }
        
        File(outputPath).writeText(markdown)
    }
}
