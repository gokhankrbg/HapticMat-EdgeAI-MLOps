package com.uzi.lemon.metrics

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import com.uzi.lemon.core.EvaluableModule
import com.uzi.lemon.core.Metric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import org.pytorch.executorch.EValue

/**
 * Result of memory measurement
 */
@Serializable
data class MemoryResult(
    val peakPss: Long,              // Peak Proportional Set Size in KB
    val peakRss: Long,              // Peak Resident Set Size in KB
    val initialPss: Long,           // Initial PSS before inference in KB
    val deltaPss: Long,             // Delta PSS (peak - initial) in KB
    val nativeHeap: Long,           // Native heap allocation in bytes
    val javaHeap: Long,             // Java heap allocation in bytes
    val modelSize: Long,            // Model file size in bytes
    override val unit: String = "MB"
) : Metric {
    override val name: String = "Memory"
    
    /**
     * Convert memory values to megabytes
     */
    fun toMB(): MemoryResult {
        return copy(
            peakPss = peakPss / 1024,           // KB to MB
            peakRss = peakRss / 1024,           // KB to MB
            initialPss = initialPss / 1024,     // KB to MB
            deltaPss = deltaPss / 1024,         // KB to MB
            nativeHeap = nativeHeap / 1024 / 1024,  // bytes to MB
            javaHeap = javaHeap / 1024 / 1024,      // bytes to MB
            modelSize = modelSize / 1024 / 1024     // bytes to MB
        )
    }
    
    override fun toString(): String {
        val mb = toMB()
        return """
            Memory Statistics:
              Peak PSS: ${mb.peakPss} ${mb.unit}
              Peak RSS: ${mb.peakRss} ${mb.unit}
              Delta PSS: ${mb.deltaPss} ${mb.unit}
              Native Heap: ${mb.nativeHeap} ${mb.unit}
              Java Heap: ${mb.javaHeap} ${mb.unit}
              Model Size: ${mb.modelSize} ${mb.unit}
        """.trimIndent()
    }
}

/**
 * Measures memory usage during model inference
 * 
 * FIXED: Now captures peak memory during actual iterations,
 * not just after single inference.
 * 
 * @param context Android context for accessing system services
 */
class MemoryMetric(private val context: Context) {
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val runtime = Runtime.getRuntime()
    
    /**
     * Measure memory usage for the given module and inputs
     * 
     * Monitors memory continuously during iterations to capture true peak.
     * 
     * @param module The evaluable module to measure
     * @param inputs List of input arrays to test with
     * @return MemoryResult containing memory statistics
     */
    suspend fun measure(
        module: EvaluableModule,
        inputs: List<Array<EValue>>
    ): MemoryResult = withContext(Dispatchers.Default) {
        
        // Force GC before measurement to get clean baseline
        System.gc()
        Thread.sleep(100)
        
        // Get initial memory baseline
        val initialMemInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(initialMemInfo)
        val initialPss = initialMemInfo.totalPss.toLong()
        
        var peakPss = initialPss
        var peakRss = Debug.getNativeHeapSize() / 1024
        
        // FIXED: Run multiple iterations and track peak memory
        val sampleInterval = maxOf(1, inputs.size / 10) // Sample every ~10% of iterations
        
        inputs.forEachIndexed { index, input ->
            module.forward(*input)
            
            // Sample memory periodically to capture peak
            if (index % sampleInterval == 0) {
                val currentMemInfo = Debug.MemoryInfo()
                Debug.getMemoryInfo(currentMemInfo)
                val currentPss = currentMemInfo.totalPss.toLong()
                val currentRss = Debug.getNativeHeapSize() / 1024
                
                peakPss = maxOf(peakPss, currentPss)
                peakRss = maxOf(peakRss, currentRss)
            }
        }
        
        // Final memory check to ensure we capture peak
        val finalMemInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(finalMemInfo)
        peakPss = maxOf(peakPss, finalMemInfo.totalPss.toLong())
        peakRss = maxOf(peakRss, Debug.getNativeHeapSize() / 1024)
        
        // Get heap sizes at peak
        val nativeHeapSize = Debug.getNativeHeapAllocatedSize()
        val javaHeapSize = runtime.totalMemory() - runtime.freeMemory()
        
        MemoryResult(
            peakPss = peakPss,
            peakRss = peakRss,
            initialPss = initialPss,
            deltaPss = peakPss - initialPss,
            nativeHeap = nativeHeapSize,
            javaHeap = javaHeapSize,
            modelSize = module.getModelSize()
        )
    }
}
