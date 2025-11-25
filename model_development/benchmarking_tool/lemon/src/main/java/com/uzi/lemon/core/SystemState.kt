package com.uzi.lemon.core

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import kotlinx.serialization.Serializable
import java.io.File
import java.io.RandomAccessFile

/**
 * System state at the time of benchmark execution
 * Critical for reproducibility and understanding performance variations
 */
@Serializable
data class SystemState(
    val device: DeviceInfo,
    val thermal: ThermalInfo,
    val cpu: CpuInfo,
    val battery: BatteryInfo,
    val memory: SystemMemoryInfo,
    val processes: ProcessInfo,
    val timestamp: Long = System.currentTimeMillis()
) {
    override fun toString(): String {
        return """
            System State Report:
            ==================
            Device: ${device.model} (${device.manufacturer})
            OS: Android ${device.androidVersion} (API ${device.sdkInt})
            
            Thermal:
              State: ${thermal.thermalState}
              Temperature: ${thermal.temperature}°C
            
            CPU:
              Cores: ${cpu.coreCount}
              Current Freq: ${cpu.currentFrequencies.joinToString(", ")} MHz
              Governor: ${cpu.governor}
            
            Battery:
              Level: ${battery.level}%
              Temperature: ${battery.temperature}°C
              Charging: ${battery.isCharging}
            
            Memory:
              Total RAM: ${memory.totalRamMB} MB
              Available: ${memory.availableRamMB} MB
              Used: ${memory.usedRamMB} MB
            
            Processes:
              Running: ${processes.runningProcessCount}
              Background: ${processes.backgroundProcessCount}
            
            Timestamp: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(timestamp)}
        """.trimIndent()
    }
}

@Serializable
data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val board: String,
    val hardware: String,
    val androidVersion: String,
    val sdkInt: Int
)

@Serializable
data class ThermalInfo(
    val thermalState: String,
    val temperature: Float
)

@Serializable
data class CpuInfo(
    val coreCount: Int,
    val currentFrequencies: List<Int>,
    val maxFrequencies: List<Int>,
    val governor: String
)

@Serializable
data class BatteryInfo(
    val level: Int,
    val temperature: Float,
    val isCharging: Boolean,
    val chargingSource: String
)

@Serializable
data class SystemMemoryInfo(
    val totalRamMB: Long,
    val availableRamMB: Long,
    val usedRamMB: Long,
    val lowMemory: Boolean
)

@Serializable
data class ProcessInfo(
    val runningProcessCount: Int,
    val backgroundProcessCount: Int
)

/**
 * Collects system state information
 */
class SystemStateCollector(private val context: Context) {
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    
    /**
     * Collect current system state
     */
    fun collect(): SystemState {
        return SystemState(
            device = collectDeviceInfo(),
            thermal = collectThermalInfo(),
            cpu = collectCpuInfo(),
            battery = collectBatteryInfo(),
            memory = collectMemoryInfo(),
            processes = collectProcessInfo()
        )
    }
    
    private fun collectDeviceInfo(): DeviceInfo {
        return DeviceInfo(
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            board = Build.BOARD,
            hardware = Build.HARDWARE,
            androidVersion = Build.VERSION.RELEASE,
            sdkInt = Build.VERSION.SDK_INT
        )
    }
    
    private fun collectThermalInfo(): ThermalInfo {
        val thermalState = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            when (powerManager.currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> "None"
                PowerManager.THERMAL_STATUS_LIGHT -> "Light"
                PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
                PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
                PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
                PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
                PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
                else -> "Unknown"
            }
        } else {
            "Not Available"
        }
        
        val temperature = readBatteryTemperature() / 10.0f // Convert to Celsius
        
        return ThermalInfo(
            thermalState = thermalState,
            temperature = temperature
        )
    }
    
    private fun collectCpuInfo(): CpuInfo {
        val coreCount = Runtime.getRuntime().availableProcessors()
        val currentFreqs = mutableListOf<Int>()
        val maxFreqs = mutableListOf<Int>()
        
        // Read CPU frequencies for each core
        for (i in 0 until coreCount) {
            currentFreqs.add(readCpuFrequency(i, "scaling_cur_freq"))
            maxFreqs.add(readCpuFrequency(i, "scaling_max_freq"))
        }
        
        val governor = readCpuGovernor()
        
        return CpuInfo(
            coreCount = coreCount,
            currentFrequencies = currentFreqs,
            maxFrequencies = maxFreqs,
            governor = governor
        )
    }
    
    private fun collectBatteryInfo(): BatteryInfo {
        val batteryStatus: Intent? = context.registerReceiver(
            null, 
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
        
        val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val batteryPct = if (level >= 0 && scale > 0) {
            (level * 100 / scale.toFloat()).toInt()
        } else {
            -1
        }
        
        val temperature = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)?.let {
            it / 10.0f
        } ?: -1f
        
        val status = batteryStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                        status == BatteryManager.BATTERY_STATUS_FULL
        
        val chargePlug = batteryStatus?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1
        val chargingSource = when (chargePlug) {
            BatteryManager.BATTERY_PLUGGED_USB -> "USB"
            BatteryManager.BATTERY_PLUGGED_AC -> "AC"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "Wireless"
            else -> "None"
        }
        
        return BatteryInfo(
            level = batteryPct,
            temperature = temperature,
            isCharging = isCharging,
            chargingSource = chargingSource
        )
    }
    
    private fun collectMemoryInfo(): SystemMemoryInfo {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        
        val totalMB = memInfo.totalMem / (1024 * 1024)
        val availableMB = memInfo.availMem / (1024 * 1024)
        val usedMB = totalMB - availableMB
        
        return SystemMemoryInfo(
            totalRamMB = totalMB,
            availableRamMB = availableMB,
            usedRamMB = usedMB,
            lowMemory = memInfo.lowMemory
        )
    }
    
    private fun collectProcessInfo(): ProcessInfo {
        val runningProcesses = activityManager.runningAppProcesses?.size ?: 0
        
        // Count background processes
        val backgroundCount = activityManager.runningAppProcesses?.count { 
            it.importance != ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND 
        } ?: 0
        
        return ProcessInfo(
            runningProcessCount = runningProcesses,
            backgroundProcessCount = backgroundCount
        )
    }
    
    // Helper functions for reading system files
    
    private fun readCpuFrequency(core: Int, type: String): Int {
        return try {
            val file = File("/sys/devices/system/cpu/cpu$core/cpufreq/$type")
            if (file.exists()) {
                file.readText().trim().toInt() / 1000 // Convert to MHz
            } else {
                -1
            }
        } catch (e: Exception) {
            -1
        }
    }
    
    private fun readCpuGovernor(): String {
        return try {
            val file = File("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if (file.exists()) {
                file.readText().trim()
            } else {
                "Unknown"
            }
        } catch (e: Exception) {
            "Unknown"
        }
    }
    
    private fun readBatteryTemperature(): Int {
        return try {
            val batteryStatus: Intent? = context.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            )
            batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        } catch (e: Exception) {
            -1
        }
    }
}
