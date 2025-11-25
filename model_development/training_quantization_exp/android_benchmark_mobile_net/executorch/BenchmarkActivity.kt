package com.uzi.executorch

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.facebook.soloader.SoLoader
import com.uzi.executorch.presentation.viewmodel.BenchmarkViewModel
import com.uzi.executorch.presentation.viewmodel.BenchmarkViewModelFactory
import com.uzi.executorch.presentation.viewmodel.MultiBenchmarkState
import com.uzi.executorch.ui.theme.ExecutorchTheme

class BenchmarkActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize SoLoader for native libraries
        SoLoader.init(this, false)
        
        enableEdgeToEdge()
        setContent {
            ExecutorchTheme {
                val viewModel: BenchmarkViewModel = viewModel(
                    factory = BenchmarkViewModelFactory(this@BenchmarkActivity)
                )
                
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    BenchmarkScreen(
                        modifier = Modifier.padding(innerPadding),
                        viewModel = viewModel
                    )
                }
            }
        }
    }
}

@Composable
fun BenchmarkScreen(
    modifier: Modifier = Modifier,
    viewModel: BenchmarkViewModel
) {
    val context = LocalContext.current
    
    // Collect states
    val benchmarkState by viewModel.benchmarkState.collectAsState()
    val selectedImageBitmap by viewModel.selectedImageBitmap.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    
    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val inputStream = context.contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                viewModel.setSelectedImage(bitmap)
            } catch (e: Exception) {
                Log.e("ImagePicker", "Failed to load image: ${e.message}")
            }
        }
    }
    
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Text(
            text = "⚡ Quantization Benchmark",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold
        )
        
        Text(
            text = "Compare different quantization strategies",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        
        // Available Models Section
        AvailableModelsSection(viewModel = viewModel)
        
        // Test Input Section
        TestInputSection(
            selectedImageBitmap = selectedImageBitmap,
            isProcessing = isProcessing,
            onPickImage = { imagePickerLauncher.launch("image/*") },
            onBirdImage = {
                // Load bird image from drawable
                val bitmap = BitmapFactory.decodeResource(
                    context.resources,
                    R.drawable.bird
                )
                viewModel.setSelectedImage(bitmap)
            },
            onRunBenchmark = { useImage ->
                viewModel.runCompleteBenchmark(useImage)
            }
        )
        
        // Loading indicator
        if (isProcessing) {
            CircularProgressIndicator(
                modifier = Modifier.padding(16.dp)
            )
            Text(
                text = "⏳ Running benchmarks...",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
        
        // Results Section
        BenchmarkResultsSection(benchmarkState = benchmarkState)
    }
}

@Composable
fun AvailableModelsSection(viewModel: BenchmarkViewModel) {
    val availableModels by viewModel.availableModels.collectAsState()
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "📋 Available Models",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            availableModels.forEach { model ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = model.displayName,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = model.type.displayName,
                        style = MaterialTheme.typography.bodySmall,
                        color = model.type.color
                    )
                }
            }
        }
    }
}

@Composable
fun TestInputSection(
    selectedImageBitmap: Bitmap?,
    isProcessing: Boolean,
    onPickImage: () -> Unit,
    onBirdImage: () -> Unit,
    onRunBenchmark: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "🧪 Benchmark Input",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Image picker buttons
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = onPickImage,
                    enabled = !isProcessing,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("📷 Pick Image")
                }
                
                Button(
                    onClick = onBirdImage,
                    enabled = !isProcessing,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("🐦 Bird Image")
                }
            }
            
            // Selected image display
            selectedImageBitmap?.let { bitmap ->
                Spacer(modifier = Modifier.height(12.dp))
                
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Selected image",
                    modifier = Modifier
                        .size(200.dp)
                        .clip(RoundedCornerShape(8.dp))
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Benchmark buttons
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = { onRunBenchmark(false) },
                    enabled = !isProcessing,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("⚡ Random Data")
                }
                
                Button(
                    onClick = { onRunBenchmark(true) },
                    enabled = !isProcessing && selectedImageBitmap != null,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("⚡ Use Image")
                }
            }
        }
    }
}

@Composable
fun BenchmarkResultsSection(benchmarkState: MultiBenchmarkState) {
    when (benchmarkState) {
        is MultiBenchmarkState.Success -> {
            val results = benchmarkState.results
            
            // Performance Summary Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFFF9800).copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "📊 Performance Summary",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFFF9800)
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Speed Rankings
                    Text(
                        text = "⚡ Speed (Inference Time)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFFF9800)
                    )
                    
                    val sortedBySpeed = results.sortedBy { it.inferenceTimeMs }
                    sortedBySpeed.take(3).forEachIndexed { index, result ->
                        val rank = when(index) {
                            0 -> "🥇"
                            1 -> "🥈"
                            2 -> "🥉"
                            else -> "${index + 1}."
                        }
                        
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 2.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "$rank ${result.modelName}",
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = "${result.inferenceTimeMs}ms",
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFFFF9800)
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Memory Rankings
                    Text(
                        text = "🧠 Memory Efficiency",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF2196F3)
                    )
                    
                    val sortedByMemory = results.sortedBy { it.memoryUsedMb }
                    sortedByMemory.take(3).forEachIndexed { index, result ->
                        val rank = when(index) {
                            0 -> "🥇"
                            1 -> "🥈"
                            2 -> "🥉"
                            else -> "${index + 1}."
                        }
                        
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 2.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "$rank ${result.modelName}",
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = "${"%.2f".format(result.memoryUsedMb)}MB",
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF2196F3)
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Energy Efficiency Rankings
                    Text(
                        text = "⚡ Energy Efficiency (Lower is Better)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF4CAF50)
                    )
                    
                    val sortedByEnergy = results.sortedBy { it.energyEfficiencyScore }
                    sortedByEnergy.take(3).forEachIndexed { index, result ->
                        val rank = when(index) {
                            0 -> "🥇"
                            1 -> "🥈"
                            2 -> "🥉"
                            else -> "${index + 1}."
                        }
                        
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 2.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "$rank ${result.modelName}",
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = "${"%.2f".format(result.energyEfficiencyScore)}",
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF4CAF50)
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Overall comparison
                    if (sortedBySpeed.size >= 2) {
                        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                        
                        Text(
                            text = "📈 Overall Comparison",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFFFF9800)
                        )
                        
                        val fastestTime = sortedBySpeed.first().inferenceTimeMs
                        val slowestTime = sortedBySpeed.last().inferenceTimeMs
                        val speedup = slowestTime.toDouble() / fastestTime.toDouble()
                        
                        Text(
                            text = "Fastest: ${sortedBySpeed.first().modelName} (${fastestTime}ms)",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFFFF9800)
                        )
                        Text(
                            text = "Slowest: ${sortedBySpeed.last().modelName} (${slowestTime}ms)",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFFFF9800)
                        )
                        Text(
                            text = "Speed Improvement: ${"%.2f".format(speedup)}x",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF4CAF50)
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Individual Results
            results.forEach { result ->
                ModelResultCard(result = result)
                Spacer(modifier = Modifier.height(12.dp))
            }
        }
        
        is MultiBenchmarkState.Error -> {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFF44336).copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "❌ Benchmark Error",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFF44336)
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = benchmarkState.message,
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFFF44336)
                    )
                }
            }
        }
        
        else -> { /* Idle or Processing - no results to show */ }
    }
}

@Composable
fun ModelResultCard(result: com.uzi.executorch.data.model.BenchmarkResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = result.modelConfig.type.color.copy(alpha = 0.1f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = result.modelName,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = result.modelConfig.type.color
                    )
                    Text(
                        text = result.modelConfig.type.displayName,
                        style = MaterialTheme.typography.bodySmall,
                        color = result.modelConfig.type.color.copy(alpha = 0.7f)
                    )
                }
                
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "${result.inferenceTimeMs}ms",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = result.modelConfig.type.color
                    )
                    Text(
                        text = "${"%.2f".format(result.memoryUsedMb)}MB",
                        style = MaterialTheme.typography.bodySmall,
                        color = result.modelConfig.type.color.copy(alpha = 0.8f)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Performance metrics
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "⚡ Speed: ${result.inferenceTimeMs}ms",
                        style = MaterialTheme.typography.bodySmall,
                        color = result.modelConfig.type.color.copy(alpha = 0.8f)
                    )
                    Text(
                        text = "🧠 Memory: ${"%.2f".format(result.memoryUsedMb)}MB",
                        style = MaterialTheme.typography.bodySmall,
                        color = result.modelConfig.type.color.copy(alpha = 0.8f)
                    )
                    Text(
                        text = "⚡ Energy: ${"%.2f".format(result.energyEfficiencyScore)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = result.modelConfig.type.color.copy(alpha = 0.8f)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Top predictions
            Text(
                text = "🏆 Top Predictions:",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.Bold,
                color = result.modelConfig.type.color
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            result.classificationResult.predictions.take(3).forEach { pred ->
                Text(
                    text = "${pred.className}: ${"%.2f".format(pred.confidencePercentage)}%",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = result.modelConfig.type.color.copy(alpha = 0.8f)
                )
            }
        }
    }
}
