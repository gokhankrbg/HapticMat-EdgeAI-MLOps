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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
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
import com.uzi.executorch.data.model.InferenceState
import com.uzi.executorch.data.model.ModelLoadState
import com.uzi.executorch.presentation.viewmodel.MainViewModel
import com.uzi.executorch.presentation.viewmodel.MainViewModelFactory
import com.uzi.executorch.ui.theme.ExecutorchTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize SoLoader for native libraries
        SoLoader.init(this, false)
        
        enableEdgeToEdge()
        setContent {
            ExecutorchTheme {
                val viewModel: MainViewModel = viewModel(
                    factory = MainViewModelFactory(this@MainActivity)
                )
                
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    MobileNetTestScreen(
                        modifier = Modifier.padding(innerPadding),
                        viewModel = viewModel
                    )
                }
            }
        }
    }
}

@Composable
fun MobileNetTestScreen(
    modifier: Modifier = Modifier,
    viewModel: MainViewModel
) {
    val context = LocalContext.current
    
    // Collect states
    val modelLoadState by viewModel.modelLoadState.collectAsState()
    val inferenceState by viewModel.inferenceState.collectAsState()
    val uiState by viewModel.uiState.collectAsState()
    val selectedImageBitmap by viewModel.selectedImageBitmap.collectAsState()
    
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
        Text(
            text = "🤖 MobileNetV2 ExecuTorch Demo",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold
        )
        
        Text(
            text = "Test your MobileNetV2 model with random data or real images",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        
        // Model Loading Section with Minimize/Expand functionality
        ModelLoadingSection(
            modelLoadState = modelLoadState,
            isMinimized = uiState.isModelSectionMinimized,
            isProcessing = viewModel.isProcessing(),
            onLoadModel = { viewModel.loadModel() },
            onToggleMinimized = { viewModel.toggleModelSectionMinimized() }
        )
        
        // Testing Section
        if (viewModel.isModelLoaded()) {
            TestingSection(
                selectedImageBitmap = selectedImageBitmap,
                isProcessing = viewModel.isProcessing(),
                onRandomTest = { viewModel.runRandomInference() },
                onPickImage = { imagePickerLauncher.launch("image/*") },
                onClassifyImage = { bitmap -> viewModel.runImageInference(bitmap) }
            )
        }
        
        // Loading indicator
        if (viewModel.isProcessing()) {
            Text(
                text = "⏳ Processing...",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
        
        // Results section
        ResultsSection(inferenceState = inferenceState)
    }
}

@Composable
fun ModelLoadingSection(
    modelLoadState: ModelLoadState,
    isMinimized: Boolean,
    isProcessing: Boolean,
    onLoadModel: () -> Unit,
    onToggleMinimized: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Header with minimize/expand button
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "📥 Model Loading",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Button(
                    onClick = onToggleMinimized,
                    modifier = Modifier.size(32.dp),
                    contentPadding = PaddingValues(0.dp)
                ) {
                    Text(
                        text = if (isMinimized) "+" else "−",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            
            // Expandable content
            if (!isMinimized) {
                Spacer(modifier = Modifier.height(12.dp))
                
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Button(
                        onClick = onLoadModel,
                        enabled = !isProcessing && modelLoadState !is ModelLoadState.Loaded
                    ) {
                        Text(
                            when (modelLoadState) {
                                is ModelLoadState.Loading -> "Loading..."
                                is ModelLoadState.Loaded -> "Model Loaded"
                                else -> "Load Model"
                            }
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = getModelLoadStatusText(modelLoadState),
                        style = MaterialTheme.typography.bodyMedium,
                        color = getModelLoadStatusColor(modelLoadState)
                    )
                }
            } else {
                // Minimized state - show just the status
                Spacer(modifier = Modifier.height(4.dp))
                val (emoji, status) = getModelLoadStatusForMinimized(modelLoadState)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = emoji,
                        style = MaterialTheme.typography.bodySmall
                    )
                    Text(
                        text = status,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                    )
                }
            }
        }
    }
}

@Composable
fun TestingSection(
    selectedImageBitmap: Bitmap?,
    isProcessing: Boolean,
    onRandomTest: () -> Unit,
    onPickImage: () -> Unit,
    onClassifyImage: (Bitmap) -> Unit
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
                text = "🧪 Model Testing",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Test buttons row
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = onRandomTest,
                    enabled = !isProcessing,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("🎲 Random Test")
                }
                
                Button(
                    onClick = onPickImage,
                    enabled = !isProcessing,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("📷 Pick Image")
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
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = { onClassifyImage(bitmap) },
                    enabled = !isProcessing
                ) {
                    Text("🧠 Classify Image")
                }
            }
        }
    }
}

@Composable
fun ResultsSection(inferenceState: InferenceState) {
    when (inferenceState) {
        is InferenceState.Success -> {
            val result = inferenceState.result
            
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF4CAF50).copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "📊 Results",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = formatClassificationResult(result),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF4CAF50)
                    )
                }
            }
        }
        
        is InferenceState.Error -> {
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
                        text = "❌ Error",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = inferenceState.message,
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFFF44336)
                    )
                }
            }
        }
        
        else -> { /* Idle or Processing - no results to show */ }
    }
}

// Helper functions
private fun getModelLoadStatusText(state: ModelLoadState): String {
    return when (state) {
        is ModelLoadState.NotLoaded -> "Model not loaded yet"
        is ModelLoadState.Loading -> "Loading model..."
        is ModelLoadState.Loaded -> "✅ Model loaded successfully!"
        is ModelLoadState.Error -> "❌ ${state.message}"
    }
}

private fun getModelLoadStatusColor(state: ModelLoadState): Color {
    return when (state) {
        is ModelLoadState.Loaded -> Color(0xFF4CAF50)
        is ModelLoadState.Error -> Color(0xFFF44336)
        else -> Color.Gray
    }
}

private fun getModelLoadStatusForMinimized(state: ModelLoadState): Pair<String, String> {
    return when (state) {
        is ModelLoadState.NotLoaded -> "⏳" to "Not Loaded"
        is ModelLoadState.Loading -> "⏳" to "Loading..."
        is ModelLoadState.Loaded -> "✅" to "Model Loaded"
        is ModelLoadState.Error -> "❌" to "Load Failed"
    }
}

private fun formatClassificationResult(result: com.uzi.executorch.data.model.ClassificationResult): String {
    val sb = StringBuilder()
    sb.append("✅ Inference completed successfully!\n\n")
    sb.append("📊 Input: ${result.inputType}\n")
    sb.append("⏱️ Inference time: ${result.inferenceTimeMs}ms\n\n")
    sb.append("📋 Output Details:\n")
    sb.append("Shape: ${result.outputShape.contentToString()}\n")
    val actualClasses = if (result.outputShape.size >= 2) result.outputShape[1] else result.outputShape[0]
    sb.append("Expected: [1, $actualClasses] (dynamic)\n")
    sb.append("Shape correct: ${if (result.isShapeCorrect) "✅ Yes" else "❌ No"}\n\n")
    
    sb.append("🏆 Top 5 Predictions:\n")
    result.predictions.forEachIndexed { index, prediction ->
        sb.append("${index + 1}. ${prediction.className}\n")
        sb.append("   Class ${prediction.classIndex}: ${"%.2f".format(prediction.confidencePercentage)}%\n")
    }
    
    sb.append("\n📈 Output Statistics:\n")
    val stats = result.statistics
    sb.append("Confidence range: ${"%.2f".format(stats.minConfidence * 100)}% - ${"%.2f".format(stats.maxConfidence * 100)}%\n")
    sb.append("Top prediction: ${"%.2f".format(stats.topConfidence * 100)}%\n")
    
    return sb.toString()
}
