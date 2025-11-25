package com.uzi.executorch.data.repository

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.gson.Gson
// import com.uzi.executorch.BuildConfig // Hata verdiği için yoruma alındı/silindi
import com.uzi.executorch.data.model.*
import com.uzi.executorch.ImageNetClasses
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import kotlin.random.Random

/**
 * MLOps Model Metadata yapısı (latest.json içeriği)
 */
data class ModelMetadata(
    val sha256: String,
    val objects: Map<String, String>
)

/**
 * Repository for managing ExecuTorch model operations
 */
class ModelRepository(private val context: Context) {

    companion object {
        private const val TAG = "ModelRepository"

        private const val METADATA_FILE = "models/latest.json"
        private const val MODEL_LOCAL_NAME = "downloaded_model.pte"
    }

    private val httpClient = OkHttpClient()
    private val gson = Gson()
    private var module: Module? = null

    /**
     * Load the ExecuTorch model:
     * 1. Check for local model.
     * 2. Download metadata (latest.json).
     * 3. Download model file from MinIO.
     * 4. Verify SHA256.
     * 5. Load the model.
     */
    suspend fun loadModel(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Starting MLOps model load process...")

            // 1. Metadata (latest.json) İndirme
            // BuildConfig'i tam yoluyla kullanıyoruz: com.uzi.executorch.BuildConfig
            val metadataUrl = com.uzi.executorch.BuildConfig.MINIO_BASE_URL + METADATA_FILE
            Log.d(TAG, "Downloading metadata from: $metadataUrl")
            val metadata = downloadMetadata(metadataUrl).getOrElse {
                return@withContext Result.failure(Exception("Failed to download model metadata: ${it.message}"))
            }

            val modelRemotePath = metadata.objects["model"] ?: return@withContext Result.failure(Exception("Model path not found in metadata."))
            val modelUrl = com.uzi.executorch.BuildConfig.MINIO_BASE_URL + modelRemotePath
            val expectedSha256 = metadata.sha256

            // 2. Model Dosyasını İndirme
            val localModelFile = File(context.filesDir, MODEL_LOCAL_NAME)
            Log.d(TAG, "Downloading model from: $modelUrl to ${localModelFile.absolutePath}")
            downloadModelFile(modelUrl, localModelFile).getOrElse {
                return@withContext Result.failure(Exception("Failed to download model file: ${it.message}"))
            }

            // 3. SHA256 Kontrolü (Güvenlik ve Bütünlük)
            val actualSha256 = calculateSha256(localModelFile)
            if (actualSha256 != expectedSha256) {
                val error = "SHA256 mismatch! Expected: $expectedSha256, Actual: $actualSha256"
                Log.e(TAG, error)
                localModelFile.delete() // Hatalı dosyayı sil
                return@withContext Result.failure(Exception(error))
            }
            Log.d(TAG, "✅ SHA256 verification successful.")

            // 4. Modeli Yükleme
            module = Module.load(localModelFile.absolutePath)
            Log.d(TAG, "✅ Model loaded successfully from MinIO: ${localModelFile.name}")
            Result.success(Unit)

        } catch (e: Exception) {
            val error = "MLOps Model Load Failed: ${e.message}"
            Log.e(TAG, error, e)
            Result.failure(Exception(error))
        }
    }

    /**
     * MinIO'dan latest.json dosyasını indirir ve ayrıştırır.
     */
    private fun downloadMetadata(url: String): Result<ModelMetadata> {
        return try {
            val request = Request.Builder().url(url).build()
            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return Result.failure(IOException("Failed to download metadata: ${response.code}"))
                }
                val jsonString = response.body?.string() ?: return Result.failure(IOException("Empty metadata response."))
                val metadata = gson.fromJson(jsonString, ModelMetadata::class.java)
                Result.success(metadata)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Model dosyasını MinIO'dan indirir ve yerel dosyaya kaydeder.
     */
    private fun downloadModelFile(url: String, outputFile: File): Result<Unit> {
        return try {
            val request = Request.Builder().url(url).build()
            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return Result.failure(IOException("Failed to download model file: ${response.code}"))
                }
                response.body?.byteStream()?.use { inputStream ->
                    FileOutputStream(outputFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                } ?: return Result.failure(IOException("Empty model file response."))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Verilen dosyanın SHA256 özetini hesaplar.
     */
    private fun calculateSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { inputStream ->
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    // --- Mevcut Kodun Geri Kalanı (Değişmedi) ---

    /**
     * Run inference with random input data
     */
    suspend fun runRandomInference(): Result<ClassificationResult> = withContext(Dispatchers.IO) {
        val randomInput = createRandomInput()
        runInference(randomInput, InputType.RANDOM)
    }

    /**
     * Run inference with image input
     */
    suspend fun runImageInference(bitmap: Bitmap): Result<ClassificationResult> = withContext(Dispatchers.IO) {
        val imageInput = preprocessImage(bitmap)
        runInference(imageInput, InputType.IMAGE)
    }

    /**
     * Check if model is loaded
     */
    fun isModelLoaded(): Boolean = module != null

    /**
     * Create random input tensor
     */
    private fun createRandomInput(): FloatArray {
        val inputSize = 1 * 3 * 224 * 224
        return FloatArray(inputSize) { Random.nextFloat() * 2.0f - 1.0f }
    }

    /**
     * Preprocess image for model input
     */
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        // Resize image to 224x224
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // Convert to float array with ImageNet normalization
        val inputSize = 1 * 3 * 224 * 224
        val input = FloatArray(inputSize)

        val pixels = IntArray(224 * 224)
        resizedBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        // ImageNet normalization values
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f) // RGB
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)  // RGB

        for (i in pixels.indices) {
            val pixel = pixels[i]

            // Extract RGB values (0-255)
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Apply ImageNet normalization and arrange as CHW format
            input[i] = (r - mean[0]) / std[0]                    // R channel
            input[224 * 224 + i] = (g - mean[1]) / std[1]        // G channel
            input[2 * 224 * 224 + i] = (b - mean[2]) / std[2]    // B channel
        }

        return input
    }

    /**
     * Run inference with input data
     */
    private fun runInference(inputData: FloatArray, inputType: InputType): Result<ClassificationResult> {
        return try {
            val currentModule = module ?: return Result.failure(Exception("Model not loaded"))

            Log.d(TAG, "Creating input tensor for ${inputType.displayName}...")
            Log.d(TAG, "Input data size: ${inputData.size}")

            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, 224, 224))
            val inputEValue = EValue.from(inputTensor)

            Log.d(TAG, "Running inference on ${inputType.displayName}...")
            val startTime = System.currentTimeMillis()

            // Run inference
            val outputs = currentModule.forward(inputEValue)

            val endTime = System.currentTimeMillis()
            val inferenceTime = endTime - startTime

            Log.d(TAG, "Inference completed in ${inferenceTime}ms")

            // Validate output
            if (outputs.isEmpty()) {
                return Result.failure(Exception("No outputs received from model"))
            }

            val outputTensor = outputs[0].toTensor()
            val outputShape = outputTensor.shape()
            val outputData = outputTensor.dataAsFloatArray

            Log.d(TAG, "Output shape: ${outputShape.contentToString()}")
            Log.d(TAG, "Output data length: ${outputData.size}")

            // Apply softmax to get probabilities
            val probabilities = applySoftmax(outputData)

            // Check if shape is correct (adjust for actual output size)
            val actualClasses = if (outputShape.size >= 2) outputShape[1].toInt() else outputData.size
            val expectedShape = longArrayOf(1, actualClasses.toLong())
            val isShapeCorrect = outputShape.contentEquals(expectedShape)

            Log.d(TAG, "Actual classes: $actualClasses")
            Log.d(TAG, "Expected shape adjusted: ${expectedShape.contentToString()}")
            Log.d(TAG, "Probabilities array size: ${probabilities.size}")

            // Create top 5 predictions with bounds checking
            val predictions = probabilities.mapIndexed { index, confidence ->
                ClassificationPrediction(
                    classIndex = index,
                    className = try {
                        // Handle different model outputs (998, 1000, etc.)
                        if (index < 1000 && actualClasses <= 1000) {
                            ImageNetClasses.getFormattedClassName(index)
                        } else {
                            "Class $index (Custom Model)"
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to get class name for index $index: ${e.message}")
                        "Class $index"
                    },
                    confidence = confidence
                )
            }.sortedByDescending { it.confidence }
                .take(5)

            // Calculate statistics
            val statistics = ClassificationStatistics(
                minConfidence = probabilities.minOrNull() ?: 0f,
                maxConfidence = probabilities.maxOrNull() ?: 0f,
                meanConfidence = probabilities.average().toFloat(),
                topConfidence = probabilities.maxOrNull() ?: 0f
            )

            val result = ClassificationResult(
                predictions = predictions,
                inferenceTimeMs = inferenceTime,
                inputType = inputType.displayName,
                outputShape = outputShape,
                isShapeCorrect = isShapeCorrect,
                statistics = statistics
            )

            Log.d(TAG, "Classification completed successfully")
            Result.success(result)

        } catch (e: Exception) {
            val error = "Inference failed: ${e.message}\nStack trace: ${e.stackTraceToString()}"
            Log.e(TAG, error, e)
            Result.failure(Exception("Inference failed: ${e.message}"))
        }
    }

    /**
     * Apply softmax to convert logits to probabilities
     */
    private fun applySoftmax(logits: FloatArray): FloatArray {
        return try {
            val maxLogit = logits.maxOrNull() ?: 0f
            Log.d(TAG, "Applying softmax - input size: ${logits.size}, max logit: $maxLogit")

            val expValues = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
            val sum = expValues.sum()

            if (sum <= 0f) {
                Log.w(TAG, "Invalid softmax sum: $sum, returning uniform distribution")
                return FloatArray(logits.size) { 1f / logits.size }
            }

            val result = expValues.map { it / sum }.toFloatArray()
            Log.d(TAG, "Softmax applied successfully - output sum: ${result.sum()}")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Error applying softmax: ${e.message}", e)
            // Return uniform distribution as fallback
            FloatArray(logits.size) { 1f / logits.size }
        }
    }

    /**
     * Get file path from assets
     * ARTIK KULLANILMIYOR: Bu fonksiyon, modelin assets'ten yüklenmesi için kullanılıyordu.
     */
    @Throws(IOException::class)
    private fun getAssetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}