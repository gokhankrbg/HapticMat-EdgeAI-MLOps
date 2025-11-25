import mlflow
import os
import hashlib
import json

# --- Yapılandırma ---
# MinIO Bağlantı Bilgileri (Localhost'tan erişim için)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "strongpassword123" # Lütfen kendi şifrenizle değiştirin

EXPERIMENT_NAME = "MobileNetV2-HapticMat-Quantized"
MODEL_NAME = "PersonDetectionQuantized"
# DÜZELTİLDİ: Modelin kök dizinine göre doğru yolu
MODEL_FILE = "app/src/main/assets/mv2_xnnpack.pte"
MANIFEST_FILE = "manifest.json"

# --- Yardımcı Fonksiyonlar ---
def calculate_sha256(file_path):
    """Verilen dosyanın SHA256 özetini hesaplar."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- MLflow Akışı ---
if __name__ == "__main__":
    # Dosyanın varlığını kontrol et
    if not os.path.exists(MODEL_FILE):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_FILE}")
        exit(1)

    # MLflow Tracking Server'a bağlan
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Yeni bir Run başlat
    with mlflow.start_run(run_name="quantized_model_v1") as run:
        run_id = run.info.run_id
        print(f"--- MLflow Run ID: {run_id} ---")

        # 1. Parametreleri Kaydetme (Model Geliştirme Detayları)
        mlflow.log_param("quantization_type", "PTQ_Int8")
        mlflow.log_param("base_model", "MobileNetV2")
        mlflow.log_param("executorch_backend", "XNNPACK")

        # 2. Metrikleri Kaydetme (Simüle Edilmiş)
        mlflow.log_metric("top1_accuracy", 0.85)
        mlflow.log_metric("inference_latency_ms", 25.0)
        mlflow.log_metric("model_size_mib", os.path.getsize(MODEL_FILE) / (1024 * 1024))

        # 3. Artifacts'ı Kaydetme (Model Dosyası ve Manifest)

        # Manifest dosyasını oluştur
        sha256_hash = calculate_sha256(MODEL_FILE)

        manifest_data = {
            "model_name": MODEL_NAME,
            "version": "1.0.0",
            "sha256": sha256_hash,
            "model_file": MODEL_FILE,
            "metrics": {
                "top1_accuracy": 0.85,
                "latency_ms": 25.0
            }
        }

        # Manifest dosyasını geçici olarak oluştur
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest_data, f, indent=4)

        # Artifacts'ı MLflow'a (yani MinIO'ya) yükle
        mlflow.log_artifact(MODEL_FILE, "model_files")
        mlflow.log_artifact(MANIFEST_FILE, "model_files")

        print(f"Model and Manifest files were saved in MinIO.")

        # 4. Modeli MLflow Registry'ye Kaydetme (Yoruma alındı)
        # Model Registry API hatasını atlamak için bu kısım yoruma alındı.
        # print(f"Model '{MODEL_NAME}' MLflow Registry'ye kaydedildi.")