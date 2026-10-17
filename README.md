# Aeroelastic Surrogate Model Architecture for Flutter & Structural Load Prediction

Bu proje, açık kaynaklı çözücülerle (DoE - Latin Hypercube) oluşturulan veri setleri üzerinde, aeroelastik flutter (rezonans) bölgelerini ve tahmin belirsizliğini otonom olarak tespit eden Aktif Öğrenme (Adaptive Sampling) tabanlı bir vekil model (Surrogate Model) mimarisidir. 

Hesaplama maliyeti yüksek olan CFD/FEA analiz sürelerini, fiziksel sınırları koruyarak milisaniyelere düşürmeyi hedefler.

## 🚀 Proje Mimarisinde Kullanılan Teknolojiler
* **Veri Üretimi:** Latin Hypercube Sampling (SciPy)
* **Aktif Öğrenme & GPR:** Gaussian Process Regressor ile yüksek belirsizlik (varyans) noktalarının otonom tespiti (Scikit-Learn)
* **Derin Öğrenme:** PyTorch ile uçtan uca DNN eğitimi ve ONNX formatına model dışa aktarımı (Export)
* **Hassasiyet Analizi:** Model girdi parametrelerinin baskınlık oranlarını ölçen Sobol Hassasiyet Analizi (SALib)
* **Hiperparametre Optimizasyonu:** SVR modelleri için Optuna
* **Canlı Servis:** Eğitilen ONNX modelinin bağımsız bir FastAPI sunucusu üzerinden dışa açılması.

## ⚙️ Nasıl Çalıştırılır?

1. Bağımlılıkları yükleyin:
`pip install -r requirements.txt`

2. Sentetik Latin Hypercube veri setini üretin:
`python data/generate_doe.py`

3. Aktif Öğrenme döngüsünü ve DNN'i eğitin:
`python src/models/train_gpr.py`
`python src/models/train_dnn.py`
`python src/models/export_onnx.py`

4. FastAPI Sunucusunu Başlatın:
`uvicorn api.main:app --reload`
Sunucu başladığında `http://127.0.0.1:8000/docs` adresinden Swagger UI ile tahminleri test edebilirsiniz.