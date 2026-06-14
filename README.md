# Plant Disease Detection - EfficientNetB0 + Classical Preprocessing

Tugas besar mata kuliah Pemrosesan Citra Digital (CII4F3). Sistem klasifikasi penyakit daun tanaman menggunakan pipeline preprocessing klasik dikombinasikan dengan transfer learning EfficientNetB0.

**Dataset:** PlantVillage (54.305 citra, 38 kelas)  
**Hasil akhir:** Akurasi 89,22% | F1-score Makro 83,28%

---

## Deskripsi

Proyek ini membangun pipeline deteksi penyakit tanaman secara end-to-end. Setiap citra diproses terlebih dahulu dengan beberapa teknik pengolahan citra klasik sebelum dilatih menggunakan EfficientNetB0 yang sudah dilatih di ImageNet (transfer learning).

Ide utamanya: daripada langsung melempar gambar mentah ke model, preprocessing klasik dijalankan dulu untuk memperkuat fitur visual daun, baru setelah itu model deep learning bekerja.

---

## Pipeline Preprocessing

Setiap gambar melewati 5 tahap sebelum masuk ke training:

1. **Gaussian Blur** (kernel 5x5) - menekan noise frekuensi tinggi
2. **Median Blur** (kernel 3x3) - menghilangkan noise impuls
3. **CLAHE** pada channel L (ruang warna LAB) - peningkatan kontras lokal
4. **Contrast Stretching** - normalisasi rentang piksel per citra
5. **HSV Masking** - isolasi area daun dari latar belakang

Masking HSV punya mekanisme fallback: kalau hasil masking terlalu gelap (misalnya daun yang sudah menguning atau sangat terinfeksi), gambar dikembalikan tanpa masking. Ini mencegah input all-zero yang bisa menyebabkan NaN loss saat training.

Semua preprocessing dijalankan sekali di awal dan disimpan ke disk. Generator saat training hanya baca cache dan rescale -- tidak ada operasi berat per batch.

---

## Arsitektur Model

- Base: **EfficientNetB0** (pretrained ImageNet, 4,38 juta parameter total)
- Tambahan: GlobalAveragePooling2D -> Dense(256, ReLU) -> Dropout(0.4) -> Dense(38, softmax)

Training dua fase:
- **Fase 1** (base frozen): 5 epoch, LR 1e-4, melatih kepala klasifikasi saja
- **Fase 2** (fine-tuning penuh): 10 epoch, LR 1e-5, semua layer dibuka

---

## Hasil

| Metrik | Nilai |
|--------|-------|
| Akurasi | 89,22% |
| F1-score Makro | 83,28% |
| F1-score Weighted | 85,70% |

Sebagian besar kelas performa sangat baik (21 dari 38 kelas dengan F1 > 0.93). Kelas yang masih jadi tantangan adalah kelas-kelas jagung dan Tomato Late Blight karena kemiripan visual antar subkelasnya.

---

## Struktur File

```
.
├── sourcecode_m.radityafaturrahman_1304221050.ipynb   # notebook utama (Kaggle)
├── m-raditya-faturrahman-1304221050-laporan-plant-disease.pdf  # laporan tugas
└── README.md
```

---

## Cara Pakai (di Kaggle)

1. Upload notebook ke Kaggle
2. Tambahkan dataset PlantVillage via **+ Add Data** (cari: "PlantVillage Dataset" by abdallahalidev)
3. Aktifkan GPU: **Settings -> Accelerator -> GPU T4**
4. Run All

Notebook akan otomatis mendeteksi path dataset, build cache preprocessing, lalu training. Total estimasi waktu sekitar 3-4 jam (termasuk preprocessing ~15 menit dan training ~2-3 jam).

---

## Dependensi

Semua sudah tersedia di environment Kaggle:

- Python 3.10
- TensorFlow 2.x
- OpenCV (cv2)
- scikit-learn
- scikit-image
- NumPy, Matplotlib, Seaborn

---

**M. Raditya Faturrahman | 1304221050 | Pemrosesan Citra Digital CII4F3**
