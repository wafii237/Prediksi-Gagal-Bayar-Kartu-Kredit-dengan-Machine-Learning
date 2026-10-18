# Prediksi Gagal Bayar Kartu Kredit dengan Machine Learning

Studi kasus klasifikasi biner untuk memprediksi apakah nasabah kartu kredit akan **gagal bayar pada bulan berikutnya**, menggunakan Logistic Regression, Random Forest, dan Gradient Boosting.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/NAMA-REPO/blob/main/Studi_Kasus_Klasifikasi_Credit_Default.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

> Tugas Individu: Studi Kasus Klasifikasi  
> **Muhammad Faeruz Wafi Abidin** | D3 Statistika Terapan dan Komputasi, Universitas Negeri Semarang

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Metode](#metode)
- [Tahapan Penerapan](#tahapan-penerapan)
- [Hasil](#hasil)
- [Manfaat](#manfaat)
- [Cara Menjalankan](#cara-menjalankan)
- [Struktur Repositori](#struktur-repositori)
- [Keterbatasan](#keterbatasan)
- [Referensi](#referensi)

## Latar Belakang

Bank penerbit kartu kredit menghadapi risiko nasabah yang tidak mampu membayar tagihan (*default*). Gagal bayar menimbulkan kerugian finansial, sedangkan penolakan nasabah yang terlalu ketat mengurangi pendapatan bank. Proyek ini membangun model klasifikasi untuk membantu menilai risiko nasabah secara objektif berdasarkan profil demografi, riwayat pembayaran, tagihan, dan jumlah pembayaran selama 6 bulan terakhir.

Masalah ini bersifat **klasifikasi biner dengan kelas tidak seimbang**: sekitar 22% nasabah gagal bayar, sehingga akurasi saja tidak cukup sebagai ukuran kinerja.

## Dataset

**Default of Credit Card Clients**, UCI Machine Learning Repository  
Sumber: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

| Kelompok | Variabel |
|---|---|
| Kredit | `LIMIT_BAL` |
| Demografi | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Riwayat pembayaran | `PAY_1` sampai `PAY_6` |
| Jumlah tagihan | `BILL_AMT1` sampai `BILL_AMT6` |
| Jumlah pembayaran | `PAY_AMT1` sampai `PAY_AMT6` |
| **Target** | `DEFAULT` (1 = gagal bayar, 0 = tidak) |

Dataset memuat 30.000 nasabah kartu kredit di Taiwan (April sampai September 2005) dan diunduh otomatis oleh notebook.

## Metode

| Model | Keterangan |
|---|---|
| Logistic Regression | *Baseline*, mudah diinterpretasi, memakai `class_weight="balanced"` |
| Random Forest | Ensemble pohon keputusan (*bagging*), memakai `class_weight="balanced"` |
| Gradient Boosting | `HistGradientBoostingClassifier`, pohon dibangun berurutan |

Metrik evaluasi: Accuracy, Precision, Recall, F1-score, ROC-AUC, serta kurva ROC dan Precision-Recall.

## Tahapan Penerapan

1. **Memuat data** dari UCI (cadangan: OpenML).
2. **EDA**: distribusi target, nilai hilang, duplikat, korelasi, dan hubungan status pembayaran dengan gagal bayar.
3. **Preprocessing**: merapikan kode kategori yang tidak terdokumentasi pada `EDUCATION` dan `MARRIAGE`, pembagian data latih/uji 80:20 dengan stratifikasi, standardisasi di dalam `Pipeline` (khusus Logistic Regression) untuk mencegah kebocoran data.
4. **Pemodelan** dengan validasi silang 5-fold (`StratifiedKFold`).
5. **Evaluasi** pada data uji, termasuk *confusion matrix*.
6. **Penyesuaian threshold** untuk menyeimbangkan precision dan recall.
7. **Interpretasi** faktor risiko dengan *permutation importance*.

## Hasil

> Isi tabel ini dengan output dari notebook setelah dijalankan.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - |

**Model terbaik:** `[isi]` dengan ROC-AUC `[isi]`.  
**Faktor paling berpengaruh:** `[isi]`.

## Manfaat

- **Manajemen risiko kredit**: mengidentifikasi nasabah berisiko tinggi lebih awal.
- **Penentuan limit kredit**: skor risiko menjadi dasar penyesuaian batas kredit.
- **Efisiensi operasional**: tim penagihan dapat memprioritaskan nasabah dengan risiko tertinggi.
- **Keputusan berbasis data**: faktor risiko yang teridentifikasi dapat menjadi dasar kebijakan penerbitan kartu.

## Cara Menjalankan

**Opsi 1: Google Colab (disarankan)**

Klik tombol *Open In Colab* di bagian atas, lalu pilih *Runtime > Run all*.

**Opsi 2: Lokal**

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
pip install -r requirements.txt
jupyter notebook Studi_Kasus_Klasifikasi_Credit_Default.ipynb
```

Isi `requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
ucimlrepo
jupyter
```

## Struktur Repositori

```
.
├── Studi_Kasus_Klasifikasi_Credit_Default.ipynb   # Notebook utama
├── requirements.txt
└── README.md
```

## Keterbatasan

- Data berasal dari satu bank di Taiwan tahun 2005, sehingga belum tentu mewakili kondisi lain.
- *Hyperparameter* belum dioptimasi menyeluruh (misalnya dengan `GridSearchCV`).
- Hasil menunjukkan asosiasi prediktif, bukan hubungan sebab-akibat.
- Penggunaan variabel demografi (misalnya jenis kelamin) dalam keputusan kredit nyata perlu ditinjau dari sisi etika dan regulasi.

## Referensi

1. Yeh, I-C., & Lien, C-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480. https://doi.org/10.1016/j.eswa.2007.12.020
2. Yeh, I-C. (2016). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
6. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
