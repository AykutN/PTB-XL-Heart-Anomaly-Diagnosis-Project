# SVM Model Hiperparametreleri - Top-200 Class-Weight Yaklaşımı

## Kullanılan Hiperparametreler

SVM modeli, Top-200 özellik seti ile class-weight yaklaşımı kullanılarak eğitilmiştir. Kullanılan hiperparametreler aşağıda detaylandırılmıştır.

### Temel SVM Parametreleri

**SVC (Support Vector Classifier) Parametreleri:**

```python
SVC(
    kernel='rbf',              # Radial Basis Function kernel
    class_weight='balanced',   # Otomatik sınıf ağırlıklandırması
    probability=True,          # Olasılık tahminleri için gerekli
    random_state=42,          # Tekrarlanabilirlik için
    cache_size=1000           # Bellek optimizasyonu (MB)
)
```

**Açıklamalar:**

1. **kernel='rbf'**: 
   - Radial Basis Function (RBF) kernel kullanılmıştır
   - Non-lineer ilişkileri yakalamak için uygundur
   - Yüksek boyutlu veri setlerinde (200 özellik) etkilidir
   - Formül: K(x, x') = exp(-γ||x - x'||²)

2. **class_weight='balanced'**:
   - Sınıf dengesizliğini ele almak için kullanılmıştır
   - Her sınıf için otomatik ağırlık hesaplanır: `n_samples / (n_classes * np.bincount(y))`
   - Azınlık sınıflarına (HYP, CD) daha yüksek ağırlık verilir
   - Çoğunluk sınıfına (NORM) daha düşük ağırlık verilir

3. **probability=True**:
   - Olasılık tahminleri üretmek için gerekli
   - Platt scaling (sigmoid calibration) kullanılarak olasılıklar hesaplanır
   - Threshold optimizasyonu için kritik

4. **random_state=42**:
   - Tekrarlanabilirlik için seed değeri
   - Platt scaling'in rastgele başlangıç değerlerini sabitler

5. **cache_size=1000**:
   - Kernel matrisinin cache'lenmesi için bellek miktarı (MB)
   - Büyük veri setlerinde eğitim süresini kısaltır
   - 17,000+ örnek için 1000 MB uygun bir değerdir

### Varsayılan Parametreler (Belirtilmemiş)

Aşağıdaki parametreler varsayılan değerlerle kullanılmıştır:

- **C=1.0**: Regularization parametresi
  - Margin ile sınıflandırma hatası arasındaki dengeyi kontrol eder
  - Daha yüksek C değeri daha sıkı margin (daha az tolerans) anlamına gelir
  - Varsayılan değer (1.0) genellikle iyi bir başlangıç noktasıdır

- **gamma='scale'**: RBF kernel parametresi
  - Kernel'in etki alanını kontrol eder
  - 'scale' değeri: `gamma = 1 / (n_features * X.var())`
  - Daha yüksek gamma değeri daha lokal etki alanı (overfitting riski)
  - 'scale' otomatik olarak özellik sayısına ve varyansa göre ayarlanır
  - 200 özellik için uygun bir değerdir

- **tol=1e-3**: Convergence tolerance
  - Optimizasyon algoritmasının durdurma kriteri
  - Varsayılan değer genellikle yeterlidir

### Multi-Label Classification Yaklaşımı

**OneVsRestClassifier:**

```python
OneVsRestClassifier(
    base_svm,    # SVC modeli
    n_jobs=-1   # Tüm CPU çekirdeklerini kullan
)
```

- Her sınıf için ayrı bir binary SVM modeli eğitilir
- 5 sınıf (NORM, MI, STTC, CD, HYP) için 5 ayrı model
- Her modelin tahmin olasılıkları birleştirilir
- `n_jobs=-1` ile paralel eğitim yapılır (hızlandırma)

### Veri Ön İşleme

**StandardScaler:**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

- SVM, özelliklerin ölçeklendirilmesine duyarlıdır
- Her özellik ortalama 0, standart sapma 1 olacak şekilde normalize edilir
- Train setinde fit edilir, validation ve test setlerine aynı transformasyon uygulanır
- RBF kernel'in performansı için kritik

### Threshold Optimizasyonu

Her sınıf için optimal threshold belirlenir:

```python
# Validation setinde F1-Score'u maksimize eden threshold bulunur
for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_val_prob[:, i] >= threshold).astype(int)
    f1 = f1_score(y_val[col], y_pred)
    # En yüksek F1-Score'u veren threshold seçilir
```

- Her sınıf için 0.1 ile 0.9 arasında 0.01 adımlarla test edilir
- Validation setinde F1-Score maksimize edilir
- Optimal threshold'lar test setine uygulanır

## Sonuç

Bu hiperparametre konfigürasyonu ile SVM modeli:
- **F1-Score: 0.746** (macro-averaged)
- **ROC-AUC: 0.922** (macro-averaged)
- **Recall: 0.790** (macro-averaged)

performansı elde etmiştir. RBF kernel ve class-weight yaklaşımı, sınıf dengesizliği olan multi-label sınıflandırma problemi için etkili bir kombinasyon sağlamıştır.

