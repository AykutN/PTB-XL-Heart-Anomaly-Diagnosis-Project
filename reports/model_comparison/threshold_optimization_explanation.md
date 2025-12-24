# Threshold Optimizasyonu Açıklaması

## 1. Random Forest ve Olasılık Tahminleri

**Evet, Random Forest olasılıksal çalışır ve threshold optimizasyonu doğru bir yaklaşımdır.**

Random Forest, `predict_proba()` metodu ile her sınıf için olasılık tahminleri üretir. Her ağaç bir tahmin yapar ve final olasılık, tüm ağaçların tahminlerinin ortalaması alınarak hesaplanır. Örneğin, 250 ağaçlı bir Random Forest'te, her ağaç bir örneğin pozitif sınıfa ait olma olasılığını tahmin eder ve bu 250 tahminin ortalaması final olasılık olur.

**Neden threshold optimizasyonu gerekli?**

1. **Varsayılan threshold (0.5) her zaman optimal değildir:** Özellikle sınıf dengesizliği olan veri setlerinde, optimal threshold 0.5'ten farklı olabilir. Örneğin, azınlık sınıflar için daha düşük bir threshold (örneğin 0.3-0.4) kullanmak, recall'ı artırabilir.

2. **Sınıf dengesizliği:** Veri setimizde NORM sınıfı 7,596 örnek içerirken, HYP sınıfı sadece 2,119 örnek içermektedir. Bu durumda, model NORM sınıfını daha yüksek olasılıkla tahmin etme eğiliminde olabilir. Threshold optimizasyonu, bu yanlılığı azaltır.

3. **F1-Score optimizasyonu:** Kodumuzda, her sınıf için validation setinde F1-Score'u maksimize eden threshold bulunur (0.1 ile 0.9 arasında 0.01 adımlarla). Bu sayede, her sınıf için en iyi precision-recall dengesi sağlanır.

## 2. SVM ve Olasılık Tahminleri

**Evet, SVM'de de threshold optimizasyonu gerekli ve yapılıyor.**

SVM, `probability=True` parametresi ile olasılık tahminleri üretir. Ancak SVM'in olasılık tahminleri, Random Forest'ten farklı bir şekilde hesaplanır:

1. **Platt Scaling:** SVM'in olasılık tahminleri, Platt scaling (sigmoid calibration) kullanılarak hesaplanır. Bu, SVM'in decision function çıktılarını olasılığa dönüştüren bir kalibrasyon yöntemidir.

2. **Daha az güvenilir olasılıklar:** SVM'in olasılık tahminleri genellikle Random Forest'ten daha az güvenilir olabilir çünkü:
   - SVM'in temel çıktısı decision function'dır (mesafe), olasılık değildir
   - Platt scaling, ek bir kalibrasyon adımıdır ve her zaman mükemmel değildir
   - Ancak yine de kullanışlıdır ve threshold optimizasyonu ile iyileştirilebilir

3. **Threshold optimizasyonu neden gerekli:**
   - SVM'de de sınıf dengesizliği problemi vardır
   - Class-weight kullanılsa bile, optimal threshold 0.5'ten farklı olabilir
   - Her sınıf için ayrı threshold optimizasyonu, model performansını önemli ölçüde artırabilir

## Kodda Nasıl Uygulanıyor?

Her iki model için de aynı threshold optimizasyonu yöntemi kullanılıyor:

```python
def find_optimal_thresholds(y_val, y_val_prob, target_cols):
    thresholds = {}
    for i, col in enumerate(target_cols):
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (y_val_prob[:, i] >= threshold).astype(int)
            f1 = f1_score(y_val[col], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[col] = best_threshold
    return thresholds
```

Bu fonksiyon:
1. Validation setinde her sınıf için 0.1 ile 0.9 arasında threshold'ları test eder
2. Her threshold için F1-Score hesaplar
3. F1-Score'u maksimize eden threshold'u seçer
4. Bu optimal threshold'lar test setine uygulanır

## Sonuç

- **Random Forest:** Olasılıksal çalışır, threshold optimizasyonu doğru ve gerekli
- **SVM:** Olasılık tahminleri Platt scaling ile yapılır, threshold optimizasyonu gerekli ve yapılıyor
- Her iki model için de sınıf dengesizliği nedeniyle optimal threshold'lar 0.5'ten farklı olabilir
- Threshold optimizasyonu, özellikle azınlık sınıflar için performansı önemli ölçüde artırabilir

