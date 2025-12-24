# Yaklaşım Açıklamaları

## 1. Class-Weight Yaklaşımı

**Nasıl Çalışır:**

Class-weight yaklaşımı, sınıf dengesizliği problemini çözmek için kullanılan bir yöntemdir. Bu yaklaşımda, model eğitimi sırasında azınlık sınıflarına (örneğin HYP, CD) daha yüksek ağırlık verilirken, çoğunluk sınıfına (NORM) daha düşük ağırlık verilir. Bu sayede model, azınlık sınıflarındaki örnekleri daha iyi öğrenir ve sınıf dengesizliğinden kaynaklanan yanlılığı azaltır.

**Random Forest için:** `class_weight='balanced'` parametresi kullanılarak, her sınıf için otomatik olarak ağırlık hesaplanır. Ağırlık, sınıfın ters orantılı frekansına göre belirlenir (n_samples / (n_classes * np.bincount(y))).

**SVM için:** Benzer şekilde `class_weight='balanced'` parametresi kullanılır. SVM'in margin maximization prensibi, ağırlıklı örneklerle çalıştığında azınlık sınıflarına daha fazla önem verir.

**Avantajları:**
- Veri kaybı olmadan sınıf dengesizliğini ele alır
- Tüm veriyi kullanır, bu nedenle bilgi kaybı yoktur
- Hızlı ve basit bir yaklaşımdır
- Top-50, Top-100, Top-200 özellik setleri için uygulanabilir

**Dezavantajları:**
- Çok dengesiz veri setlerinde yeterli olmayabilir
- Ağırlık hesaplaması veri dağılımına bağlıdır

## 2. Ensemble Undersampling Yaklaşımı

**Nasıl Çalışır:**

Ensemble undersampling yaklaşımı, 50 farklı model eğiterek her birinin farklı dengelenmiş alt örneklemler üzerinde eğitilmesini sağlar. Her iterasyonda:

1. **Sınıf sayıları belirlenir:** Tüm sınıfların örnek sayıları hesaplanır ve en küçük sınıf (genellikle HYP) belirlenir.

2. **Birincil sınıf ataması:** Multi-label veri setinde her örnek için bir "birincil sınıf" belirlenir. Eğer bir örnek birden fazla sınıfa aitse, en nadir sınıf birincil sınıf olarak seçilir (örneğin, HYP > CD > STTC > MI > NORM sırası).

3. **Dengelenmiş alt örneklem oluşturma:** Her sınıftan en küçük sınıf sayısı kadar örnek rastgele seçilir. Bu sayede her iterasyonda dengelenmiş bir alt örneklem oluşturulur.

4. **Model eğitimi:** Her dengelenmiş alt örneklem üzerinde ayrı bir model eğitilir (50 iterasyon için 50 model).

5. **Olasılık ortalaması:** Tüm modellerin tahmin olasılıkları toplanır ve ortalaması alınır. Bu ensemble yaklaşımı, tek bir modelden daha güvenilir ve genelleme yeteneği yüksek tahminler üretir.

**Random Forest için:** Her iterasyonda 100 ağaçlı, max_depth=10 olan daha küçük bir Random Forest modeli eğitilir. Farklı random_state değerleri kullanılarak çeşitlilik sağlanır.

**SVM için:** Her iterasyonda RBF kernel'li SVM modeli eğitilir. Veri ölçeklendirmesi (scaling) her iterasyonda yapılır. class_weight kullanılmaz çünkü zaten dengelenmiş veri üzerinde eğitim yapılır.

**Avantajları:**
- Çok güçlü bir ensemble yaklaşımıdır
- Her model farklı bir veri perspektifinden öğrenir
- Azınlık sınıflarının öğrenilmesini güçlendirir
- Overfitting riskini azaltır
- Yüksek genelleme yeteneği sağlar

**Dezavantajları:**
- Eğitim süresi uzundur (50 model eğitilir)
- Daha fazla hesaplama kaynağı gerektirir
- Sadece Top-200 özellik seti için uygulanmıştır (hesaplama maliyeti nedeniyle)

**Neden 50 Model?**

50 model sayısı, ensemble çeşitliliği ile hesaplama maliyeti arasında bir denge sağlar. Daha az model (örneğin 10-20) yeterli çeşitlilik sağlamayabilir, daha fazla model (örneğin 100+) ise hesaplama maliyetini artırırken marjinal iyileşme sağlar. 50 model, literatürde yaygın olarak kullanılan ve etkili bir ensemble boyutudur.

