# Model Seçim Gerekçeleri

## Seçilen Modeller

Bu çalışmada dört farklı makine öğrenmesi modeli karşılaştırılmıştır:

1. **Random Forest (RF)**
2. **Support Vector Machine (SVM)**
3. **Naive Bayes (NB)**
4. **Logistic Regression (LR)**

## Model Seçim Gerekçeleri

### 1. Random Forest

**Neden seçildi:**
- **Ensemble yöntemi**: Birden fazla karar ağacının birleşimiyle güçlü bir model oluşturur
- **Özellik önemliliği**: Hangi özelliklerin tanı için kritik olduğunu belirleyebilir (interpretability)
- **Aykırı değerlere dayanıklılık**: Tıbbi veri setlerinde yaygın olan aykırı değerlere karşı nispeten dayanıklıdır
- **Non-lineer ilişkileri yakalama**: Özellikler arasındaki karmaşık, non-lineer ilişkileri modelleyebilir
- **Sınıf dengesizliği**: `class_weight='balanced'` parametresi ile sınıf dengesizliğini ele alabilir
- **Tıbbi veri analizinde yaygın kullanım**: ECG ve kardiyovasküler veri analizinde kanıtlanmış başarılı sonuçlar

**Beklenen avantajlar:**
- Yüksek genelleme yeteneği
- Özellik önemliliği analizi için uygun
- Overfitting'e karşı nispeten dayanıklı

### 2. Support Vector Machine (SVM)

**Neden seçildi:**
- **Farklı öğrenme paradigması**: RF'den farklı olarak margin maximization prensibiyle çalışır
- **Yüksek boyutlu veri**: 200 özellikli veri setinde etkili çalışabilir
- **Kernel trick**: RBF kernel ile non-lineer decision boundary'ler oluşturabilir
- **Sınıf dengesizliği desteği**: `class_weight='balanced'` ile sınıf dengesizliğini ele alır
- **Tıbbi görüntü ve sinyal analizinde yaygın**: ECG sinyali analizinde yaygın kullanılan bir yöntemdir

**Beklenen avantajlar:**
- RF'den farklı bir yaklaşım, farklı özellikleri vurgulayabilir
- Yüksek boyutlu özellik uzayında etkili
- Güçlü genelleme yeteneği

**Dikkat edilmesi gerekenler:**
- Veri ölçeklendirmesi (scaling) gereklidir
- Büyük veri setlerinde eğitim süresi uzun olabilir

### 3. Naive Bayes

**Neden seçildi:**
- **Basit ve hızlı**: Eğitim ve tahmin süreleri çok kısadır
- **Probabilistic çıktı**: Sınıf olasılıkları doğal olarak üretilir
- **Baseline karşılaştırması**: Diğer modellere göre performans karşılaştırması için baseline sağlar
- **Bayesian yaklaşım**: Olasılıksal çıkarım yapabilir
- **Küçük veri setlerinde etkili**: Sınırlı veri durumlarında iyi performans gösterebilir

**Beklenen avantajlar:**
- Hızlı eğitim ve tahmin
- Baseline olarak diğer modellerle karşılaştırma imkanı
- Interpretable olasılık çıktıları

**Dikkat edilmesi gerekenler:**
- Özellikler arası bağımsızlık varsayımı (Naive assumption) gerçekçi olmayabilir
- Non-lineer ilişkileri yakalamada sınırlı olabilir

### 4. Logistic Regression

**Neden seçildi:**
- **Linear model**: Basit ve interpretable bir model
- **Probabilistic çıktı**: Sınıf olasılıkları doğal olarak üretilir
- **Regularization desteği**: Overfitting'i önlemek için regularization kullanılabilir
- **Tıbbi veri analizinde yaygın**: Klinik karar destek sistemlerinde yaygın kullanılan bir yöntem
- **Baseline karşılaştırması**: Daha karmaşık modellere göre baseline sağlar
- **Hızlı eğitim**: Ensemble modellere göre çok daha hızlı eğitilir

**Beklenen avantajlar:**
- Interpretable model katsayıları
- Hızlı eğitim ve tahmin
- Probabilistic çıktılar
- Overfitting'e karşı dayanıklı (regularization ile)

**Dikkat edilmesi gerekenler:**
- Linear decision boundary, non-lineer ilişkileri yakalamada sınırlı
- Özellik ölçeklendirmesi (scaling) önerilir

## Karşılaştırma Stratejisi

Bu dört model farklı öğrenme paradigmalarını temsil eder:
- **RF**: Ensemble, tree-based, bagging
- **SVM**: Margin-based, kernel methods
- **NB**: Probabilistic, Bayesian
- **LR**: Linear, probabilistic

Bu çeşitlilik, farklı model yaklaşımlarının veri setimizdeki performansını objektif olarak değerlendirmemize olanak sağlar.

## Sonuç

Her model farklı güçlü yönlere sahiptir ve tıbbi veri analizinde yaygın olarak kullanılmaktadır. Bu karşılaştırma, veri setimiz için en uygun modeli belirlememize ve farklı model yaklaşımlarının avantaj/dezavantajlarını anlamamıza yardımcı olacaktır.

