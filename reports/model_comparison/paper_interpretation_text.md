# Makale İçin Metrik Yorumlama Metni

## Model Karşılaştırması Sonuçları ve Yorumlama

Tablo X'te dört farklı makine öğrenmesi modelinin (Random Forest, SVM, Naive Bayes, Logistic Regression) performans karşılaştırması sunulmaktadır. Tüm modeller Top-200 özellik seti ile eğitilmiş ve her sınıf için optimal threshold optimizasyonu uygulanmıştır.

### Genel Performans Analizi

**En iyi performans: SVM modeli** tarafından gösterilmiştir. SVM, F1-Score (0.746) ve ROC-AUC (0.922) metriklerinde en yüksek değerlere ulaşmıştır. Bu sonuç, SVM'in yüksek boyutlu özellik uzayında (200 özellik) margin maximization prensibiyle güçlü bir genelleme yeteneği sergilediğini göstermektedir. SVM'in recall değeri (0.790) özellikle yüksektir, bu da modelin pozitif vakaları tespit etmede etkili olduğunu işaret etmektedir.

**Random Forest modeli** SVM'e çok yakın bir performans göstermiştir (F1-Score: 0.732, ROC-AUC: 0.915). RF'in precision değeri (0.713) SVM'den biraz daha yüksektir, bu da RF'in yanlış pozitif oranını daha düşük tuttuğunu göstermektedir. RF'in ensemble yapısı ve özellik önemliliği analizi yeteneği, tıbbi veri analizinde önemli avantajlar sağlamaktadır.

**Logistic Regression** modeli orta düzeyde bir performans sergilemiştir (F1-Score: 0.712, ROC-AUC: 0.906). LR'in recall değeri (0.801) en yüksektir, bu da modelin pozitif vakaları tespit etmede başarılı olduğunu ancak precision'ın daha düşük olması nedeniyle bazı yanlış pozitifler ürettiğini göstermektedir. LR'in basit yapısı ve hızlı eğitim süresi, klinik uygulamalarda pratik avantajlar sağlayabilir.

**Naive Bayes** modeli en düşük performansı göstermiştir (F1-Score: 0.633, ROC-AUC: 0.847). NB'in özellikler arası bağımsızlık varsayımı, ECG özellikleri arasındaki güçlü korelasyonlar nedeniyle gerçekçi olmayabilir. Ancak NB'in recall değeri (0.711) hala kabul edilebilir düzeydedir ve modelin hızlı eğitim süresi, baseline karşılaştırması için değerli bir referans noktası sağlamaktadır.

### Sınıf Bazlı Performans Analizi

**NORM (Normal) sınıfı:** Tüm modeller için en yüksek performans gösteren sınıftır. RF (F1: 0.852) ve SVM (F1: 0.860) özellikle başarılıdır. Bu sonuç, normal ECG sinyallerinin diğer patolojik durumlardan ayırt edilmesinin nispeten kolay olduğunu göstermektedir.

**MI (Myocardial Infarction) sınıfı:** SVM (F1: 0.749) ve RF (F1: 0.738) benzer performans göstermiştir. MI sınıfının tespiti kritik öneme sahiptir ve her iki model de klinik uygulamada kullanılabilir düzeyde performans sergilemektedir.

**STTC (ST/T Change) sınıfı:** SVM (F1: 0.744) ve RF (F1: 0.738) yine benzer performans göstermiştir. SVM'in recall değeri (0.846) RF'den daha yüksektir, bu da STTC vakalarını tespit etmede SVM'in daha hassas olduğunu göstermektedir.

**CD (Conduction Disturbance) sınıfı:** Tüm modeller için benzer performans gözlenmiştir (F1: 0.73-0.75 arası). LR (F1: 0.740) bu sınıf için en iyi performansı göstermiştir.

**HYP (Hypertrophy) sınıfı:** Tüm modeller için en zor sınıf olmuştur. SVM (F1: 0.630) en iyi performansı göstermiş, ancak tüm modeller bu sınıf için daha düşük performans sergilemiştir. Bu durum, HYP sınıfının daha az örnek içermesi (2,119 örnek) ve diğer sınıflarla örtüşen özelliklere sahip olmasından kaynaklanıyor olabilir.

### Model Seçim Gerekçeleri

Bu çalışmada dört farklı model seçilmiştir çünkü her biri farklı öğrenme paradigmalarını temsil etmektedir: Random Forest (ensemble, tree-based), SVM (margin-based, kernel methods), Naive Bayes (probabilistic, Bayesian), ve Logistic Regression (linear, probabilistic). Bu çeşitlilik, farklı model yaklaşımlarının veri setimizdeki performansını objektif olarak değerlendirmemize olanak sağlamıştır. Tüm modeller tıbbi veri analizinde yaygın olarak kullanılmakta ve ECG sinyali analizinde kanıtlanmış başarılı sonuçlar göstermektedir.

### Klinik Uygulama Önerileri

SVM ve Random Forest modelleri benzer ve yüksek performans göstermiştir. SVM'in biraz daha yüksek recall değeri, özellikle kritik vakaların (MI, STTC) tespitinde avantaj sağlayabilir. Ancak Random Forest'in özellik önemliliği analizi yeteneği, modelin interpretability açısından klinik uygulamalarda daha değerli olabilir. Her iki model de klinik karar destek sistemlerinde kullanılabilir düzeyde performans sergilemektedir.

