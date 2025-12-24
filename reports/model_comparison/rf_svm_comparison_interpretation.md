# Random Forest ve SVM Karşılaştırması - Yorum Metni

## Tablo 7 ve Tablo 8 Yorumlaması

Tablo 7'de Random Forest ve SVM modellerinin Top-200 özellik seti ile class-weight yaklaşımı kullanılarak elde edilen genel performans metrikleri karşılaştırılmaktadır. Tablo 8'de ise SVM modelinin her bir tanı sınıfı için detaylı performans metrikleri sunulmaktadır.

### Genel Performans Karşılaştırması (Tablo 7)

**SVM modeli** genel olarak Random Forest'ten daha iyi performans göstermiştir. SVM'in F1-Score değeri (0.746) Random Forest'ten (0.732) %1.9 daha yüksektir. Bu fark, özellikle recall metriklerinde belirgindir: SVM'in recall değeri (0.790) Random Forest'ten (0.756) %4.5 daha yüksektir. Bu sonuç, SVM'in pozitif vakaları tespit etmede daha başarılı olduğunu göstermektedir. ROC-AUC değerleri de SVM lehine (0.922 vs 0.915) olup, modelin genel ayırt etme yeteneğinin daha güçlü olduğunu işaret etmektedir.

**Random Forest modeli** ise precision değerinde (0.713) SVM'den (0.711) biraz daha yüksek performans göstermiştir, ancak bu fark istatistiksel olarak anlamlı değildir. RF'in özellik önemliliği analizi yeteneği, modelin interpretability açısından avantaj sağlamaktadır ve klinik uygulamalarda hangi özelliklerin tanı için kritik olduğunu belirlemede değerli bilgiler sunabilir.

### Sınıf Bazlı Performans Analizi (Tablo 8)

**NORM (Normal) sınıfı:** SVM modeli bu sınıf için en yüksek performansı göstermiştir (F1: 0.860, ROC-AUC: 0.944). Normal ECG sinyallerinin diğer patolojik durumlardan ayırt edilmesi nispeten kolaydır ve her iki model de bu sınıf için yüksek performans sergilemektedir.

**MI (Myocardial Infarction) sınıfı:** SVM modeli bu kritik sınıf için iyi bir performans göstermiştir (F1: 0.749, ROC-AUC: 0.923). MI sınıfının tespiti klinik açıdan kritik öneme sahiptir ve SVM'in yüksek recall değeri (0.740), pozitif vakaları tespit etmede başarılı olduğunu göstermektedir.

**STTC (ST/T Change) sınıfı:** SVM modeli bu sınıf için yüksek recall değeri (0.846) göstermiştir, bu da STTC vakalarını tespit etmede modelin hassas olduğunu işaret etmektedir. F1-Score (0.744) ve ROC-AUC (0.929) değerleri kabul edilebilir düzeydedir.

**CD (Conduction Disturbance) sınıfı:** SVM modeli bu sınıf için dengeli bir performans göstermiştir (F1: 0.748, ROC-AUC: 0.908). Precision (0.746) ve recall (0.750) değerleri birbirine yakındır, bu da modelin bu sınıf için dengeli bir tahmin yaptığını göstermektedir.

**HYP (Hypertrophy) sınıfı:** Tüm sınıflar içinde en zor sınıf olmuştur. SVM modeli bu sınıf için F1-Score değeri 0.630 ve ROC-AUC değeri 0.905'tir. Bu durum, HYP sınıfının daha az örnek içermesi (2,119 örnek) ve diğer sınıflarla örtüşen özelliklere sahip olmasından kaynaklanıyor olabilir. Ancak, diğer sınıflara göre daha düşük kalsa da tüm sınıflarda AUC'nin **0.88+** olması, modelin genel ayrıştırma kapasitesinin güçlü olduğunun göstergesidir.

### Sonuç ve Klinik Öneriler

SVM ve Random Forest modelleri benzer ve yüksek performans göstermiştir. SVM'in biraz daha yüksek recall ve F1-Score değerleri, özellikle kritik vakaların (MI, STTC) tespitinde avantaj sağlayabilir. Ancak Random Forest'in özellik önemliliği analizi yeteneği, modelin interpretability açısından klinik uygulamalarda daha değerli olabilir. Her iki model de klinik karar destek sistemlerinde kullanılabilir düzeyde performans sergilemektedir ve tüm sınıflarda 0.88+ ROC-AUC değerleri, modellerin güçlü bir ayrıştırma kapasitesine sahip olduğunu göstermektedir.

