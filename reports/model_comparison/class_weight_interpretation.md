# Class-Weight Yaklaşımı Sonuçları ve Yorumlama

## Sonuçlar Özeti

Tablo X'te Random Forest ve SVM modellerinin class-weight yaklaşımı ile Top-200 özellik seti kullanılarak elde edilen performans sonuçları sunulmaktadır.

### Genel Performans Karşılaştırması

**SVM modeli** genel olarak daha iyi performans göstermiştir. SVM'in F1-Score değeri (0.746) Random Forest'ten (0.732) daha yüksektir. Ayrıca SVM'in recall değeri (0.790) Random Forest'ten (0.756) belirgin şekilde daha yüksektir, bu da SVM'in pozitif vakaları tespit etmede daha başarılı olduğunu göstermektedir. ROC-AUC değerleri de SVM lehine (0.922 vs 0.915) olup, modelin genel ayırt etme yeteneğinin daha güçlü olduğunu işaret etmektedir.

**Random Forest modeli** ise precision değerinde (0.713) SVM'den (0.711) biraz daha yüksek performans göstermiştir. Bu, RF'in yanlış pozitif oranını biraz daha düşük tuttuğunu göstermektedir. RF'in özellik önemliliği analizi yeteneği, modelin interpretability açısından avantaj sağlamaktadır.

### Sınıf Bazlı Performans Analizi

**NORM (Normal) sınıfı:** Her iki model de bu sınıf için yüksek performans göstermiştir. SVM (F1: 0.860) ve RF (F1: 0.852) benzer sonuçlar üretmiştir. Normal ECG sinyallerinin diğer patolojik durumlardan ayırt edilmesi nispeten kolaydır.

**MI (Myocardial Infarction) sınıfı:** SVM (F1: 0.749) RF'den (F1: 0.738) biraz daha iyi performans göstermiştir. MI sınıfının tespiti kritik öneme sahiptir ve her iki model de klinik uygulamada kullanılabilir düzeyde performans sergilemektedir.

**STTC (ST/T Change) sınıfı:** SVM (F1: 0.744) ve RF (F1: 0.738) benzer performans göstermiştir. SVM'in recall değeri (0.846) RF'den (0.791) daha yüksektir, bu da STTC vakalarını tespit etmede SVM'in daha hassas olduğunu göstermektedir.

**CD (Conduction Disturbance) sınıfı:** SVM (F1: 0.748) RF'den (F1: 0.735) biraz daha iyi performans göstermiştir. Her iki model de bu sınıf için kabul edilebilir düzeyde performans sergilemektedir.

**HYP (Hypertrophy) sınıfı:** Tüm sınıflar içinde en zor sınıf olmuştur. SVM (F1: 0.630) RF'den (F1: 0.597) daha iyi performans göstermiştir. SVM'in recall değeri (0.702) RF'den (0.637) daha yüksektir. Bu durum, HYP sınıfının daha az örnek içermesi ve diğer sınıflarla örtüşen özelliklere sahip olmasından kaynaklanıyor olabilir.

## Class-Weight Yaklaşımının Çalışma Prensibi

Class-weight yaklaşımı, sınıf dengesizliği problemini çözmek için kullanılan bir yöntemdir. Bu yaklaşımda, model eğitimi sırasında azınlık sınıflarına (örneğin HYP: 2,119 örnek, CD: 3,907 örnek) daha yüksek ağırlık verilirken, çoğunluk sınıfına (NORM: 7,596 örnek) daha düşük ağırlık verilir. Ağırlık hesaplaması, sınıfın ters orantılı frekansına göre otomatik olarak yapılır (n_samples / (n_classes * np.bincount(y))). Bu sayede model, azınlık sınıflarındaki örnekleri daha iyi öğrenir ve sınıf dengesizliğinden kaynaklanan yanlılığı azaltır. Random Forest için `class_weight='balanced'` parametresi kullanılarak, her sınıf için otomatik ağırlık hesaplanır. SVM için de benzer şekilde `class_weight='balanced'` parametresi kullanılır ve SVM'in margin maximization prensibi, ağırlıklı örneklerle çalıştığında azınlık sınıflarına daha fazla önem verir. Bu yaklaşımın avantajı, veri kaybı olmadan sınıf dengesizliğini ele alması ve tüm veriyi kullanmasıdır.

