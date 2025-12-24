# Makale İçin Metodoloji Metni - Güncellenmiş Versiyon

## Kodlama Ortamı ve Deney Akışı

Bu çalışma kapsamında tüm veri hazırlama, görselleştirme ve modelleme adımları Python 3 (çalışma sırasında Python 3.11+) ile gerçekleştirilmiştir. Deneylerde kullanılan temel kütüphaneler; veri işleme için pandas ve numpy, modelleme ve değerlendirme için scikit-learn, görselleştirme için matplotlib ve seaborn olarak belirlenmiştir. Bağımlılıklar proje kökündeki requirements.txt dosyasında merkezi olarak tanımlanmış; böylece ortamın yeniden kurulumu ve sürüm uyumluluğu standartlaştırılmıştır.

Deney akışı modüler betikler üzerinden yürütülmüştür: veri ön işleme hattı `src/preprocessing/` altındaki sıralı betikler (01_merge_and_label.py, 02_clean_columns.py, 03_split_data.py, 04_impute_and_flag.py) ile, öznitelik seçimi `src/analysis/05_feature_selection_rf.py` ile, model eğitimi ve değerlendirmesi ise `src/modeling/` altındaki betikler (özellikle 17_class_weight_comparison.py) ile gerçekleştirilmiştir. Üretilen performans özetleri ve sınıf bazlı metrikler `results/` altında biriktirilerek, raporlanan bulguların doğrudan yeniden üretilebilir artefaktlar üzerinden izlenmesi sağlanmıştır. Görselleştirmeler ve analiz raporları `reports/` dizini altında organize edilmiştir.

Proje dizin yapısı, tekrarlanabilirliği destekleyecek şekilde veri–kod–çıktı ayrımını açıkça kurmaktadır: ham veri kaynakları `ptb-xl/` ve `ptb-xl+/` altında tutulurken, deneylerde kullanılan ara/nihai tablolar `data/processed/` altında sürümlenebilir dosyalar olarak saklanmaktadır. Keşifsel analiz betikleri `src/analysis/`, ön işleme adımları `src/preprocessing/`, model eğitim ve değerlendirme akışı ise `src/modeling/` dizinlerinde konumlandırılmıştır.

Kod, betikler ve deney çıktıları sürüm kontrolü altında tutulmuş ve kamuya açık olarak paylaşılmıştır. Deneylerin yeniden çalıştırılmasına yönelik temel adımlar `README.md` dosyasında açıklanmıştır.

