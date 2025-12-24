# PTB-XL+ Veri Seti ve Stratified Fold Yapısı

## Önemli Açıklama

**Evet, tam olarak doğru anlıyorsunuz!**

PTB-XL+ veri seti, **zaten önceden tanımlanmış stratified fold'lara sahiptir**. Veri setinin orijinal yapısında `strat_fold` kolonu bulunmaktadır ve bu kolon, her örneğin hangi fold'a ait olduğunu belirtir.

## Ne Yaptık?

**Biz stratified fold oluşturmadık.** Sadece PTB-XL+ veri setinin orijinal yapısında bulunan `strat_fold` kolonunu kullanarak veriyi böldük:

```python
# src/preprocessing/03_split_data.py
train_df = df[df['strat_fold'].isin([1, 2, 3, 4, 5, 6, 7, 8])].copy()
val_df = df[df['strat_fold'] == 9].copy()
test_df = df[df['strat_fold'] == 10].copy()
```

## PTB-XL+ Veri Setinin Yapısı

PTB-XL+ veri seti şu dosyalardan oluşur:
- `ptbxl_database.csv`: Metadata ve `strat_fold` kolonu içeren ana veri seti
- `12sl_features.csv`: 12SL analiz aracından çıkarılan özellikler

`ptbxl_database.csv` dosyasında `strat_fold` kolonu **zaten mevcuttur** ve bu kolon:
- Sınıf dağılımını koruyacak şekilde (stratified) oluşturulmuştur
- Her örneğe 1-10 arası bir fold numarası atanmıştır
- Literatürde yaygın olarak kullanılan standart bir yaklaşımdır

## Stratified Fold'un Anlamı

`strat_fold` kolonu, PTB-XL+ veri setinin orijinal yapısında bulunur ve:
- **Stratified (katmanlı)**: Her fold'da sınıf dağılımı orijinal veri setine benzer şekilde korunur
- **10 fold**: Veri seti 10 fold'a bölünmüştür
- **Standart kullanım**: 1-8 train, 9 validation, 10 test olarak kullanılır

## Sonuç

- ✅ PTB-XL+ veri seti zaten stratified fold'lara sahip
- ✅ `strat_fold` kolonu veri setinin orijinal yapısında mevcut
- ✅ Biz sadece bu kolonu kullanarak veriyi böldük
- ✅ Stratified yaklaşım veri setinin orijinal tasarımına uygun
- ✅ Literatürde yaygın olarak kullanılan standart bir yöntem

**Makalede belirtilmesi gereken:** "PTB-XL+ veri seti, önceden tanımlanmış stratified fold'lara sahiptir. Veri setinin orijinal yapısında bulunan `strat_fold` kolonu kullanılarak, fold 1-8 train seti, fold 9 validation seti ve fold 10 test seti olarak ayrılmıştır."

