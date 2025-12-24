# Stratified K-Fold Cross-Validation Açıklaması

## Veri Bölme Stratejisi

**PTB-XL veri seti**, önceden tanımlanmış stratified fold'lara sahiptir. Veri setinde `strat_fold` kolonu bulunmaktadır ve bu kolon, her örneğin hangi fold'a ait olduğunu belirtir.

## Kullanılan Bölme Stratejisi

Kodumuzda **stratified train/validation/test split** kullanılmaktadır:

- **Train seti:** `strat_fold` değerleri 1-8 arası olan örnekler (yaklaşık %80)
- **Validation seti:** `strat_fold` değeri 9 olan örnekler (yaklaşık %10)
- **Test seti:** `strat_fold` değeri 10 olan örnekler (yaklaşık %10)

## Stratified Yaklaşım

PTB-XL veri setinin orijinal yapısında, `strat_fold` kolonu sınıf dağılımını koruyacak şekilde oluşturulmuştur. Bu sayede:

1. **Sınıf dengesizliği korunur:** Her fold'da sınıf dağılımı orijinal veri setine benzer şekilde kalır
2. **Bias azaltılır:** Stratified yaklaşım, rastgele bölme yapıldığında oluşabilecek bias'ı azaltır
3. **Tutarlı değerlendirme:** Validation ve test setlerinde sınıf dağılımı train setine benzer olduğu için, model performansı daha tutarlı değerlendirilir

## K-Fold Cross-Validation Kullanılıyor mu?

**Hayır, k-fold cross-validation kullanılmıyor.** Bunun yerine:

- PTB-XL veri setinin önceden tanımlanmış fold yapısı kullanılıyor
- Bu, veri setinin orijinal tasarımına uygun bir yaklaşımdır
- Train/validation/test split, literatürde yaygın olarak kullanılan bir stratejidir

## Kodda Nasıl Uygulanıyor?

```python
# src/preprocessing/03_split_data.py
train_df = df[df['strat_fold'].isin([1, 2, 3, 4, 5, 6, 7, 8])].copy()
val_df = df[df['strat_fold'] == 9].copy()
test_df = df[df['strat_fold'] == 10].copy()
```

Bu yaklaşım, PTB-XL veri setinin orijinal tasarımına uygundur ve sınıf dağılımını koruyarak stratified bir split sağlar.

