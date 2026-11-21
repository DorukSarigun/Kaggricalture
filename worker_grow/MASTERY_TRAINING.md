# Crew başarı müfredatı

Giriş noktası: `python -m worker_grow.train_mastery`. Eski `train_crew`, joint
manager ve ekonomik DAG ödülleri varsayılan davranışlarını korur. Bu koşu
yalnız crew becerilerini öğretir; kâr veya yarışma kazanma deneyi değildir.

## Sorumluluk

Sabit öğretmen ürün, tile ve ilk teslimat kotasını verir. Bütün birimlerin
(ana çiftçi dahil) kısa iş seçimi, yürümesi, bakımı, hasadı ve teslimi ağdan
çıkar. Öğretmen yön veya birim aksiyonu vermez. İşçiler geceleri oyunun
kuralıyla ayrılır; öğretmen gündüz gerçek HIRE emriyle eksik sayıyı tamamlar.
Başlangıç tohumu, hayvan, yem ve işe alım parası açıkça gifted malzemedir.
Oyun kuralları, gece otomatik teslimi ve biyolojik süreler değiştirilmez.

## Müfredat

1. Hazır buğday, çiftçi hedefte: hasat ve teslim.
2. Hazır buğday, hedefe iki kare uzakta.
3. Daha uzak ve çeşitli konumlar.
4. Büyüyen buğday: bakım → hasat → teslim.
5. Tek tohumdan tam buğday zinciri.
6. İki birim ve iki hazır ürün: iş bölümü.
7. İki birim ve iki ekim görevi.
8. Kaz yerleştirme, yemleme ve yumurta teslimi.
9. İnek ve süt.
10. Koyun ve yün.
11. Üç birim, üç görev; tüm bitki ve hayvan türlerinden karışık örnekler.

Tile ve başlangıç konumları her episode deterministik seed ile örneklenir.
Görevler kapanınca episode erken biter. Derslerin farklı, biyolojik sürelere
yeterli üst sınırları vardır; her ders 720 tur oynamaz.

Her 10 update'te ayrı 20 değerlendirme oyunu oynanır. İki ardışık değerlendirmede
oyunların en az %90'ında **bütün görevler** tamamlanırsa ilerlenir. İlk üç
derste başarılı oyunların ortalama süre sınırı da vardır (12/18/28 tur).
Geceyi beklemek yasal başarıdır fakat yakın mesafe hız sınırını geçirmeyebilir.
Başarısız değerlendirme sayacı sıfırlar; episode sayısı kendi başına ilerletmez.
Önceki dersler %15 olasılıkla tekrar edilir. Son ders de geçilirse eğitim erken
tamamlanır; bütçe biterse dosya `mastered=false` sonucunu açıkça kaydeder.

Değerlendirme stokastik politikayla ve eğitimden ayrık seed'lerle yapılır.
Değerlendirme satırları optimizere girmez; model ağırlıkları değişmez ve RNG
durumu geri yüklenir. Bu bir başarı eşiğidir, %90 gerçek başarı için istatistiksel
güven aralığı iddiası değildir.

## Ödül ve kredi

Varsayılanlar:

| Kaynak | Miktar | Alıcı |
|---|---:|---|
| Kotaya uygun ilk gerçek depo teslimi | birim başına +2 | Mevcut fiziksel ürün/aksiyon DAG'ı |
| Doğrulanmış kısa üretim işi | +0.05'ten başlayıp dersle azalır | İşlemin kendisi ve doğrulanmış yaklaşma segmenti |
| Bir issue'nun toplam ara destek üst sınırı | +0.25 | Önceki satırların toplamına sınır |
| Yürüme denemesi | −0.002 | O hareket kararı, yalnız bir kez |
| Seçim yapılabilen birim kararı | −0.001 | O karar; iş varken seçilen PASS dahil |

Tek yürüme kararı toplam −0.003 alabilir: hareket ve karar zamanı iki ayrı,
loglanan maliyettir. Seçeneksiz biyolojik beklemelere karar maliyeti yazılmaz.
Ara iş desteği aynı issue/üretici/işlem/gün için yeniden verilmez. Hedef
değiştirerek kotayı yenilemek mümkün değildir. Son karışık derste ara destek 0.
Ara destek bir beceri ödülüdür, gerçekleşmemiş satış kârı değildir; ürün
sonradan kaybedilirse geri alınmaz, fakat issue başına sınırı teslim ödülünün
çok altındadır. Satış, satın alma ve terminal gold crew hedefi değildir.

Kayıt defterinde `credit` tek kez atanmış ödüldür; toplamı episode ödülüne eşittir.
PPO için ayrı `target_return`, **aynı işçinin aynı kısa görev segmentinde kalan
credit toplamıdır** (gamma=1). Böylece ilk iş seçimi sonraki yaklaşma maliyetini
görür. Başka işçinin veya yerine seçilen başka görevin ödülü bu toplama girmez.
Terk edilen segment kendi maliyetleriyle kalır; yeni işe sahte soy bağı kurulmaz.
Hedeflerin toplamı defter toplamına eşit olmak zorunda değildir: bunlar tekrar
ödeme değil, karar anından itibaren kalan getiri tahminleridir.

İş seçimi ve ona koşullu tek primitive emir ortak log-olasılıkla eğitilir.
Bu düzen ayrı işçi/market kararlarını tek turn avantajında toplamaz; yine de
iş seçimi ile primitive arasında karşıolgusal marjinal katkı ölçümü yapmaz.
Herhangi bir görev değişimine kör ceza veya mesafeye bağlı potansiyel yoktur.
Ortak çok-hedefli rotalar için yeni bir planlayıcı eklenmemiştir.

## Kayıt ve güvenlik

- `metrics.jsonl`: teslim, görev başarısı, yürüyüş, süre, terk edilen segmentler,
  review dersi, optimizasyon ve bağımsız değerlendirme.
- `episodes/.../trace.jsonl.gz`: observation, gerçek aksiyon, next observation.
- `decisions.json.gz`: seçilen kısa görev, maskeler, credit ve target_return.
- `delivery_credit.json`: dağıtım, yaklaşma kayıtları ve bütün maliyetler.
- `evaluations/...`: ayrı ve her oyun için replay dahil tam değerlendirme izi.
- `updates/.../crew_batches.jsonl`: kullanılan/reddedilen PPO parçaları.
- `latest.pt`: model, optimizer, müfredat sayacı ve RNG; atomik yenileme.
- `mastered_XX.pt`: başarı eşiğini geçen ders checkpoint'i.
- `source/` ve kaynak hash'leri: worker_grow kodunun koşu başlangıcındaki hali.

Resume, ödül/müfredat/kaynak farklılığını veya checkpoint'e girmemiş oyunları
sessizce kabul etmez. Eski koşular silinmez. Eski reward checkpoint'i sadece
açıkça `--init-crew` verilirse weights-only alınır; varsayılan temiz modeldir.

## Uzun CUDA koşusu

```bash
.venv/bin/python -m worker_grow.train_mastery \
  --run-dir runs/crew_mastery_cuda_01 \
  --episodes 5000 --episodes-per-update 8 \
  --eval-every 10 --eval-episodes 20 \
  --promotion-threshold 0.9 --promotion-passes 2 \
  --learning-rate 3e-5 --epochs 3 --sequence-length 24 \
  --credit-decay 0.95 --walk-cost 0.002 --decision-cost 0.001 \
  --progress-cap 0.25 --delivery-bonus 2 \
  --entropy-coef 0.01 --target-kl 0.02 --max-grad-norm 1 \
  --save-every 20 --device cuda
```

Bu bir **en fazla 5000 eğitim oyunu** bütçesidir; değerlendirmeler ayrıca oynanır.
Başarı yoksa zor dersi açmak yerine mevcut derste kalır.
