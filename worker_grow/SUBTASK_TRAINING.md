# Scripted manager + öğrenen alt görevli crew

Yeni başarıya bağlı ön eğitim: [Crew mastery müfredatı](MASTERY_TRAINING.md).
Bu belgedeki `train_crew` eski, otomatik issue artıran koşuyu anlatır.

## Sorumluluklar

Scripted teacher issue'ya ürün ve tile verir. Tile canlı crew'ün ayağı değil,
teacher'ın ayrı plan simülasyonundaki hedeftir. Canlı oyunda geçersiz/ayrılmış
tile atlanır. Eğitimde her issue'nun tile'ı zorunlu; config `scripted_ghost_tile`
yazar. Teacher hâlâ takvim tabanlıdır; adaptif ekonomik planlayıcı değildir.

Crew her işçi kararında önce motor kurallarıyla yapılabilir alt işleri çıkarır:
belirli issue/tile'da PLANT, BUILD, WATER, FEED, HARVEST vb.; eksik girdi için
PICKUP; kendi yükü için DROP. Öğrenen assignment head'i bunlardan veya WAIT'ten
birini seçer. Seçim eylem head'lerini koşullandırır. Crew yürüyüş yönlerini
öğrenir; scripted en-kısa-yol oracle'ı yoktur.

Seçim her tur değişebilir, işçi issue bitene kadar kilitlenmez. Aynı issue'nun
farklı alt işleri farklı işçilerce üstlenilebilir. Aynı alt işe çift atamayı
geçici rezervasyon önler. PICKUP için farklı depo girişleri aynı işi iki kez
üstlenmeyi sağlamaz. DROP işçinin kendi yüküdür, girişler ortak kullanılabilir.
Stale işçi rezervasyonları zaman aşımına uğrar, gün değişiminde temizlenir.
Başarılı shadow işleminden sonra rezervasyon bırakılır; ödül gerçek olay ister.
İptal edilen/artık yapılabilir olmayan işler aday listesinden çıkar.

## PPO

`logπ = logπ(alt iş) + logπ(op | alt iş) + logπ(item | ...) + logπ(q | ...)`.
Alt görev seçimi yalnız log etiketi değildir. Davranış anındaki aday özellikleri
ve seçim saklanır; PPO replay farklı aday listesi üretmez. Critic seçim öncesi
durumu değerlendirir. Manager ve crew optimizer'ları ayrı kalır.

## Yol ve kredi

Kararlar assignment_id, issue, hedef tile, işlem ve bilinen üretici kimliğini
taşır. Aynı alt iş sürerken yürüyüş aynı segmentte birikir. İş değişince yeni
segment açılır; eski yol yeni ürüne geçirilmez. Gerçek işlemde issue, işçi,
tile ve üretici doğrulanırsa yaklaşma yolu üretim DAG'ına bağlanır. Salt
yaklaşma/görev seçme ödül üretmez. Taşınan gerçek lotların fiziksel geçmişi
görev değişse de korunur. Gece otomatik teslimat hayali DROP üretmez.

Şimdilik tek aktif alt iş seçilir. Önceden seçilmiş çok-hedefli rotanın ortak
boş-yük yolunu iki issue'ya bölen ek bir planlayıcı yoktur; yakınlık tahmininden
pay üretilmez.

Crew toplam kredisi = mevcut grow-event + teslimat DAG kredisi − yürüyüş maliyeti.
`--walk-cost 0.001` deneme başlangıcıdır; optimum olduğu iddia edilmez. Her
yürüyüş girişimi kendi kararına bir kez yazılır; no-op da sayılır. Hareketin
birden çok lot geçmişinde bulunması maliyeti çoğaltmaz. `--walk-cost 0` maliyeti
kapatır. Teslimat kotası ve manager ekonomik paylaşımı değiştirilmedi.

## Devam: weights-only, eski optimizer değil

Site-v2/v3 encoder/GRU/eylem/critic ağırlıkları korunur. Yeni assignment
score/condition katmanları yeni başlar; condition sıfır başlatılır. Eski
anahtarlar eksik/fazla ise aktarım reddedilir. Yeni model `feudal-crew-subtasks-v4`.
Eski koşu resume edilmez; ayrı dizin/yeni optimizer kullanılır. Yeni 200 oyun:

```bash
.venv/bin/python -m worker_grow.train_crew \
  --run-dir runs/feudal_crew_subtasks_01 \
  --stage best_agent \
  --init-crew runs/feudal_crew_best_agent_site_02/latest.pt \
  --episode-steps 720 --episodes 200 --episodes-per-update 4 \
  --learning-rate 3e-5 --epochs 3 --sequence-length 16 \
  --credit-decay 0.8 --walk-cost 0.001 \
  --grow-event-reward 1 --delivery-bonus 2 \
  --issue-unlock-initial 1 --issue-unlock-interval 10 \
  --save-every 10 --device cuda
```

Görev sayısı 1'den tekrar artırılır; yeni atama head'i önce az işte öğrenir.
Hediye varlık yok; rakip PASS, yeni ot çıkışı kapalı. Yarışma değerlendirmesi
değildir. Yeni v4 checkpoint kendi koşusunda resume edilebilir; ödül
katsayıları sessizce değiştirilemez. Uzun koşu başlatma kullanıcı komutudur.

## Loglar / ölçüm

Her oyunda observation/action trace, alt görevler, aday özellikleri, lifecycle,
doğrulanmış yol bağlantıları, tamamlanmayan yol kökleri kaydedilir.
walk_penalties gerçek hareket/no-op ve tekil maliyeti gösterir. PPO batch
loglarında işçi/karar/alt görev/kredi/advantage ve kabul-atlama bilgisi vardır.
Replay save aralığında. Teslimat/oyun, tamamlanan görev oranı, ürün başına
yürüyüş, PASS ve görev değişimi birlikte ölçülmeli; sırf az yürümek başarı değil.
