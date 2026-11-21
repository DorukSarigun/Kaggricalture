# Tile seçimi

Güncel ön eğitim, alt görevler ve checkpoint geçişi: [v4](SUBTASK_TRAINING.md).

Öğrenilmiş manager hâlâ `['ISSUE', 'WHEAT']` (tile head yok; sonra eklenecek).
Scripted teacher hayalet bantta chassis unit’lerinin durduğu kareyi
`ISSUE.tile_xy` yazar. Canlı crew ayağı pin değildir. Canlı tarlada o kare
boş/açık değilse ISSUE açılmaz.

## Bağlama

1. Scripted ISSUE `tile_xy=pin` ve PLANNED olarak açılır. Öğrenilmiş manager
   hâlâ `tile_xy=null` açabilir. Açık issue sayısı crew boyutuyla sınırlı
   değildir. Crew eğitiminde eşzamanlı tavan episode ile açılır:
   `--issue-unlock-initial` (varsayılan 1) ilk `--issue-unlock-interval`
   (varsayılan 10) oyunda; her sonraki 10 oyunda +1. Tamamlanan issue yuva
   boşaltır. `interval=0` tavanı kapatır.
2. Pinli issue’da crew o kareye yürür ve eker/yapar. Pinsiz issue’da yer
   seçimi hâlâ crew ekiminin sonucudur.
3. Aynı üründen birden çok **pinsiz** açık issue varsa **en eski uygun,
   bağlanmamış issue** seçilir. Pinli issue yalnız kendi karesinde bağlanır.
4. BUILD_PASTURE, COW/SHEEP talepleri arasında en eski uygun issue'yu bağlar.
   Sonraki PLACE bu issue'nun türüne uymalıdır. Hazır boş yapıya PLACE yapılırsa
   bağlama doğrudan o olayda gerçekleşir. DIG tek başına issue bağlamaz.
5. Cursor yalnız kopya görev defterinde rezervasyon tutar. Aynı turda ikinci
   işçi ilk işçinin seçtiği tile'ı ve tükettiği issue'yu görür. Tek issue iki
   tile'a ekilemez. Sonraki işçi yeni ekilmiş aynı bitkiyi sulayabilir.
6. Kalıcı bağlama yalnız gerçek STRUCTURE_BUILT / PLANTED / ANIMAL_PLACED
   olayında olur. Geçersiz/no-op eylem, yürüyüş veya model örneklemesi bağlamaz.
7. `tile_xy`, `bound_step`, `bound_decision_id` teslimat raporunda ve manager
   issue kayıtlarında tutulur. Bound karar kimliği crew ID'sidir; issue'nun
   `decision_id` alanı manager'ın ISSUE kararı olarak kalır.

Üretici kimliği atanıp issue sync edildikten sonra mevcut FIFO/DAG sistemi
aynı issue üzerinden hasat, ara ürün ve satışları takip eder. Üretici kaybı,
görev kapanışı ve satış ödülü kuralları değiştirilmedi.

## Crew gözlemi

Görev haritası 18 kanal: 8 bağlı/pinli ürün türü, 1 kalan kota oranı,
1 üretici başladı bilgisi, 8 `tile_xy=null` talep sayısı. Pinli ISSUE açılınca
tür kanalı o karede yanar; broadcast yalnız pinsiz talebe kalır.
Görev vektörü bu harita + idle biti üzerinden normalize edilir (1801 boyut).
Sabit tile'lı `ProductionTask` test/curriculum desteği korunur; manager
politikaları bu yolu kullanmaz.

## Checkpoint ve koşular

- Crew: `feudal-contract-crew-site-v3`
- Manager: `feudal-manager-crew-site-v2`
- Joint: `feudal-joint-crew-site-v3`

Eski checkpoint'ler farklı head/girdi boyutu nedeniyle reddedilir. `--init-crew`
aynı mimarideki `site-v2` ağırlığını `v3` koşuya alır; `--resume` tam sürüm ister.

```bash
.venv/bin/python -m worker_grow.train_crew \
  --run-dir runs/feudal_crew_best_agent_site_02 \
  --overwrite \
  --stage best_agent --episode-steps 720 --episodes 100 \
  --issue-unlock-interval 10 --issue-unlock-initial 1 \
  --grow-event-reward 1 --delivery-bonus 2 \
  --episodes-per-update 4 --learning-rate 3e-5 --epochs 3 \
  --sequence-length 16 --credit-decay 0.8 --save-every 10 --device cuda
```

Yeni crew checkpoint'i sonraki yeni joint veya `best_agent` koşuya `--init-crew` ile aktarılabilir.
Önceden başlatılmış süreçler bu dosya değişikliğiyle yeni sisteme geçmez.

## Mini tutorial pretrain (ayrı mod)

`python -m worker_grow.train_crew` aşamalarına dokunmaz. Kısa hediye sahneler
`python -m worker_grow.train_tutorials` ile çalışır: drop → harvest →
walk_harvest → water → plant, sonra aynı ağırlık `best_agent`'e:

```bash
.venv/bin/python -m worker_grow.train_tutorials \
  --run-dir runs/feudal_crew_tutorials_01 \
  --episodes 100 --slice-episodes 20 \
  --grow-event-reward 1 --delivery-bonus 2 --episodes-per-update 4 \
  --learning-rate 3e-5 --epochs 3 --sequence-length 16 \
  --credit-decay 0.8 --save-every 10 --device cuda
```

```bash
.venv/bin/python -m worker_grow.train_crew \
  --run-dir runs/feudal_crew_best_agent_site_02 \
  --init-crew runs/feudal_crew_tutorials_01/latest.pt \
  --stage best_agent --episode-steps 720 --episodes 100 \
  --issue-unlock-interval 10 --issue-unlock-initial 1 \
  --grow-event-reward 1 --delivery-bonus 2 \
  --episodes-per-update 4 --learning-rate 3e-5 --epochs 3 \
  --sequence-length 16 --credit-decay 0.8 --save-every 10 --device cuda
```

`--delivery-bonus` resume sırasında eşleşmek zorunda değil; 2×’e geçmek için çalışan süreci durdur, checkpoint’ten sonraki yarım episode klasörlerini arşivle, sonra:

```bash
.venv/bin/python -m worker_grow.train_crew \
  --run-dir runs/feudal_crew_best_agent_site_02 \
  --resume \
  --stage best_agent --episode-steps 720 --episodes 100 \
  --issue-unlock-interval 10 --issue-unlock-initial 1 \
  --grow-event-reward 1 --delivery-bonus 2 \
  --episodes-per-update 4 --learning-rate 3e-5 --epochs 3 \
  --sequence-length 16 --credit-decay 0.8 --save-every 10 --device cuda
```

`--overwrite` kullanma.

## Bu değişikliğin kapsamı dışında

Teslimat kotası duruyor; kotanın shed’e ilk birimi hâlâ DAG ile pay edilir.
O ödeme `--delivery-bonus` (varsayılan 2) ile çarpılır: ürünü baştan sona
shed’e götürmek grow olaylarından daha değerli olsun diye. Grow olayları
1× kalır. Buna ek, bağlanmış üreticiyi gerçekten ilerleyen her crew olayı
(`PLANTED` / yapı / `PLACE` / `WATER` / `FEED` / `CARE` / `FERTILIZE` /
`HARVEST` / `COLLECT_FERTILIZER`) o karara `--grow-event-reward` (varsayılan 1)
yazar. No-op, yürüyüş, `DIG`, `FAILED`/`EXPIRED` kapanışı ve gece otomatik
olay 0. `--grow-event-reward 0` eski yalnız-teslimat hedefini açar.

Scripted teacher hâlâ takvimden ürün talepleri ve market emirleri çıkarıyor; bağımsız adaptif
ekonomik planlayıcıya dönüştürülmedi. Scripted marketin kendi uygulanmayan
unit emirleriyle depo tahmini yapması bu patch'te çözülmedi. Bunlar ayrı işler.
