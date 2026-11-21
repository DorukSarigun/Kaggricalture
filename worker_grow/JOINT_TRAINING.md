# Manager + crew ortak eğitim

Giriş: `worker_grow.train_joint`, checkpoint sözleşmesi `feudal-joint-issue-gold-dag-v9`.
Görev/işçi sınırı yoktur. Gözlenen görev maliyetleri ve log formatı: [MANAGER_TRAINING.md](MANAGER_TRAINING.md).
Manager artık tile seçmez: [crew-site sözleşmesi](CREW_SITE.md).
İki ajan aynı takımda aynı oyunu oynar; rakip `starter` olarak kalır.
İki ayrı bağımsız training süreci başlatmak bu kurulumun yerine geçmez.
Tek ürün kısıtı `--issue-items` ile [PRODUCT_PATHS.md](PRODUCT_PATHS.md) üzerindedir.

```bash
.venv/bin/python -m worker_grow.train_joint \
  --run-dir runs/feudal_joint_crew_site_cuda_01 \
  --episodes 10000 --episode-steps 720 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --sequence-length 16 \
  --manager-learning-rate 3e-5 --crew-learning-rate 3e-5 \
  --gae-lambda 1.0 --credit-decay 0.8 \
  --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 \
  --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 \
  --save-every 10 --opponent starter --device cuda --seed 2026
```

## Birlikte öğrenme ne demek?

1. Manager ne üretileceğini `ISSUE → ürün` kararıyla belirler; tile seçmez.
2. Ana çiftçi ve o tur mevcut bütün işçiler **öğrenen crew** tarafından yönetilir.
   Crew tile'ı ekim/yapı kurma sırasında seçer. Çalışanlar sırayla, önceki çalışanın
   işlemi ve issue rezervasyonu uygulanmış shadow'u görür.
3. Manager ekip işlemleri sonrası shadow üzerinden market emirlerini seçer.
4. Gerçek motor geçişi, nakit defteri, FIFO lifecycle ve gerçek teslimatlar kaydedilir.
5. Dört episode boyunca iki politika da sabit tutulur. Sonra manager PPO kendi
   kararlarıyla, crew recurrent PPO kendi kararlarıyla güncellenir. İki optimizer
   sırayla çalışır; sonraki oyunlar **iki güncellenmiş politikayla** başlar.

Crew parametreleri dondurulmaz. Modeller parametre paylaşmaz; crew loss'u manager'a
geri yayılmaz. Hedefler manager'ın somut kontratlarıdır, tam FuN makalesindeki
latent yön/transition-policy-gradient sistemi değildir. Eski `train_manager`
komutunun `--crew feudal` seçeneği hâlâ yalnız donmuş crew ile manager eğitir.

## Gerçek satış ve issue DAG'ı

Ayrıntılı kural ve örnekler: [ISSUE_DAG.md](ISSUE_DAG.md).

| Ajan | Hedef |
|---|---|
| Manager | Terminal altın farkı; gerçek satış DAG kredisi; satılmayan girdilerin maliyeti satın alma kararına; başlanamayan iş ve aksayan günlük bakım cezası ISSUE kararına |
| Crew | Görev kotasına kadar ilk gerçek depo teslimatı; kendi karar grafiğine geriye azalan kredi; ayrı critic |

Crew için ikinci GAE yok. Manager terminal GAE `lambda=1` ile tam bölüm hedefini
kullanır. Crew'da teslimat yoksa teslimat kredisi sıfırdır; critic ve entropy
güncellemeleri yine mümkündür. Sıfırdan iki ajanla seyrek teslimat problemi devam
eder; çalışan optimizer, başarılı üretim öğrenildiği anlamına gelmez.

**Alınıp issue'ya dönüşmeyen tohumlar:** episode sonunda gerçek alış maliyeti
BUY_SEED kararına negatif yazılır. Teslimat/issue kapanışı manager'a ödül üretmez.
Satılmış ürünün maliyeti zaten net kârdan düşüldüğü için BUY'da tekrar düşülmez.
Satış geliri de ISSUE paylarına dağıtıldıktan sonra SELL'e tekrar yazılmaz.
Saf market al-sat ve HIRE/BUY_LAND, üretim issue ağacından ayrı muhasebeleştirilir.
Eski fiyat değerlemesi kaldırılmıştır; maliyet muhasebesinin ortak çıktı birimlerine
paylaşım kuralı ISSUE_DAG belgesinde açıkça yazılıdır.

## Kayıtlar

- `metrics.jsonl`, `train.log`: episode altını, teslimatlar, görev sonuçları;
  güncellemede **manager ve crew ayrı** loss/KL/entropy/gradient norm/kullanılan-atlanan sayıları.
- Her episode `trace.jsonl.gz`: observation, sonraki observation, gerçek action,
  manager/crew karar ID'leri ve teslimatlar. `moves.jsonl`: emirler ve issue akışı.
- `manager_decisions.json`, `crew_decisions.json`: karar, maske, eski log olasılığı,
  critic tahmini, kredi ve advantage. ID'ler ajan/episode ad alanları içinde yereldir.
- `delivery_credit.json`, `cash_ledger.json`, `issues.json`, lifecycle dosyaları:
  ödül ve kaynak eşleştirmesi denetimi.
- `issue_production_graph.json`, `issue_sale_credit.json`: ürünlerin tüm dalları,
  gerçek satış kârı, issue payları ve satın alma maliyeti mutabakatı.
- `issue_task_cost.json`: ham/nihai görev maliyeti, kapanış DAG'ı ve kaynak kanıtları;
  crew ödülüne eklenmez. `issue_completion.json`: benzersiz tamamlanma ve ürün miktarları.
- `updates/update_*/manager_batches.jsonl`: kullanılan/atlanan her minibatch,
  kararların episode ID'leriyle. `crew_batches.jsonl`: recurrent parçalar ve
  `episode_in_update`; bu indeks `episodes.json` listesindeki episode'a karşılık gelir.
- `--save-every 10`: her 10 oyunda `replay.json.gz`. Trace her oyunda yazılır;
  uzun eğitimde disk kullanımını izlemek gerekir.

Tek `latest.pt`, **iki modeli, iki optimizer'ı, ortak episode/update numarasını
ve Python/NumPy/CPU/CUDA RNG durumlarını** içerir. İki güncelleme de başarılı
olmadan checkpoint ilerletilmez. Yazım geçici dosyadan atomik rename ile yapılır.
Tamamlanan oyunlar henüz güncellenmemişse ya da bir update yarıda kalmışsa resume
bu artıkları sessizce ezmez; arşivleme gerektirerek durur.

Devam etmek için aynı komuta `--resume` ekle. `--episodes` toplam hedefi artırabilir;
ödül veya optimizer ayarları aynı kalmalıdır. Bağımsız crew/manager koşuları ortak
resume olarak kabul edilmez. Yeni ortak koşuya yalnız ağırlık aktarmak için
`--init-crew PATH` ve/veya `--init-manager PATH` kullanılabilir; iki ajan da sonrasında
öğrenir, optimizer'lar sıfırdan başlar. Varsayılan sıfırdan eğitim, hediyesiz gerçek
başlangıçtır. Mevcut çalışan koşular otomatik durdurulmaz.

## Önceki altyapı doğrulaması — 14 Eylül 2026 (v1)

- Tüm test paketi: **201 test geçti**. Ortak eğitim testleri iki modelin gerçekten
  değişmesini, manager güncellemesinin crew ağırlıklarına dokunmamasını, yerel karar
  kimliklerini, kullanılmayan alış cezasını ve yarım koşu korumasını kapsar.
- CPU'da kesintisiz eğitim ile checkpoint üzerinden devam edilen eğitimin iki
  model ağırlıkları ve Torch RNG durumu birebir aynı çıktı.
- CUDA: iki kısa episode + iki optimizer güncellemesi; resume ile üçüncü episode
  + iki optimizer güncellemesi tamamlandı. Atlanan manager minibatch/crew chunk yok.
- CPU: `episode-steps=720` gerçek maç (motorun kaydettiği 719 geçiş), iki optimizer
  güncellemesi ve replay yazımı tamamlandı. Atlanan güncelleme parçası yok.
- Bu kısa sıfırdan koşularda teslimat ödülü **0**: testler ekonomik başarı veya
  yeterli keşif kanıtı değildir. Uzun ortak eğitim başlatılmadı.
