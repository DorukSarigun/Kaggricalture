# Manager + scripted crew: talep ve gerçek satış sözleşmesi

19 Eylül 2026. Manager checkpoint: `feudal-manager-issue-gold-dag-v7`.
Ortak eğitim checkpoint: `feudal-joint-issue-gold-dag-v9`.
Önceki checkpoint'ler bu mimariye `--resume` ile alınmaz; eski kayıtlar korunur.

## Düzeltilen akış

1. Manager mevcut görev yükünü görür, ürün seçerek ISSUE açar. Tile hâlâ crew'a aittir.
2. Crew, mevcut işçinin altında farklı bir görev olsa da diğer boş karelerdeki
   işleri değerlendirebilir. Unbound FIFO seçimi işçinin anlık karesinden bağımsızdır.
   İlk sekiz aday kare seçilirken başka görevlere ayrılmış kareler elenir.
3. Crew işlemleri shadow'da yürütülür. Manager market kararında hem güncel
   envanteri hem shadow'daki ekim/yerleştirme bağlarını görür. Gerçek IssueBook
   yalnız motorun doğrulanmış olaylarıyla güncellenir.
4. Market emirleri arasında talep özellikleri tekrar hesaplanır. Örneğin buğday
   alınca karşılanmamış buğday girdisi sayısı azalır. PPO aynı dondurulmuş
   özellikleri ve seçim anındaki maskeyi kullanır; güncel IssueBook'a tekrar bakmaz.

## Görev kapasitesi ve ağ girdisi

**İşçi başına görev sınırı kaldırıldı.** `--max-open-per-worker` artık yoktur;
eski argümanı içeren komut hata verir. Tur başına 32 ISSUE kesmesi de kaldırıldı.
İşçi sayısı, tahmini rota veya günlük kapasite ISSUE'yu maskelemez. Manager
yapılabilecek iş miktarını ödülden öğrenir; görev çokluğu tek başına ceza değildir.

Yalnız fiziksel yer uygunluğu kontrolü korunur: boş/temizlenebilir kareler ve
boş uygun hayvan yapıları; konumsuz üretim talepleri aynı yeri iki kere saymaz.
Üretimi bitmiş teslimat kuyruğu geçmiş tile'ı işgal etmez. Fiziksel yer kalmadığında
STOP zorunludur; dolayısıyla ISSUE döngüsü ek sayısal kota olmadan sonlanır.
Tohum/para yokluğu planlamayı yasaklamaz. Tile seçimi hâlâ crew'a aittir.

Ek 52 global talep özelliği: sekiz ürün için PLANNED/ACTIVE/DELIVERING sayıları,
konumsuz görev sayısı, kalan teslimat kotası, eksik başlangıç girdisi; ayrıca
işçi sayısı, açık görev sayısı, henüz üreticisi olmayan görev sayısı, ayrılmamış kare sayısı.
Sayaçlar `log1p` ile verilir; özellik sırası checkpoint config'inde saklanır.
Manager'ın ISSUE/market politikaları **ve iki critic'i** bu girdiyi kullanır.
Mevcut tarla kanalları ve crew modelinin boyutları değiştirilmedi.

## Manager görev maliyeti — gözlenen sonuç, kapasite tahmini değil

`issue_outcomes.py`, manager-only ve ortak eğitimde her **gerçek oyun turu** sonunda
çalışır. Bir issue'nun birden fazla market emri/işçisi olması cezayı çoğaltmaz.

| Durum | Varsayılan manager kredisi |
|---|---:|
| Henüz ekilmemiş/yerleştirilmemiş görev, başlangıç girdisi eksik | −0.0001 / tur |
| Başlangıç girdisi var ama görev hâlâ ekilmemiş/yerleştirilmemiş | −0.0001 / tur |
| Motor gün sonunda sulanmadığını/beslenmediğini doğruladı | −0.001 / issue / gün |
| Normal bitki büyümesi, hayvanın ilk ürününü bekleme | 0 |

`--issue-wait-cost` ve `--missed-care-cost` bu **reward birimi** katsayılarını
ayarlar; `0` ilgili maliyeti kapatır. Bunlar gold değildir, optimal katsayı iddiası
yoktur. Örneğin 100 tur başlanmayan issue −0.01 kredi biriktirir. Yürüyüş/kurulum
süresinin de ilk üretici oluşana kadar bekleme maliyeti vardır; sınırsız kuyruk
yerine manager'ın bu gecikmeyi hesaba katması amaçlanır.

## Tek ürün path'leri

`--issue-items CARROT` (veya `COW`, `WHEAT`, …) ISSUE'yu ve üretim marketini o
ürüne kısıtlar. Ağ boyutu değişmez. Komut listesi: [PRODUCT_PATHS.md](PRODUCT_PATHS.md).
Kısıtsız eğitim flag'i atlar; `config.issue_items` resume kilidindedir.

Yeni issue aynı turda kaynağa kavuştuysa açıldığı tur cezalandırılmaz: motor önce
unit, sonra market çalıştırır; yeni alınan tohum aynı turda ekilemez. Sonraki
turlarda başlamayan işin maliyeti işler. Aynı stok birimi yalnız bir bekleyen
issue'yu karşılar (eski issue önce); bu yalnız hesaplama, fiziksel rezervasyon değil.

**Ceza kapanış DAG'ından manager kararlarına geri yazılır**, crew kararına değil.
Varsayılan kapanış kuralı aşağıda açıklanmıştır; tablodaki değerler önce ham bütçeyi oluşturur.
Gecikmenin
kesin nedeninin kapasite yetersizliği olduğu iddia edilmez; kaynak eksikliği,
uzun yol, kötü crew politikası da neden olabilir. Yük planlama sorumluluğu manager'dadır.
Hasadı daha yüksek verim için bekletmeye veya teslimat kuyruğuna yeni bir süre
cezası eklenmedi. Normal ömür sonu çürümesi, eksik sulama diye etiketlenmez.

Ekonomik DAG ve kasa mutabakatı değişmedi:

```text
local_credit = gerçek ekonomik DAG kredisi
task_credit = tamamlanmayan pay için kapanış DAG'ının dağıttığı görev maliyeti
local_target = local_credit + task_credit
chain_advantage = local_target − chain_value
throughput_credit = fulfill_reward × fulfilled − unmet_reward × unmet   (ISSUE)
throughput_advantage = throughput_credit − throughput_value
```

Yerel critic `local_target` öğrenir. Throughput ayrı critic ve
`--throughput-policy-coef` PPO terimidir; nakit hedefe eklenmez.
Terminal hedef ayrı kalır. `sum(local_credit) == net_cash` yalnız gerçek
parayı denetler. `task_credit` ve `throughput_credit` nakit kenarlarına yazılmaz.
Crew teslimat ödülü değişmez.

Kayıtlı kredi ve advantage altındır; `/3000` yoktur. Optimizer
(`gold-stream-rms-v1`) her PPO akışını kendi RMS'ine böler ve critic
MSE'sini hedef ölçeğine göre koşullar. BestAgent karşısında terminal
~−160000 olduğu için ham MSE zincir leftover sinyalini boğuyordu; bu
bölme krediyi değiştirmez, üç başlığın aynı adımda görünür kalmasını sağlar.

Bakınız: [ISSUE_COMPLETION.md](ISSUE_COMPLETION.md).

### Kısmi tamamlanma ve kaynak çekilmesi

`--task-cost-settlement unfulfilled_fraction` varsayılandır. Gerçek teslimat,
satış veya ara ürün kullanımına ulaşan **benzersiz** miktar kotayı karşılar.
Hedef 10, karşılanan 6 ise birikmiş görev cezasının %40'ı dağıtılır. Hedef tamamlandıysa
`EXPIRED` kaydı bulunsa bile kapanış görev cezası sıfırdır. `preserve` seçeneği ham
bekleme/bakım bütçesini korur; iki durumda da gerçek alış maliyeti ve kâr değişmez.

Hasat miktarı, depoya ulaşan miktar ile aynı değildir. Üretilip elde taşınan ürün,
henüz teslim/kullanım kotasını karşılamadıysa `held` olarak raporlanır; üretim
başarısı kaybolmaz. Hasada hazır tile verimi ayrıca raporlanır, gelir sayılmaz.
Bakınız: [ISSUE_COMPLETION.md](ISSUE_COMPLETION.md).

İhtiyaç varken gerçekleşen buğday satışı ve sonradan kapanmayan yem açığı için
doğrulanmış kanıt varsa, tek ceza bütçesi ISSUE ve SELL rolleri arasında eşit
bölünür. Kanıt yoksa sorumluluk ISSUE'da kalır; bir BUY/SELL kararı uydurulmaz.
Bu, açık bir kredi paylaşım kuralıdır; kesin nedensel katkı tahmini değildir.

`issue_task_cost.json` tüm neden/turn/issue/manager karar eşleşmelerini içerir.
`moves.jsonl.task_cost_events`, episode metrikleri ve `train.log` maliyeti gösterir.
`manager_decisions.json` ekonomik kredi, görev kredisi ve birleşik hedefi ayrı yazar.
`issue_completion.json` ürün miktarlarını, gerçekleşen kaynak kullanımını ve
başlangıç girdisinin mevcut/eksik olduğu zaman aralıklarını içerir.
`issue_production_graph.json.task_outcome_graph` kapanış DAG'ını da taşır.
Yeni semantik eski optimizer/checkpoint'e sessiz `--resume` edilmez. Çalışan süreçler
otomatik güncellenmez; yeni kod yeni koşuda devreye girer.

## Ekonomik ödül

`train_manager --crew scripted` ve `train_joint` aynı
[gerçek satış DAG'ını](ISSUE_DAG.md) kullanır. Aralarındaki fark crew optimizer'ının
çalışıp çalışmamasıdır, manager için satışın tanımı değildir.

- Satılan üretilmiş ürün: net altın, aksiyon yolundaki BUY / ISSUE / SELL atalarına.
- Satılmayan/kaybolan girdi: aynı yol, eksi net.
- Saf al-sat, HIRE ve arazi harcaması: kendi manager kararları, yine altın.
- Teslimat ve görev kapanışı manager'a hayali piyasa değeri kazandırmaz.
- Hiç yapılmamış alış/hiç alınmamış girdi katalog fiyatından cezalandırılmaz.
  Önceki katalog-ceza yardımcıları eski deneyler için dosyalarda kalır;
  yeni eğitim yolu bunları çağırmaz.
- Terminal `(bizim altın − rakibin altını)` ayrı hedef olarak korunur.

Ekonomik kredi toplamı `(son nakit − başlangıç nakdi)` ile mutabık olmak
zorundadır. Bu, terminal rakip farkıyla aynı defter değildir. Crew'un teslimat
kredisi manager'a eklenmez.

Sürekli üretici (domates, çilek, hayvan) ilk teslimat kotasında bırakılmaz;
üreticinin ömrü boyunca aynı görev ve üretici kimliği korunur. **Crew teslimat
ödülünün kotası değiştirilmedi:** kota üstündeki ürünler ekonomik DAG'da izlenir,
ama ayrıca sınırsız crew teslimat bonusu yaratmaz. Scripted crew bu sonraki işleri
de planlayabilir; öğrenen crew'un sonraki döngüler için motivasyonu ayrı tasarım konusudur.

## Loglar

- `metrics.jsonl`: küçük bölüm/güncelleme özeti; bütün görev sözlüğü tekrarlanmaz.
- `episodes/episode_*/moves.jsonl`: emirler, crew görev/rota atamaları, yalnız
  değişmiş görev kayıtları (`issue_updates`), açık görev sayısı ve para.
- `all_moves.jsonl`: bölüm/adım ve dosya yolu indeksi; ikinci büyük log kopyası değil.
- `trace.jsonl.gz`: her adımın observation/sonraki observation/aksiyonları; sıkıştırılmış.
- `crew_decisions.json`: birim, işlem, görev, atama değişimi ve teslimat kredisi.
- `manager_decisions.json`: gerçek yerel kredi, terminal advantage, maskeler ve talep girdisi.
- `issues.json`, `cash_ledger.json`, `issue_sale_credit.json`,
  `issue_production_graph.json`: bölüm sonunda tam denetim verisi.

Eski log okuyucularındaki `open_issues` listesi yerine `issue_updates` sözlüğünü
`task_id` üzerinden biriktirmek gerekir. Son durumun tamamı `issues.json` içindedir.

## Yeni uzun eğitim komutu

Bu komut yalnız manager'ı eğitir; crew algoritmik `tour` ataması ve `greedy`
yürüme kullanır. İki ağı da eğitmek için `train_joint` gerekir.

```bash
.venv/bin/python -m worker_grow.train_manager \
  --run-dir runs/feudal_manager_issue_outcomes_v4_01 \
  --episodes 10000 --episode-steps 720 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --learning-rate 3e-5 \
  --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 \
  --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 \
  --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 \
  --crew scripted --assignment-strategy tour --walk-policy greedy \
  --opponent starter --device cuda --seed 2026
```

Sıfırdan kısa doğrulamanın başarılı olması öğrenme/rekabet başarısı kanıtı değildir.

## Son doğrulama — 15 Eylül 2026

- `.venv/bin/python -m pytest tests -q`: **291 test, 203 alt test geçti**.
  Kapsam: aday kilitlenmesi, FIFO, uygun fiziksel kapasite, değişmez talep girdisi,
  PPO log-olasılık/critic tekrar hesabı, manager/crew kredi ayrımı, gerçek motor
  alım-satım akışı, sürekli bitki/hayvan görevleri ve checkpoint devamlılığı.
- Nihai kod ile CUDA kontrolü: `runs/manager_dag_v3_validation_20260915`.
  İki `episodeSteps=720` oyun, her birinde motorun 719 geçişi, ardından bir PPO
  güncellemesi. Başlangıç hediyesiz; rakip starter, manager öğrenen, crew scripted.
- Toplam **23 PLANT, 39 HARVEST, 8 PLACE**. Her iki oyunda ekonomik kredi/kasa
  mutabakat hatası **0**. PPO'da **78 minibatch işlendi, 0 atlandı**;
  raporlanan ortalama KL yaklaşık `0.0140`.
- İki oyun da **0 altınla kaybedildi**. Bu yalnız yürütme ve kredi bağlantısı
  kontrolüdür; kârlı politika öğrenildiği iddia edilmez. Uzun eğitim başlatılmadı.
- `manager_dag_v3_smoke_20260915_01` ve `_02` geliştirme sırasındaki ara kontrol
  kayıtlarıdır; nihai doğrulama veya yeni uzun koşu için bunlardan resume yapmayın.

Tüm testlerde ayrıca eski mastery çağrısının eksik `assignment_strategy` /
`walk_policy` alanları görüldü. `train_crew.collect` bu çağrılarda eskisi gibi
öğrenen atama/yürüme varsayılanlarını kullanır; mastery ödül kuralı değişmedi.

## Üretim ömrü, kare rezervasyonu ve ürün geçmişi — düzeltme

`task_lifecycle=production-occupancy-v2`:

- `tile_xy` tarihsel üretim konumudur; ürünün geçmişinden silinmez.
- `reserves_tile` yalnız açık ve üretimi bitmemiş görevin kareyi tuttuğunu söyler.
  Son hasat kareyi serbest bırakır. Aynı tur ikinci işçi de oraya yeniden ekebilir.
- Ürünün teslimatını bekleyen eski görev yeni bitkinin görevini gölgelemez.
  Görev CNN kanalları, aday seçimi ve kapasite hesabı aynı ayrımı kullanır.
- Gerçek FEED/FERTILIZE olayında, depoya daha önce teslim edilmemiş kaynak
  partilerinin tüketimi görev kotasını karşılayabilir (`consumed`). Bu sayaç
  görev kotasına kadar olan karşılanmış miktardır; bütün tüketim miktarları
  lifecycle/DAG kayıtlarındadır. Daha önce teslim edilen birim tekrar sayılmaz.
- Canlı domates/çilek/hayvan, kota karşılansa da üretici ömrü boyunca rezervasyonunu
  korur. Üretimi biten ve kotası teslimat/tüketimle karşılanan görev COMPLETE olur.
- Tüketim yeni teslimat ya da para ödülü yaratmaz. Mevcut depo teslimatı ödülünün
  kotası korunur. Eski partinin sonraki satışı/yeme dönüşmesi aynı DAG kimliğiyle
  takip edilir; yeni ekimin üretici kimliği ayrıdır.

CNN gerçek kareyi boş görebilir; fakat önceki görev defteri PLANT'i maskeliyorsa
bu aksiyonun seçilme olasılığı sıfırdır. Bu bir görüntü anlama eksikliği değildir.
Üstelik manager-only `--crew scripted` koşusunda saha seçimini algoritmik planner
yapar. Düzeltme bu yüzden CNN ağırlığında değil, rezervasyon/aday kurallarındadır.

Doğrulama: **302 test, 203 alt test geçti**. Önceki koşunun ilk 40 oyununa ait
gerçek olayların yeniden işlenmesinde 18 artık rezervasyon kaldırıldı; 16 yemlik
buğday görevi EXPIRED yerine COMPLETE oldu. Manager'ın ISSUE/karar kredileri ve
crew'un toplam ödülü önceki kayıtlarla aynı kaldı. En büyük mutabakat farkı
`4.55e-13`. Bu olay tekrarıdır; yeni politikayla 40 maç oynanmış değildir.

Aktif Python süreçleri dosya değişikliğini otomatik yüklemez. Yeni manager/joint/
crew koşuları yaşam döngüsü sözleşmesini config'e yazar; eski sözleşmeyle doğrudan
resume sessizce kabul edilmez. Eski eğitim kayıtlarına dokunulmaz.
