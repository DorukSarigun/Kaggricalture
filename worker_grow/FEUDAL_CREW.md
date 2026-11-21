# FeUdal görev koşullu crew — crew-site v2

Güncel crew ön eğitimi: [başarı müfredatı ve kısa görev kredisi](MASTERY_TRAINING.md).

Güncel ön eğitim ve checkpoint geçişi: [öğrenen alt görevler v4](SUBTASK_TRAINING.md).

Yeni görev sözleşmesi: [tile seçimi crew'de](CREW_SITE.md).

**İki ajanı aynı oyunlarda birlikte eğitmek için:**
[`train_joint` kurulumu ve CUDA komutu](JOINT_TRAINING.md).
Manager ve crew ayrı optimizer'larla aynı rollout'lardan güncellenir.
Ortak eğitimin yeni manager ödülü: [nihai satış → issue DAG'ı](ISSUE_DAG.md).
Bu belgedeki manager-only komutu legacy ödülü korur; yeni ortak eğitim değildir.

Mevcut production-DAG eğitimlerine dokunulmaz. Giriş:
`python -m worker_grow.train_crew`. Checkpoint: `feudal-contract-crew-site-v3`.

## Sorumluluklar

- Manager: ne üretilecek; kaynak, arazi, HIRE ve market. Tile seçmez.
- Crew: **farmer + bütün hands**, tek politika, tek ekip belleği. Manager
  tek tek yürüyüş/sulama emri değil kalıcı üretim görevleri verir. Crew yürüyüp
  ekim/yapı kurarak tile seçer; gerçek olay issue'yu o tile'a bağlar.
- Crew manager'ın nakit/terminal hedefini öğrenmez. Teslimat ödülü de manager
  kasasına eklenmez. Ayrı optimizer ve critic sözleşmesi korunur.

## FeUdal Networks ile ilişki

Kaynak: [FeUdal Networks, ICML 2017, denklem 3–4 ve bölüm 3.1](https://proceedings.mlr.press/v70/vezhnevets17a.html).

```text
g_t = normalize manager görevi
w_t = phi(sum(g_{t-c+1}, ..., g_t))     phi: bias içermeyen Linear
h_t = ortak crew GRU(güncel saha, önceki ekip belleği)
logits(op) = U_op(h_t) @ w_t
logits(item | op) = U_item(h_t, op) @ w_t
logits(q | op,item) = U_q(h_t, op,item,q) @ w_t
```

Hedef sınırında `detach`: crew loss manager hedefine geri yayılmaz; `phi`
crew tarafından öğrenilir. Son 10 **turun** hedefleri toplanır, her işçi için
tekrar eklenmez. Güncel görev maskesi eski hedefe ait geçersiz işi engeller.

**Tam FuN replikasyonu değildir.** Bu aşamada semantic kontratlar vardır;
makaledeki öğrenilen latent durum yönleri yoktur. Crew görev haritasını ayrıca
görür. Cosine intrinsic reward yerine anlaştığımız teslimat ödülü, optimizer
olarak PPO, bellek olarak GRU kullanılır. Dilated-LSTM manager ve transition
policy gradient henüz yazılmadı. FuN tarzı goal-modulated worker uygulanmıştır;
tam FeUdal manager eğitildiği iddia edilmez.

## Bir tur

```text
Manager görevleri
      ↓
farmer → güncel shadow → hand[0] → güncel shadow → hand[1] → ...
      ↓
Manager market(post-crew shadow)
      ↓
Gerçek environment.step → gerçek lifecycle → ilk teslimatlar
```

Bir çalışan bir turda tek işlem yapar; yürüme ve WATER aynı tur olmaz.
İlk el ekerse sonraki el dolu tile görür. HIRE aynı tur yeni el açmaz. Gece
eller silinir; ekip belleği kalır, episode değişince `CrewState` sıfırlanır.

`attach_manager_market(observation, crew_action, market_policy)` callback'e
unit işlemleri uygulanmış shadow verir; callback yalnız market listesi üretir.
Alınan tohum geçmişteki PLANT'e kaynak olmaz. Shadow ödül kaynağı değildir.

## Görev arayüzü

```python
from worker_grow import ProductionTask, TaskBook, FeudalCrew, propose_crew_turn, CrewDeliveryTracker

tasks = TaskBook([
    ProductionTask("wheat:42", (3, 5), "WHEAT", (("WHEAT", 4),)),
    ProductionTask("goose:7", (1, 2), "GOOSE", (("EGG", 4), ("FERTILIZER", 2))),
])
model = FeudalCrew()
tracker = CrewDeliveryTracker(observation, tasks)
state = None  # sadece episode başında
action, decisions, state = propose_crew_turn(model, observation, tasks, state)
```

Her gerçek environment geçişinden sonra `tracker.observe_transition(...)`
çağrılmalıdır; bu çağrı görev ilerlemesini ve teslimatları günceller. Tek başına
`propose_crew_turn` görevi tamamlanmış saymaz. Tam çağrı sırası `train_crew.py`
içindeki `collect` fonksiyonundadır. Tracker'a gerçek market execution kayıtları
verilir, fakat bunlar crew ekonomik ödülü oluşturmaz.

Quotalar talep edilen gerçek miktardır, manager'ın belirlediği sınırsız para
ödülü değildir. Aynı açık tile'a ikinci görev ve tekrar aynı task ID yasaktır.

Durumlar: `PLANNED → ACTIVE → DELIVERING → COMPLETE`; ayrıca `FAILED`,
`CANCELLED`, episode sonunda `EXPIRED`. `resource_state` kaynak/arazi beklemeyi
ayrıca bildirir. Kaynak yokluğuna otomatik crew cezası verilmez.

PLANT istenen crop için crew'ün seçtiği uygun tile'da, BUILD ve hayvan PLACE istenen yapı/tür için;
bakım/hasat görev tile'ı ve motor maskeleriyle sınırlıdır. DIG yalnız yeni
görevdeki otu temizler: canlı üretici/yapı sökümü ayrı manager görevi gerektirir.
PICKUP görevlerin girdileriyle filtrelenir. Kilitli kareye yürümek motoren
yasaldır; ekstra ceza yoktur. Açık iş ve taşınan yük yoksa PASS kalır.

Mevcut canlı üreticiyi yeni task ID'ye devretme ve iptal sonrası söküm henüz
ayrı görev tipleri değildir. Bir üreticinin görevi kotası boyunca kalıcıdır.

## Gözlem / ağ

Yalnız kendi farm CNN'i, bütün çalışan konumları, seçili çalışanın konumu/cebi,
takvim ve hedef haritaları kullanılır. Rakip farm yok; observation değişmez.
8 bağlı üretici türü + kalan kota + başladı + 8 konumsuz talep sayısı = 18 görev kanalı.
Varsayılan hedef adaptörü `18×10×10 + idle = 1801` boyutludur; `phi` bunu 16 boyuta indirir.
Ekip GRU'su 128 boyutludur. Tamamlanan komutun op/item/quantity bilgisi belleğe
yazılır. Yeni turn aynı ekip belleğinden devam eder. Quantity `1..limit`
aralığındaki bütün tamsayıları temsil eder.

## Ödül: ilk gerçek teslimat

```text
r = min(ilk teslim edilen yeni miktar, görevde kalan kota) × 1
```

Yalnız görev üreticisinden gerçekten hasat/toplama ile edinilip depoya ulaşan
ürün sayılır. Her talep edilmiş birimin tarifesi sabit 1'dir; canlı fiyat,
hayali satış veya tohum maliyetinden crew grow bonusu oluşturulmaz. Gübre
yalnız görevde açıkça istenmişse ödeme alır.

FIFO parçaları kök hasat batch'i ve `delivered` bayrağını korur. Lot ID'si
taşıma sırasında değişse de aynı birim tekrar ödüllendirilmez. Marketten
alınmış/başlangıçta depoda verilmiş ürünlerin taşınması 0 getirir. PICKUP →
DROP döngüsü ve SELL ek ödül yaratmaz. Gece otomatik depolama sayılır ama
hayali DROP kararı yaratılmaz. Depo taşmasıyla kaybolan ürün sayılmaz.

Teslimat bütçesi ilgili crew kararlarına geriye `decay=0.8` ile dağıtılır;
girdi rollerinde dallanır, ortak atalar birleşir. Payların toplamı teslimat
ödülüne eşittir. Manager BUY/SELL/HIRE kararları kredi almaz. Farklı işçilerin
emeği aynı grafikte birleşir; sonraki bakım önceki hasada sızmaz.

Taşıma ve ilk işe yaklaşma izlenir; henüz elde olmayan iki komşu hedefe ortak
yaklaşmayı bulan planner değildir. Bu, tasarlanmış kredi paylaşımıdır;
karşı-olgusal katkı veya tarafsız policy-gradient garantisi değildir.

Başarısız/eksik görevde teslim edilmemiş miktar 0 alır. Şimdilik ayrıca zaman,
PASS veya başarısızlık cezası yok. Eski fiyat/aksiyon `grow_step_reward` ve
yürüme potansiyeli yeni eğitimde **çağrılmaz**; eski modüller uyumluluk için durur.

## PPO

`A_decision = dağıtılan teslimat kredisi − crew_value(state, goal)`.
İkinci GAE/return-to-go toplamı yoktur. Bütün işçilere aynı turda aynı A
zorla yazılmaz; tek politikanın koşullu kararlarına ayrı yerel kredi atanır.

Maskeler ve eski log-prob saklanır. Recurrent eğitim ardışık 16 kararın
TBPTT parçalarıdır: başlangıçta rollout'un detached hidden state'i kullanılır.
Episode başından tam recurrent replay yapılmaz; parça sınırı belleği eski
politikadan kaldığı için yaklaşık eğitimdir. KL kontrolü, gradient clipping
ve entropy loglanır. Kayıtlı hedefler manager nakit değerleri içermez.

## Crew ön-eğitimi

```bash
.venv/bin/python -m worker_grow.train_crew \
  --run-dir runs/feudal_crew_mature_01 \
  --episodes 100 --episode-steps 96 --episodes-per-update 4 \
  --stage mature --item WHEAT --hands 2 --quantity 1 \
  --learning-rate 3e-5 --epochs 3 --sequence-length 16 \
  --credit-decay 0.8 --max-grad-norm 1.0 --target-kl 0.02 \
  --save-every 10 --device cuda
```

Manager yerine sabit görev sağlayıcı kullanılır. `mature`: hasada hazır ürün;
`growing`: yeni üretici; `seeded`: tohum/hayvan stoku. Başlangıç işçi/kaynakları
**hediyedir**. Bu, tam ekonomik oyun veya rekabet değerlendirmesi değildir.
Seyrek ödül sorunu sürer; dersler bunu kontrollü biçimde ele almak içindir.

Her episode: `delivery_credit.json`, `decisions.json`, `summary.json`, lifecycle
olay/lotları. `--save-every` sıklığında `trace.jsonl.gz` ve `replay.json.gz`.
Loglar ekonomik kâr değil crew teslimatı olarak etiketlenir.

Devam: aynı komuta `--resume`. Yalnız episodes/save sıklığı değiştirilebilir.
Yarım episode dosyaları veya farklı sürüm varsa sessizce devam edilmez. Ders
aşamasını değiştirerek ağırlık transferi henüz ayrı bir CLI özelliği değildir.

## Manager eğitimi

```bash
.venv/bin/python -m worker_grow.train_manager \
  --run-dir runs/feudal_manager_cuda_01 \
  --episodes 10000 --episode-steps 720 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --learning-rate 3e-5 \
  --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 \
  --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 \
  --max-grad-norm 1.0 --save-every 10 --crew pass --opponent starter \
  --device cuda --seed 2026
```

Varsayılanlar da budur: tam 720 adımlık maç, CUDA, 10000 episode. Dolu bir
klasöre yazılmaz. `--crew feudal` crew optimizer'ını güncellemez; isteğe bağlı
`--crew-checkpoint` donmuş ağırlık yükler. `moves.jsonl`, kararlar, issue ve
her PPO minibatch her zaman yazılır; `replay.json.gz` yalnızca `--save-every`.

**Var:** farmer+hands, hedef çarpımı/gradient sınırı, ekip belleği, sıralı shadow,
market adaptörü, görev maskesi, teslimat grafiği, PPO, CLI/replay/checkpoint,
öğrenen ekonomik manager; `train_joint` ile iki ajanın ortak oyunlarda eğitimi.

**Yok:** tam FuN manager/TPG, devir/söküm
kontratları, yürüme potansiyeli ve uzun başarı değerlendirmesi.

## Bu uygulamada doğrulananlar

- Tam paket: **186 test geçti**; bunun 17'si yeni crew testidir.
- Beş bitki ve üç hayvanın gerçek motor teslimatları; alıştan başlayan
  ekim/bakım ve yapı/yerleştirme/yem akışları; kısmi teslimat ve tekrar depolama;
  kota, iptal, depo taşması, gece kaynağın korunması test edildi.
- Goal stop-gradient, bütün ekip sırası, shadow, log-prob tekrar hesabı,
  recurrent gradyan, tam miktar desteği ve PPO güncellemesi test edildi.
- CPU ve NVIDIA GB10 CUDA'da ayrı ayrı iki kısa episode + PPO, ardından
  checkpoint'ten üçüncü episode + PPO tamamlandı. Her koşuda ilk güncelleme
  12, devam güncellemesi 6 chunk kullandı; atlanan chunk 0.
- Geçici kayıtlar: `/tmp/feudal-crew-check-RoeaPf` ve
  `/tmp/feudal-crew-cuda-check-0KNNuS`. Bunlar 48 kayıt uzunluğundaki küçük
  başlangıç-hediyeli derslerdir; ekonomik ajan başarısı kanıtı değildir.
- Uzun training başlatılmadı; mevcut manager/production-DAG koşuları değiştirilmedi.
