# Nihai net kârın aksiyon DAG'ından dağıtılması

Ortak eğitim: `worker_grow.train_joint`; manager-only eğitim: `worker_grow.train_manager`.
İkisi de aynı DAG'ı kullanır. Checkpoint: `feudal-joint-issue-gold-dag-v9` /
`feudal-manager-issue-gold-dag-v7`; kredi raporu: `issue-sale-dag-v2`.
Görev gecikme/bakım maliyeti ayrıca `task_credit` olarak aynı ISSUE'ya bağlıdır;
kota tamamlama/eksik `throughput_credit` ayrı PPO terimidir. İkisi de DAG'ın
gerçek gold hesabına dahil değildir. PPO nakit hedefi `local_credit + task_credit`.
Kapanışta kısmi tamamlanma ve kanıtlı kaynak satışı atfı:
[ISSUE_COMPLETION.md](ISSUE_COMPLETION.md). Yeni task sözleşmesi
`manager-completion-dag-cost-v2` eski task optimizer semantiğine resume edilmez.
Talep girdisi ve yeni komut: [MANAGER_TRAINING.md](MANAGER_TRAINING.md).
Ekonomik kredi altın cinsinden aksiyon yolunu izler. [Crew tile seçimi](CREW_SITE.md) ile issue önce
`tile_xy=null` açılır; gerçek ekim/yapı kurma olayı konumu bağlar. Producer
kimliği ve satış DAG'ı yine aynı issue'ya gider. Eski checkpoint resume edilmez.

## Kural

Her sink'te bütçe **gerçek gelir − taşınmış gerçek girdi maliyeti**dır (altın).
Satış yoksa gelir 0'dır. Bu net, lot'un aksiyon DAG'ındaki manager atalarına
gider. ISSUE, üretici bağlandığında (PLANT/PLACE) yola eklenir. Crew düğümü
ödenmez.

Maliyet malzeme DAG'ında miktarla yürür. Bir tohumdan 2 birim çıktıysa birim
başına maliyet yarıdır. Hiç hasat edilmemiş çıktı değerlenmez. Yeme giden
buğdaya piyasa fiyatından yeni maliyet eklenmez.

## Hangi bağlantı gerçekten var?

- `PLANTED`: gerçek tüketilen `seed_lot_id` ve yeni `plant_id`.
- `ANIMAL_PLACED`: tüketilen hayvan stok kimliği ve yeni `animal_id`.
- `HARVESTED` / `FERTILIZER_COLLECTED`: üreticiye bağlı yeni ürün partisi.
- `FEED` / `FERTILIZE`: gerçek `consumed_allocations` FIFO parçaları.
- PICKUP/PLACE/DROP/gece aktarımı: aynı parçanın yeni lot kimliğine geçişi.
- `SELL`: motorun gerçekten yürüttüğü miktar ve fiyatlara bağlı satış geliri.

Üretici yalnız tile koordinatıyla eşleştirilmez: kalıcı `plant_id`/`animal_id`
issue'ya bağlanır. Aynı tile'a daha sonra gelen başka üretici başka kimliktir.
Her hasat kendi değişmez geçmişini taşır. Sonraki yemleme, eski hasadın
issue ağacına sonradan ebeveyn ekleyemez.

Bir partinin ikiye bölünmesi iki yeni üretim dalı yaratmaz. Buna karşılık aynı
issue'nun **iki farklı hasat partisi** iki ayrı ara üründür; ikisi de tüketilirse
ikisi de dal olur, issue yaprağındaki payları en sonda birleşir.

Hayvan → gübre → buğday → aynı hayvan döngüsü, üretici isimlerine göre çizilmez:
zaman sıralı partilerle çizilir. Bu yüzden son hasat önceki hasada bağlanabilir
ama gelecekteki partiye bağlanamaz; yapı DAG olarak kalır.

## Maliyet ile ödül payı

Maliyet miktarla lot üzerinde yürür. Net kâr, o lot'un aksiyon yolundaki
manager kararlarına paylaştırılır. Pahalı girdi tek başına ISSUE payını
büyütmez; net zaten o maliyeti düşmüştür. `--credit-decay` crew teslimat
grafiğinedir; manager ekonomik yolda çarpan 1'dir.

## Hangi karar ödül alır?

Kredi **altındır**; `/3000` yoktur. Her sink'te `net = gelir − taşınmış maliyet`
(satışsız sink'te gelir 0). Bu net, lot'un aksiyon yolundaki manager
atalarına paylaştırılır: BUY, o üreticiyi bağlayan ISSUE, SELL ve yoldaki
diğer manager kararları. Crew düğümleri pay almaz; onların kütlesi aynı
yoldaki manager atalarına kalır.

- Ekilen/yerleştirilen üreticinin ISSUE'si plant/place anında yola girer.
- Satılmayan, kaybolan veya hiç kullanılmayan girdinin eksi net'i de aynı
  yolu izler; yalnız alışa kısaltılmaz.
- HIRE/BUY_LAND kendi kararında durur.
- Üretilmeden alınıp satılan stokta ISSUE yoktur; net BUY ve SELL atalarına
  gider.
- Kaynağı bilinmeyen satışta veya üretilmiş sink ISSUE'sini kaybederse eğitim
  durur.
- Terminal `(bizim altın − rakip altını)` ayrı hedeftir, yine altın.
  `sum(local_credit) == net_cash`.

Kontroller: her alışın maliyeti tam bir kez dağılmış olmalı; satış miktarı ve
geliri motor kayıtlarıyla eşleşmeli; manager yerel kredi toplamı kendi net nakit
değişimine eşit olmalı. Hata toleransı `1e-7` altındadır.

## Görev yaşamı ve crew sınırı

Yeni ortak eğitimde canlı üretici ilk teslimatta kapatılmaz. Domates, çilek ve
hayvan sonraki hasatlarda aynı issue'da kalır. Üreticinin gerçekten bitmesi,
ölmesi veya sezonun sonu fiziksel görev durumunu kapatır; önceden çıkan ürünün
ekonomik geçmişi bundan sonra da satışa/sonraki üretime kadar izlenir.

**Crew ödülü ayrıca değiştirilmedi:** ilk teslimat başına mevcut kota içinde
birim ödül ve kendi geriye azalan aksiyon kredisi duruyor. Kota sonrasındaki
üretim ekonomik DAG'da izlenir ama yeni crew teslimat ödülü üretmez. Bu, uzun
üretim görevlerinin öğrenilmesinde ayrıca değerlendirilmesi gereken bir sınırdır.
`total_delivered` bütün ilk fiziksel teslimatları; `delivered` ödül kotasına
sayılan miktarı tutar. Günler arası yürüyüş kredisinin önceki sınırlaması da
bu manager muhasebesi değişikliğinin kapsamı dışındadır.

## Dosyalar ve denetim

- `issue_production_graph.json`: miktar/maliyet kenarları, üretici→issue ve
  üretici→bütün çıktı partileri, her lot'un issue kökü, issue yaprakları.
- `issue_sale_credit.json`: satış bazında gerçek gelir, taşınmış maliyet, net
  kâr, `issue_shares`, son karar kredileri ve nakit mutabakatı.
- `issues.json`: `close_gold=0` (kapanış ödül yaratmaz), `realized_gold` nihai
  DAG'dan issue'ya ulaşan ekonomik kredi.
- `manager_decisions.json`: PPO'ya verilen `local_credit` ve `chain_advantage`.
- `delivery_credit.json`: ayrı crew hedefleri. Manager nakdi buraya eklenmez.

Ekonomik olay günlüğünde manager karar kimliği `2*id`, crew kimliği `2*id+1`.
Bu iki ajanın yerel kimliklerinin çakışmasını engeller. PPO raporundaki
`decision_credit` yeniden **yalın manager kimliğine** çevrilir; tek sayılı
ekonomik kredi hedefi yakalanırsa hata verilir.

Eski `feudal-joint-v1` checkpoint'i yeni ödülle `--resume` edilemez.
Yeni run dizini kullan; uzun koşular otomatik durdurulmaz/başlatılmaz.
