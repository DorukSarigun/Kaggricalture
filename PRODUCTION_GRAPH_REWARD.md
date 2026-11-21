# Üretim grafiğiyle ödül — v2

Giriş: `train_realized_chains.py` · Muhasebe: `production_graph_credit.py`.
Checkpoint biçimi: `structured-production-dag-v2`.

## Değişen şey

Kredi artık kronolojik bir aksiyon listesini düzleştirerek dağıtılmaz.
Gerçekleşmiş işlemlerden **zaman sıralı bir DAG** kurulur. Bir kaynak birden
fazla ürüne ayrılabilir; bir ürün birden fazla girdiyi birleştirebilir.
Paylaşılan geçmiş kopyalanarak yeni kâr yaratılmaz.

```text
BUY_SEED → PLANT → WATER → HARVEST (2 buğday)
                              ├─ 1 buğday → DROP → SELL
                              └─ 1 buğday → FEED ─────────────┐
BUY_ANIMAL → PICKUP → PLACE → CARE ──────────────────────────┤
BUILD_COOP ────────────┘                                     │
                              hayvan → HARVEST → DROP → SELL
                                    └→ COLLECT_FERTILIZER
                                          → FERTILIZE → bitki → hasat → satış
```

Hayvan → gübre → buğday → aynı hayvan bağlantısı nesne kimlikleri üzerinde
döngü gibi görünür. Grafikte ise farklı tarihlerdeki tüketim/hasat düğümleri
vardır; hiçbir kenar zamanda geriye gitmez. Ödülün geriye yayılması, malzeme
akışının geriye gitmesi demek değildir.

## Birbirine karışmayan iki grafik

| Grafik | Yön | Taşıdığı bilgi |
| --- | --- | --- |
| Malzeme/maliyet | Alıştan sonuca | Lot, gerçek miktar, maliyet kaynağı |
| Aksiyon kredisi | Sonuçtan önceki işlere | Karar kimliği ve azalan paylaşım ağırlığı |

`seed_lot_id`, `animal_lot_id`, `plant_id`, `animal_id`, `product_lot_id`
tracker'ın sentetik kimlikleridir. Motor aynı türdeki ürünleri sayılarda
birleştirir; hangi fiziksel tanenin tüketildiği gözlenmez. **FIFO burada açık
bir muhasebe kuralıdır**, motorun verdiği gizli bir tohum kimliği değildir.

Her aksiyonun `decision_id` değeri vardır. Aynı DROP veya SELL birden çok
ürünü etkileyebilir. Bu durumda ürünlere özel düğümler aynı karar kimliğini
paylaşır. Bir ürünün satışına başka ürünün alış geçmişi yanlışlıkla eklenmez;
kararın farklı ürünlerden aldığı kredi en sonda toplanır.

## Maliyet nasıl izlenir?

1. Gerçek market execution kayıtlarından **fiilen alınan miktar ve ödenen
   fiyat** okunur. Ekrandaki tahmini fiyat kullanılmaz.
2. Bir lotun `q / toplam_miktar` kadarı kullanılmışsa aynı oranda maliyeti
   kullanılır. Aynı miktar ikinci defa tüketilemez.
3. Ekim bir tohum tüketir; PLACE bir hayvan tüketir. FEED/FERTILIZE gerçek
   yem/gübre lotlarını tüketir. PICKUP/DROP ve gece depolama kimliği devreder.
4. Bir üreticinin girdisinin maliyeti, **o girdiden sonra gerçekten
   hasat/toplama ile edinilmiş çıktıların miktarına** bölünür. Sonraki yem
   veya gübre daha önce hasat edilmiş partiye maliyet yazamaz.
5. Hayvanın yumurta/süt/yün ve gübre gibi farklı çıktılarına ortak maliyet,
   edinilen birim sayısına göre bölünür. Örneğin dört yumurta ve bir gübre
   için ortak maliyetin 4/5'i yumurtalara, 1/5'i gübreye gider. Bu açık bir
   maliyet paylaştırma tercihidir; fiyat/potansiyel veya nedensellik iddiası
   değildir. Ayar adı `subsequent_harvested_units`.
6. Episode sonunda satılmayan stok ve çıktısız girdilerin kalan maliyeti
   zarar olarak kapanır. Girdi hayvana/bitkiye aktarılmışsa orijinal lota
   ayrıca aynı zarar yazılmaz.

Tek SELL farklı kaynak partilerini satabilir. Gerçekleşmiş gelir FIFO
partilerine ayrılır; her parti kendi maliyetiyle karşılaştırılır. Örneğin
10'a ve 100'e alınmış iki birim 30'ar altına satıldığında zincir kârları
**+20 ve −70**'tir. İki alışa ortalama −25 yazılmaz. Ortak SELL kararı iki
zincirden de pay alır.

Örnek: 10 altınlık tohumdan iki buğday elde edilirse, 26'ya satılan birinin
maliyeti 5, kârı 21'dir. Diğeri 300 altınlık kazın yemi olursa ve kazın tek
edinilen çıktısı 200'e satılan yumurtalarsa, o zincirin maliyeti 305 ve kârı
−105 olur. Toplam kredi: **21 − 105 = −84 = 226 satış − 310 alış**.

Bu hesap episode sonunda yapılır. Sonradan edinilen ilave çıktıların ortak
maliyetten pay alması önceki çıktının nihai maliyet payını değiştirebilir;
ancak önceki çıktının aksiyon geçmişine gelecek aksiyon eklenmez.

## Dallara bölünerek geriye azalma

Her satılan parti için:

```text
P = gerçek satış geliri − partinin maliyet payı
son satış düğümünün başlangıç ağırlığı = 1
her gerçek aksiyondan önceki aksiyonlara geçerken: ağırlık × decay
dallanırken: ağırlık × dalın payı
birleşen ortak atada: gelen ağırlıklar toplanır
kararın altın kredisi = P × karar_ağırlığı / bütün_karar_ağırlıkları_toplamı
```

Varsayılan `decay=0.8`. Azalma **bağlı aksiyon kenarı** başınadır, geçen oyun
turu veya tüm episode'daki karar sıra farkı başına değildir. Muhasebe amaçlı
lot bölme/yeniden adlandırma düğümü ek iskonto oluşturmaz.

Farklı girdi rolleri (önceki üretici durumu, tüketilen malzeme, yaklaşma yolu,
hazırlık) eşit paylaşır. Tek malzeme rolü içinde kaynaklar miktar oranında
paylaşır: 1 ve 3 birimlik iki kaynak için 1/4 ve 3/4. Bu kural
`equal_roles_quantity_within_role` olarak kaydedilir.

```text
        A
       / \
      B   C
       \ /
        SELL
```

Eşit dallarda ham ağırlıklar SELL=1, B=0.4, C=0.4, A=0.64 olur.
244 altın kâr için paylar 100, 40, 40, 64'tür. Ortak A iki kere 244 almaz.
Her sonuç için payların toplamı tam olarak P'dir; negatif P aynı ağırlıklarla
negatif dağıtılır. Uzun geçmişler taşma olmaması için log uzayında hesaplanır.

Bu **tasarlanmış kredi paylaşımıdır**. Gerçek karşı-olgusal katkıyı ölçmez,
tarafsız policy-gradient veya yalnız terminal hedefini koruma garantisi vermez.

## Satışsız kalan maliyet

Üç açık seçenek vardır:

- `--unsold-weighting tree` (**varsayılan**): hiç üretime girmemiş tohum/ürün/hayvan
  stoğunun maliyeti BUY'da kalır. Ekilmiş bitki, yerleştirilmiş hayvan,
  hasat edilip satılmayan ürün veya kaçış/ot kaybı aynı bakım yoluna
  (PLANT/PLACE, WATER/FEED/CARE, HARVEST, yürüyüş, DROP) oldest-heavy yazılır.
  HIRE/BUY_LAND kendi nakit kararında kalır. PASS ve gece otomatik DROP
  uydurulmaz.
- `--unsold-weighting purchase_only`: satılmamış/kaybolmuş her maliyet yalnız
  kaynak BUY'a gider. Bakım yolu sessiz kalır.
- `--unsold-weighting oldest`: her leftover, üretime girsin girmesin, aksiyon
  grafiğine oldest-heavy yayılır.

Hiç kullanılmamış bir tohumda `tree` ve `purchase_only` aynıdır. Başarılı bir
kardeş tohum, hiç kullanılmamış tohumun maliyetine ortak olmaz. Bu seçim
checkpoint sözleşmesinin parçasıdır.

## Terminal ve PPO bağlantısı

- Ara turlara nakit değişimi veya potansiyel ödülü eklenmez.
- Terminal hedef: `(bizim son gold − rakibin son gold) / 3000`.
- Yerel hedef: grafiğin karara atadığı altın / 3000.
- Yerel advantage: `yerel_kredi − yerel_value`. Yerel kredi üzerinde ikinci
  bir GAE, return-to-go veya tekrar discount yoktur.
- Terminal value ve yerel value ayrı öğrenilir; actor iki ayrı PPO terimini
  kullanır. İkisinin katsayıları ayrı CLI seçenekleridir.
- Terminal için gamma=1, varsayılan lambda=1. Bu varsayılanda tamamlanmış
  episode'un MC terminal getirisi kullanılır.

Nakit defteri her tur bakiyeyi doğrular; **bu kontrol tur ödülü üretmez**.
İki PPO hedefinin ikisinde de ekonomi bulunur: bunlar tek tarafsız ödülün
iki parçaya ayrılması değildir, terminal + yardımcı yerel hedeftir.

## Kapsam ve dürüst sınırlar

- Beş bitki, tekrar hasatlar; kaz/inek/koyun; yumurta/süt/yün/gübre;
  satın alınmış yem ve gübre; kısmi satış/tüketim; gece depolama izlenir.
- Ürün partisi hasat/toplama anında oluşturulur. Tile'daki her meyvenin veya
  yumurtanın biyolojik oluşum anını ayrı kimlikle izlemiyoruz. Hasada kadar
  gözlenen bakım geçmişi kullanılır; daha önce elde edilmiş partiye sonraki
  bakım bağlanmaz. "Tam biyolojik nedensellik" iddiası yoktur.
- Taşınan ürün/hayvanın gerçek yürüyüşü kaynak geçmişine bağlanır. Boş elle
  yaklaşma, aynı gün aynı işçinin sonraki doğrulanmış işine bağlanır.
  Aynı yürüyüşün eldeki birden çok ürüne katkısı korunur. Henüz elde olmayan
  iki komşu hedefe ortak yaklaşmayı ayrıca planlayan bir sistem değildir.
- Zamanında alınmamış, yerde üretilip çürümüş ürün için hayali gelir yoktur.
- HIRE/BUY_LAND gerçek gideri kendi alış kararında kalır. İşçi/arazinin
  bütün gelecekteki üretimlere katkısını bu sürümde uydurarak dağıtmıyoruz.
- PASS/STOP ve başarısız işler üretim grafiğine sahte bağlantıyla eklenmez;
  terminal hedefinden öğrenebilirler.
- Kaynağı gerçekten bilinmeyen/başlangıçta verilmiş stok açıkça maliyeti
  bilinmeyen kaynak olarak kaydedilir; `unknown_source_sale_revenue` izlenir.
  Bilinen lotun kaybolan kimliği ise sessizce sıfır maliyete çevrilmez:
  hata verir. Geçmişi olmayan satışa sahte PLANT/WATER atası eklenmez.

## Denetim ve kayıtlar

Her episode için:

| Dosya | İçerik |
| --- | --- |
| `production_graph.json` | Aksiyon/malzeme düğümleri, kenarlar, alış başına maliyet mutabakatı |
| `chain_credit.json` | Her sonuçta gelir, maliyet kaynakları, net kâr ve karar payları |
| `advantages.json` | Karar → aksiyon/unit/emir → terminal ve yerel advantage |
| `cash_ledger.json` | Gerçek market nakdi ve bakiye mutabakatı |
| `lifecycle_events.jsonl.gz` | Olay günlüğü ve sentetik kimlikler |
| `trace.jsonl.gz`, `replay.json.gz` | `--save-every` sıklığıyla observation/aksiyon ve izlenebilir oyun |

Maliyet kaynağı başına dağıtılan toplam alış bedeline eşit olmalıdır.
Malzemenin tüketilen/kalan oranları 1 etmelidir. Karar bazında alış/satış
günlüğü gerçek cash execution ile tutmalıdır. Tüm yerel altın kredilerinin
toplamı kendi net nakit değişimi olmalıdır (`1e-7` tolerans). Aksi halde
güncelleme yapılmaz; eğitim hata verir.

## Kullanım

Önce bağımsız bir doğrulama koşusu:

```bash
.venv/bin/python train_realized_chains.py \
  --run-dir runs/production_dag_v2_check \
  --episodes 4 --episode-steps 720 --episodes-per-update 2 \
  --epochs 1 --minibatch-size 64 --learning-rate 3e-5 \
  --chain-decay 0.8 --unsold-weighting tree \
  --save-every 1 --device cuda
```

CPU için `--device cpu`. Dolu bir klasöre yeni koşu yazılmaz. Devam etmek
için aynı yapılandırmayla `--resume`; yalnız toplam episode/save sıklığı
değiştirilebilir. Eski v1 checkpoint yeni v2 trainer'a resume edilemez.
Mevcut çalışan süreçlerin ödül semantiği bu dosya değişimiyle değiştirilmez.

Testler: `tests/test_production_graph.py` gerçek oyun motorunda beş bitki ve
üç hayvanın akışlarını; ayrıca dallanma/birleşme, kısmi satış/yem, hayvandan
gübreye maliyet geçişi, geri beslenen üretim, gelecekteki girdilerin önceki
hasada sızmaması, alış bazında muhasebe ve hatalı provenance durumlarını
kontrol eder. Bu testler ve kısa PPO koşuları davranışsal başarı kanıtı değil,
uygulama/muhasebe doğrulamasıdır.

### Bu değişiklikte çalıştırılan doğrulama

- Tam test paketi: **160 test geçti**.
- Son v2 kodu: CPU'da üç tamamlanmış episode, `episodeSteps=720`
  (motorun başlangıç durumuyla birlikte 720 kayıt, episode başına 719 geçiş).
- İlk iki episode birlikte güncellendi; checkpoint'ten resume ile üçüncü
  episode ve ikinci PPO güncellemesi tamamlandı.
- Kabul edilen minibatch: ilk güncelleme 77, ikinci güncelleme 38;
  atlanan minibatch: ikisinde de 0. Gradient clipping aktif.
- Üç episode'da kasa/kredi mutabakat farkı 0; kaynağı belirsiz satış geliri 0.
- Son doğrulama kayıtları: `/tmp/kaggriculture-graph-v2-final-ShRLX2`.
  Bunlar geçici test kayıtlarıdır; uzun eğitim başlatılmadı.
- Bu kısa koşularda ajan başarılı bir politika öğrenmiş sayılmaz; politika
  kalitesine yönelik bir başarı iddiası yoktur. CUDA çalıştırması yapılmadı.
