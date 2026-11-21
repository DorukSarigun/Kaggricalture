# Gerçekleşmiş zincir ödülü — v1

> **Arşiv: v1 davranışının açıklamasıdır.** Güncel `train_realized_chains.py`
> artık `structured-production-dag-v2` kullanır. Yeni dallı grafik, miktar
> bazlı maliyet ve eğitim sözleşmesi: [PRODUCTION_GRAPH_REWARD.md](PRODUCTION_GRAPH_REWARD.md).
> Aşağıdaki düz zincir dağıtımı yalnız eski `realized_credit.py` içindir.

Giriş noktası: `train_realized_chains.py`. Checkpoint biçimi:
`structured-realized-chain-v1`. Eski `train_structured.py` ve çalışan koşular
korunmuştur; bu dosyayı değiştirmek o süreçlerin ödülünü değiştirmez.
Eski checkpoint bu trainer'da resume edilemez.

## İki ayrı hedef

1. **Terminal:** yalnız son geçişte `(bizim gold − rakip gold) / 3000`.
   Ara turlarda nakit veya potansiyel ödülü yok. Varsayılan gamma=1, terminal
   GAE lambda=1: tamamlanmış episode'un MC getirisi eksi terminal critic.
   `--gae-lambda` yalnız bu hedefi değiştirir.
2. **Yerel zincir kredisi:** gerçekleşmiş satış geliri eksi ilgili girdilerin
   gerçek maliyeti. Episode bittiğinde ilgili karar indekslerine yazılır.
   Bu hedefe GAE, return-to-go toplamı veya ikinci bir discount uygulanmaz.

Hiçbir potansiyel fonksiyonu çağrılmaz; `--shaping-beta` seçeneği yoktur.
Muhasebe için her tur para kontrol edilir, fakat bu kontrol ara-tur ödülü
üretmez. Terminal hedef ile yerel hedef, aynı para biriminde iki ayrı PPO
loss'udur; toplamlarına "oyunun tek gerçek ödülü" denmez. Aynı ekonomik
sonuç iki hedefte de etkili olabilir: bu bilinçli bir yardımcı hedef
tasarımıdır, tarafsız causal policy-gradient veya hedef-koruma garantisi değil.

## Azalan dağıtım

Zincirdeki benzersiz kararlar kronolojik sıralanır. Satış olmuş üründe kâr
**en yeni emir en ağır** (`heaviest=newest`):

```text
P = gerçekleşmiş gelir − ilgili maliyet
w_i = d ** (N − 1 − i)
credit_i_gold = P * w_i / sum(w)
```

Ekilmiş ama hiç satılmamış bitkide zarar **en eski emir en ağır**
(`heaviest=oldest`): `w_i = d ** i`, `P = −(tek tohum alış fiyatı)`. Zincir
alış, tile bakımı, hasat, taşıma ve sökümü kapsar.

Varsayılan `--chain-decay 0.8`, ayarlanabilir `(0,1]`. **Azalma birimi bağlı
aksiyon sırasıdır**, fiziksel tur değildir. Payların toplamı P'dir.
PPO girdisinde altın cinsindeki paylar 3000'e bölünür.

## Kimlik ve maliyet kuralları

- `seed_lot_id` satın alma partisini, `plant_id` tek ekimi,
  `product_lot_id` hasat/taşıma ürün partisini tanımlar. FIFO sentetik
  muhasebe eşleştirmesidir; motorun gizli bireysel tohum kimliği değildir.
- Lifecycle event'leri artık `decision_id` ile tam structured emre bağlanır.
  Unit için `(step, unit, unit_index)`, market için
  `(step, market, order_index)` kullanılır. Token span'lerine bağımlı değildir.
- Tek bitkinin tekrar hasatları ve parçalı satışları episode sonunda aynı
  zincirde toplanır. Bir tohum maliyeti yalnız bir kez düşülür. Bitkinin bir
  kısmı satılmışsa o bitkinin tohum maliyetinin tamamı zincire aittir; her meyve
  için yeniden tohum maliyeti çıkartılmaz.
- Başarılı gübrelemede tüketilen satın alınmış gübrenin ürün-lot kaynağı da
  korunur. Satılan hasada katkıda bulunmuş girdinin maliyeti zincire bir kez
  eklenir; sonradan, o hasattan sonra kullanılan gübre geçmiş satışa yazılmaz.
- Doğrudan ürün alıp satmada satılmış birimlerin gerçekleşmiş alış maliyetleri
  kullanılır. Ekilememiş tohum veya kullanılmamış gübre orijinal alışa yazılır.
- Ekilmiş ama hiç satılmamış bitki: o tek tohumun alış fiyatı alış→tile→hasat
  zincirine ilk emir en ağır olacak şekilde dağıtılır. Satış olmuş bitkiye
  ikinci tohum zararı yazılmaz. Kardeş tohum başarılı bitkiye yüklenmez.
- Her hasat o andaki bakım/girdi geçmişini kopyalar: sonraki sulama/hasat,
  daha önce satılmış üründen kredi alamaz. Satılmayan ayrı hasadın HARVEST
  aksiyonu başka hasadın satışına eklenmez.

## Hareketlerin kapsamı

Başarılı yürüme event'i başlangıç/bitiş koordinatı, unit ve taşınan ürün
lotlarını kaydeder. Elde taşınan ürünün yolu o ürünün satılan torunlarına
taşınır; aynı yolda taşınan birden fazla ürün kendi kârından pay verebilir.
Boş elle yaklaşma segmenti aynı gün aynı unit'in sonraki doğrulanmış
ekim/bakım/hasat/ürün-transfer işine bağlanır. Her iş segmenti kapatır;
gün değişiminde segmentler karışmaz. Başarılı ot temizleme aynı tile'ın
sonraki ekimine bağlanabilir; yaşayan bitkiyi söken DIG olumlu üretim işi sayılmaz.

Bu, yolun bütün olası komşu hedeflere nedensel katkısını çözen bir planner
değildir. Henüz elde olmayan ikinci ürün için önceki yaklaşma segmentini
otomatik olarak tekrar paylaşmaz. İlgisiz PASS/STOP, başarısız/no-op işler
zincire uydurularak eklenmez. Gece otomatik depolama kaynağı korur ama bunun
için hayali bir DROP kararı oluşturulmaz.

## Diğer giderler ve açık kaynaklar

HIRE ve BUY_LAND maliyeti kendi gerçekleşmiş satın alma kararında negatif kalır.
Hayvan alış→yapı→PLACE→FEED/CARE→hasat/gübre→satış bitkiyle aynı kuraldadır:
satılan EGG/MILK/WOOL/gübre newest-heavy net kâr; satılmamış veya kaçmış hayvan
birim fiyatı oldest-heavy. Yerleştirilmemiş hayvan alışta kalır. Kaynağı bilinmeyen ürün
satışı açıkça `unlinked_sale` olarak yalnız SELL'e verilir; olmayan bitki
zinciri uydurulmaz. `unknown_source_sale_revenue` mutlaka izlenmelidir.

İki kontrol bağımsızdır:

```text
cash ledger: gerçek bakiye değişimi == execution nakit toplamı
yerel kredi: tüm payların toplamı == kendi satışları − kendi tüm alış/giderleri
terminal:   reward toplamı == son gold farkı / 3000
```

Mutabakat hatasında koşu durur. Mutabakat doğru olsa da nedensel atamanın
tam veya politikanın başarılı olduğu sonucu çıkmaz.

## PPO bağlantısı

Eski ağ mimarisi korunur, head semantiği yeni checkpoint sürümünde değişir:

- `value_head`: terminal getiriyi tahmin eder.
- `realized_value_head`: bu karara atanacak **yerel zincir payını** tahmin eder.
  Artık sadece yardımcı bir nakit tahmincisi değildir.
- `A_terminal`: terminal ödülün GAE/MC avantajı.
- `A_chain = chain_credit − old_local_value`: doğrudan yerel avantaj.
- İki ayrı clipped surrogate, varsayılan katsayıları 1 ve 1 ile toplanır.
  İkisi de gold/3000 ölçeğinde kalır; bağımsız standartlaştırma/ortalama çıkarma
  ile hedeflerin ağırlıkları sessizce değiştirilmez.
- Sıfır kredi alan stochastic kararların yerel baseline hedefi sıfırdır.
  Reward ile advantage aynı değildir; baseline nedeniyle sıfır kredi de
  negatif/pozitif advantage üretebilir.
- Aynı emrin operation/item/quantity seçimleri aynı yerel avantajı paylaşır.
  Diğer unit/market emirleri kendi avantajını kullanır.
- Zorunlu karar ve settlement actor loss'a girmez; terminal critic bunları görür.
  Yerel critic yalnız stochastic kararlar üzerinde eğitilir.
- Gradient norm clipping, PPO ratio clipping ve KL kontrolü korunur.

## Kullanım

Önce yeni koşu; eski koşunun klasörünü kullanmayın:

```bash
.venv/bin/python train_realized_chains.py \
  --run-dir runs/realized_chain_cuda_v1 --device cuda \
  --opponent starter --episodes 100 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --learning-rate 3e-5 \
  --chain-decay 0.8 --gae-lambda 1.0 --target-kl 0.02 \
  --max-grad-norm 1.0 --save-every 10
```

Aynı ayarlarla `--resume` devam ettirir; hedef/decay değişimi reddedilir.
Eski reward semantiğiyle eğitilmiş checkpoint reddedilir. Yarım episode
kaydı sessizce ezilmez. Deneme koşusu mekanik doğrulama içindir; 100 episode
kazanç veya policy kalitesi garantisi vermez.

Her episode: `chain_credit.json`, `advantages.json`, `cash_ledger.json`,
`lifecycle_events.jsonl.gz`, bitki/ürün projection'ları ve summary.
`--save-every` sıklığında tam observation/action trace ve oynatılabilir
`replay.json.gz`. Metrics terminal/local loss, kredi toplamları, kaynaksız
satış, KL, entropy, atlanan minibatch ve clipping öncesi gradient normunu ayırır.
