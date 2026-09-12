# Farmer, Market ve İşçi Credit Assignment Planı

## Amaç

Oyundaki farmer, farm hand ve market kararları aynı environment turunda birlikte
çalışmaya devam edecek. Değişiklik execution tarafında değil, PPO'nun her karara
hangi reward/advantage sinyalini verdiği tarafta yapılacak.

Hedefler:

- Kârlı trading kararının farmer'ın rastgele hareketini ödüllendirmemesi.
- Başarısız tarımsal kararların kârlı trading kararını cezalandırmaması.
- Birden fazla tokendan oluşan bir emrin, sırf uzun olduğu için PPO loss'unda
  daha fazla ağırlık almaması.
- `HIRE` maliyetinin doğrudan ilgili HIRE aksiyonuna yazılması.
- İşçinin üretime gerçek katkısının, o işçiyi oluşturan HIRE aksiyonuna kadar
  takip edilebilmesi.
- Terminal gold farkının nihai yarışma hedefi olarak korunması.

## Mevcut problem

Bir turda aşağıdaki kararlar aynı action paketinde üretilebiliyor:

```python
{
    "farmer": ["WATER"],
    "hands": [["HARVEST"], ["EAST"]],
    "market": [
        ["BUY_PRODUCT", "WHEAT", 5],
        ["SELL", "WHEAT", 2],
    ],
}
```

Environment önce unit aksiyonlarını, sonra market emirlerini uygular. Bu
aksiyonlar birbirini bloke etmez. Buna rağmen mevcut PPO kurgusunda token
log-probability'leri ortak bir karar paketi gibi ele alındığı için aynı global
advantage farmer, işçi ve market kararlarına birlikte yayılabilir.

Ayrıca lifecycle kredisi bir action span içindeki her stochastic tokena tam
aksiyon kredisini yazarsa, üç karar tokenı bulunan bir market emri tek tokenlı
`WATER` aksiyonundan daha fazla loss ağırlığı alır.

## Tasarım özeti

> **V1 notu:** Aşağıdaki iki dal ayrı model/encoder dalı değil, ortak
> autoregressive policy üzerindeki action-span ve loss ayrımıdır.

```text
                     Shared observation encoder
                                |
                 +--------------+--------------+
                 |                             |
          Unit policy branch            Market policy branch
          farmer + farm hands        trade + seed + hire + land
                 |                             |
          Farm action spans             Market action spans
                 |                             |
          Farm advantages          Trade/Hire advantages
                 +--------------+--------------+
                                |
                    Düşük ağırlıklı global hedef
```

Observation encoder ortak kalacak. Politika kararları ve action grammar mevcut
autoregressive yapıyı kullanabilir. Ayrım, action kayıtları, ekonomik defterler,
advantage hesapları ve PPO loss'unda yapılacak.

## 1. Semantik aksiyon kaydı

Her farmer, hand veya market emri tek bir `ActionSpanRecord` olarak saklanacak.

Önerilen veri yapısı:

```python
@dataclass(frozen=True, slots=True)
class ActionSpanRecord:
    action_id: str
    episode_step: int
    role: ActionRole
    actor_id: str | None
    token_start: int
    token_end: int
    old_log_probability: float
    entropy: float
```

Önerilen roller:

```python
class ActionRole(StrEnum):
    FARM = "farm"
    TRANSPORT = "transport"
    TRADE = "trade"
    PURCHASE = "purchase"
    HIRE = "hire"
    LAND = "land"
```

Bir action span'ın log-probability değeri, span içindeki stochastic tokenların
log-probability toplamıdır:

```python
span_log_probability = token_log_probabilities[
    token_start:token_end + 1
][decision_mask[token_start:token_end + 1]].sum()
```

`FARMER`, `MARKET`, `END_ACTION` gibi deterministic grammar tokenları toplama
katılmayacak.

## 2. Kaynak lotlarını ayırma

Shed içindeki aynı isimli ürünler iki farklı ekonomik kaynaktan gelebilir:

- Bir plant'ten hasat edilen ürün.
- Marketten `BUY_PRODUCT` ile alınan ürün.

Bu kaynaklar tek FIFO kuyruğunda miktar olarak karışsa bile lot üzerinde kaynak
bilgisi korunacak:

```python
@dataclass(slots=True)
class ProductLot:
    lot_id: str
    item: str
    quantity: int
    source_type: ProductSource
    source_plant_id: str | None
    source_purchase_id: str | None
    unit_cost: float
```

`ProductSource` en az `HARVEST` ve `MARKET_PURCHASE` değerlerini taşıyacak.
DROP/PICKUP/PLACE işlemleri lot kaynağını koruyacak. SELL sırasında shed'deki
lotlar FIFO tüketilecek ve her satış geliri kaynak lotlara paylaştırılacak.

Bu ayrım reward'ın iki kere yazılmasını engeller:

- Hasat ürünü geliri farm lifecycle hesabına gider.
- Satın alınmış ürün geliri trading hesabına gider.

## 3. Tarımsal kâr defteri

Mevcut seed lot ve plant lifecycle takibi korunacak. Satılmış her bitki için:

```text
plant_net = plant_sale_revenue - one_seed_cost
```

Bu net değer aşağıdaki semantik aksiyonlara aksiyon başına eşit dağıtılacak:

- İlgili `PLANT`
- İlgili `WATER` ve `FERTILIZE`
- İlgili `HARVEST`
- Ürünü shed'e taşıyan `DROP`/`PLACE`
- İlgili ürünü satan `SELL`

Satılmayan bitkinin üretim aksiyonları gerçekleşmiş kâr kredisi almayacak.
Kullanılmamış veya satılamamış seed maliyeti satın alma lotunun PURCHASE
kredisinde negatif kalacak.

## 4. Trading kâr defteri

Her `BUY_PRODUCT` emri bir purchase lot oluşturacak:

```text
purchase_id -> item, quantity, executed_unit_cost, action_span
```

Satış sırasında satın alınmış lotlardan gelen miktar için:

```text
trade_net = realized_sale_revenue - allocated_purchase_cost
```

Kredi, aynı trade zincirindeki BUY ve SELL action span'larına dağıtılacak.
Henüz satılmamış eldeki ürün için gerçekleşmiş trading kârı yazılmayacak;
episode sonunda kalan lot maliyeti ilgili BUY aksiyonunda negatif kalacak.

Market fiyatı her birimden sonra değişebildiği için maliyet ve gelir, istenen
miktar üzerinden değil gerçekten execute edilen her birim üzerinden tutulacak.

## 5. HIRE ve işçi katkısı

Her HIRE aksiyonu günlük benzersiz bir işçi kimliği oluşturacak:

```text
hire_id = episode/day/hire_sequence
actor_id = day_N_hand_M
hire_cost = gerçekleşen Fibonacci maliyeti
```

İşçi yalnızca o gün yaşadığı için gün sonunda kayıt kapatılacak. İşçinin yaptığı
her unit aksiyonu `actor_id` üzerinden kaydedilecek. Bir üretim zincirine bağlı
aksiyon gerçekleştirdiyse o aksiyonun farm kâr payı işçi katkısı olarak
biriktirilecek:

```text
hire_net = attributed_worker_profit - hire_cost
```

HIRE action span'ı `hire_net` alacak:

- Üretime katkı vermeyen işçi: doğrudan `-hire_cost`.
- Kârlı üretime katkı veren işçi: `profit_share - hire_cost`.

İşçinin üretim aksiyonuna verilen farm kredisi ile HIRE kredisi aynı ekonomik
değeri iki kez büyütmemeli. Bunun için işçi katkısı iki seçenekten biriyle
uygulanmalı:

1. Farm aksiyon payından HIRE'a belirli bir oran transfer etmek.
2. HIRE'a yalnızca maliyet yazıp üretim faydasını global return üzerinden
   öğrenmek.

İlk sürümde daha güvenli olan ikinci seçenek kullanılacak. Katkı transferi,
ayrı bir deney ve toplam-kredi invariant testi sonrasında açılacak.

## 6. Reward bileşenleri

> **SUPERSEDED (V1):** Bu bölümdeki dört yerel reward akışı uygulanmaz.
> Bağlayıcı tasarım global terminal GAE + episode-sonu ekonomik span
> kredisidir.

Her episode için dört ayrı return akışı üretilecek:

```python
RewardComponents(
    farm_profit=...,
    trade_profit=...,
    hire_net=...,
    global_gold_difference=...,
)
```

Normalize değerler aynı `reward_scale` ile hesaplanacak. Ekonomik bir tutar
yalnızca bir yerel deftere ait olacak. Terminal gold farkı bu tutarlardan ayrı
olarak nihai yarışma hedefini temsil edecek.

Potential shaping ilk geçişte mevcut haliyle korunabilir; fakat sonuçlar hem
shaping açık hem kapalı ablation koşularıyla ölçülmeden yerel realize kâr
kredileriyle birleştirilmemeli.

## 7. Ayrı value ve advantage head'leri

> **SUPERSEDED (V1):** Farm/trade/hire value head ve GAE yoktur. Yalnız global
> critic vardır; yerel ekonomik kredi action-conditioned critic olmadan
> sign-preserving Running RMS ile ölçeklenir.

Shared encoder üzerinde ayrı value head'leri kullanılacak:

```python
global_value = global_value_head(features)
farm_value = farm_value_head(features)
trade_value = trade_value_head(features)
hire_value = hire_value_head(features)
```

Her reward akışının GAE hesabı ayrı yapılacak:

```text
A_farm   = GAE(farm rewards, farm values)
A_trade  = GAE(trade rewards, trade values)
A_hire   = GAE(hire rewards, hire values)
A_global = GAE(global rewards, global values)
```

Yerel advantage yalnızca ilgili action span'a uygulanacak. Terminal global
advantage bütün kararlara küçük bir katsayıyla uygulanabilir; ilk deneyde bu
katsayı açıkça loglanacak ve konfigürasyondan değiştirilebilecek.

Önerilen başlangıç loss'u:

```text
L_actor =
    1.0 * L_farm
  + 1.0 * L_trade
  + 1.0 * L_hire
  + 0.1 * L_global
```

Bu katsayılar doğrulanmış sabitler değil, deney başlangıç değerleridir.

## 8. Action-span PPO

Her action span için yeni ve eski log-probability toplamları hesaplanacak:

```python
ratio = exp(new_span_log_probability - old_span_log_probability)
loss = -minimum(
    ratio * span_advantage,
    clip(ratio, 1 - epsilon, 1 + epsilon) * span_advantage,
)
```

Loss önce action span'lar üzerinde ortalanacak, sonra rol loss'ları ağırlıklı
birleştirilecek. Token sayısı hiçbir semantik aksiyonun toplam ağırlığını
artırmayacak.

Advantage normalizasyonunda işaret korunmalı:

- Özellikle PURCHASE ve HIRE için ortalamayı çıkarmak, bütün örnekler zararlı
  olduğunda daha az zararlı olan aksiyonu pozitif hale getirebilir.
- Bu roller için scale-only normalizasyon (`A / RMS(A)` veya `A / std(A)`) tercih
  edilecek.
- Farm ve trade rollerinde mean-centering ancak testlerle ayrı ayrı
  doğrulanırsa açılacak.

## 9. Entropy hesabı

Entropy token frekansına göre değil, önce head içinde ortalanacak:

```text
H_unit     = mean(unit decision entropies)
H_market   = mean(market decision entropies)
H_item     = mean(item decision entropies)
H_quantity = mean(quantity decision entropies)
```

Sonra yalnızca o minibatch'te aktif head'ler üzerinden ağırlıklı ortalama
alınacak. Böylece çok sık görülen bir head diğerlerinin entropy bonusunu
bastırmayacak.

## 10. Loglama ve denetlenebilirlik

Her episode için aşağıdakiler saklanacak:

- `action_spans.jsonl.gz`
- `product_lots.jsonl.gz`
- `trade_lifecycles.json`
- `worker_lifecycles.json`
- `credit_assignments.json`

Metrics içine şu alanlar eklenecek:

- Rol başına action span sayısı.
- Rol başına pozitif/negatif/total kredi.
- Farm, trade, hire ve global policy loss.
- Her value head için explained variance.
- Trading realized profit ve açık lot maliyeti.
- HIRE toplam maliyeti, üretken işçi oranı ve işçi başına katkı.
- Head-balanced entropy ve head başına KL.

Tek bir aksiyon için şu zincir okunabilmeli:

```text
action_id -> lot/plant/hire_id -> satış allocation -> kredi -> PPO span
```

## 11. Test planı

### Unit testleri

- Üç tokenlı BUY emri ve tek tokenlı WATER eşit action ağırlığı almalı.
- Deterministic grammar tokenları span log-probability hesabına girmemeli.
- Satın alınmış ve hasat edilmiş aynı ürün FIFO satışında kaynaklarını
  kaybetmemeli.
- Kısmi satışta maliyet doğru oranda lotta kalmalı.
- Satılmamış trading lotu pozitif kredi almamalı.
- Satılmamış seed order tam gerçekleşen maliyetini negatif almalı.
- İşe yararamayan HIRE action span'ı `-hire_cost` almalı.
- Gün sonunda hire kimliği kapanmalı ve yeni günün Fibonacci sayacı sıfırlanmalı.
- Role-specific scale-only normalizasyon reward işaretini değiştirmemeli.
- Toplam yerel kredi ekonomik net kârı aşmamalı.

### Entegrasyon testleri

- Aynı turda farmer aksiyonu ve market emri uygulanmalı.
- Bu tur alınan seed ancak sonraki tur ekilebilmeli.
- Bu tur shed'e bırakılan ürün aynı turda satılabilmeli.
- Kârlı trading trajectory'sinde yalnızca TRADE/PURCHASE span'ları yerel trading
  kredisi almalı.
- Tarımsal satışta BUY_PRODUCT span'ları farm kredisi almamalı.
- PPO update sonrasında bütün value/policy head'lerinde sonlu gradient olmalı.

### Davranış testleri

Sabit seed'lerle üç kısa karşılaştırma koşusu yapılacak:

1. Mevcut joint PPO baseline.
2. Action-span PPO, eski reward.
3. Action-span PPO, ayrılmış farm/trade/hire reward.

Karşılaştırılacak metrikler:

- Starter'a karşı win rate ve gold advantage.
- Gerçekleşmiş farm ve trading kârı.
- HIRE başına net katkı.
- Hiç kullanılmayan seed/product maliyeti.
- Head entropy çöküş hızı.
- Aynı davranışın farklı token uzunluklarındaki loss ağırlığı.

## 12. Uygulama sırası

### Aşama 1 — Action-span altyapısı

- `ActionSpanRecord` oluştur.
- Rollout ve batch collation içine span bilgilerini taşı.
- Span log-probability ve span PPO loss'unu ekle.
- Mevcut lifecycle kredisini token yerine span'a bağla.
- Eşit aksiyon kredisi testlerini geçir.

### Aşama 2 — Trading lot takibi

- BUY_PRODUCT lotlarını kaydet.
- DROP/PICKUP/shed FIFO kaynaklarını koru.
- SELL gelirini kaynak lotlara paylaştır.
- Trading realized P&L ve açık lot maliyetini logla.

### Aşama 3 — Reward/value ayrımı

> **SUPERSEDED (V1):** Aşağıdaki dört tensor/head adımları sonraki
> action-conditioned critic ablation'ına ertelenmiştir.

- Farm, trade, hire ve global reward tensorlarını üret.
- Ayrı value head'lerini ekle.
- Rol bazlı GAE ve PPO loss'larını bağla.
- Sign-preserving advantage normalizasyonunu uygula.

### Aşama 4 — HIRE muhasebesi

- Günlük hire/worker kimliklerini takip et.
- İlk sürümde HIRE'a doğrudan maliyet yaz.
- İşçi katkısını yalnızca logla.
- Katkı transferini ablation sonuçlarından sonra değerlendirmeye aç.

### Aşama 5 — Entropy, KL ve gözlemleme

- Head-balanced entropy uygula.
- Head ve rol başına KL/loss/advantage logla.
- Replay inceleme aracına action-lot-credit bağlantılarını ekle.

### Aşama 6 — Eğitim doğrulaması

- Unit ve entegrasyon testlerinin tamamını çalıştır.
- Kısa smoke training yap.
- Sabit seed'li baseline karşılaştırmasını tamamla.
- Eski checkpoint ile yeni loss semantiğini karıştırmadan temiz bir run başlat.

## 13. Migration ve çalışma güvenliği

Bu değişiklikler rollout/batch semantiğini ve value head'lerini değiştireceği için
mevcut `long_v6` checkpoint'i yeni eğitim düzenine doğrudan resume edilmemeli.
Model encoder/policy ağırlıklarını kontrollü biçimde transfer etmek ayrı bir
seçenek olabilir; optimizer, value head ve PPO buffer state'i yeniden
başlatılmalıdır.

Aktif training süreci kod düzenlenirken otomatik durdurulmayacak. Yeni sistem,
testler geçtikten sonra farklı bir run dizininde başlatılacak.

## Başarı ölçütü

Uygulama şu koşullar birlikte sağlandığında tamamlanmış sayılacak:

- Kârlı saf trading oyununda farmer span'ları yerel trading kredisi almıyor.
- Zararlı HIRE, başka bir aksiyon başarılı olsa bile negatif yerel HIRE kredisi
  taşıyor.
- Tarımsal zincirin realized net kârı yalnızca zincirdeki semantik aksiyonlara
  dağıtılıyor.
- Action token uzunluğu PPO ağırlığını değiştirmiyor.
- Terminal gold farkı modelin nihai hedefinde kalıyor.
- Sabit seed değerlendirmesinde ayrıştırılmış model baseline'dan daha iyi veya
  en azından eşit win rate üretirken israf ve gereksiz HIRE oranını düşürüyor.

## İnceleme notları (2026-08-30)

`long_v6` semptomları ve mevcut kod (`training.py` span içi her karara tam
miktar, `lifecycle_credit.py` şema 2, Φ'de HIRE/bina yok) üzerinden.

### Teşhis

Joint PPO bir turun `WATER` + boş `EAST` + `BUY_PRODUCT`/`SELL` paketine tek
`A_t` veriyor. Token kaybı span'deki her stochastic tokena tam krediyi yazıyor;
üç tokenlı alış tek tokenlı `WATER`'dan üç kat ağır basıyor.

`long_v6` bunu doğruladı: alış israfı miktar tokenına yapıştı, `BUY_SEED` söndü;
`HIRE` ne token kaybında ne Φ'de var, maskeli hire'da ~148 boş işçi.

Planın hedefleri (paketi ayır, span eşitle, HIRE'a maliyet, hasat ile
`BUY_PRODUCT`'u karıştırma, terminal gold'u ayrı tut) bu kırılmalarla örtüşüyor.
Execution'a dokunmamak doğru.

### Korunacak kararlar

- Semantik emir = bir PPO terimi; `FARMER` / `END_ACTION` log-prob'a girmez.
  Aşama 1 tek başına mevcut şema 2 şişmesini keser.
- Lot kaynağı: tracker `source_plant_id is None` satışına farm kredisi vermiyor
  ama `source_purchase_id` yok. Probe churn'u farm'a sızmıyor, trade defteri de
  yok; GAE ~0 ile ödüllenebiliyor. Aşama 2 bunu kapatır.
- Farm defteri şema 2 ile aynı kalsın: `plant_net = satış - 1 tohum`, satılmayan
  ekim 0, israf `PURCHASE`'te.
- HIRE v1 = yalnız `-maliyet` (seçenek 2). Boş işe alımı keser; kâr transferi
  çift sayımı riskli, ablation sonrası.
- PURCHASE / HIRE için scale-only normalize. Mevcut rol-içi mean-center, hepsi
  zararlıyken en az kötü `QTY`'yi artı yapıyor; politika "hiç alma"ya kaçtı.
- `long_v6` resume yok. Value head ve buffer semantiği değişir.
- Test listesindeki eşit ağırlık, FIFO kaynak, satılmamış lot, boş HIRE ve
  toplam kredi ≤ ekonomik net, Aşama 3'ten önce zorunlu.

### Asıl çatlak: iki kredi mekanizması

§3–5 episode sonunda span'a realized kâr yazar (bugünkü token loss). §6–7 aynı
ekonomiyi tur `r_t` + ayrı GAE + dört value head yapar.

İkisini birden açmak bir `PLANT` için üç sinyal üretir: Φ sıçraması (`β=0.25`),
satıştan geri `A_farm`, span `plant_net`. §6 "shaping + yerel kârı ablation'sız
birleştirme" diyor; §7 loss'u ikisini varsayıyor.

İlk sürümde farm / trade / hire realized kârı yalnız span advantage olmalı.
GAE yalnızca `A_global` (terminal ve isteğe bağlı zayıf shaping). Ayrı value
head'ler Aşama 3b; shaping kapalı/açık ablation'dan sonra. Aksi halde seyrek
`r_farm` (çoğu tur 0) ve `λ=0.95` ile gürültülü critic.

### Diğer boşluklar

- `BUY_SEED` farm defterinde kalmalı (şimdi öyle). Loss formülünde `L_purchase`
  yok (`L_farm + L_trade + L_hire + L_global`). Tohum israfının `L_farm`'a mı
  gireceği açık yazılmalı; ayrı `L_purchase` veya `PURCHASE`'in farm span
  loss'unda olduğu belirtilmeli.
- HIRE v1 "üretimi global'den öğren" `λ=0.95` ve `L_global=0.1` ile
  HIRE→ekim bağını öğrenmez. v1 hedefi win rate değil, `HIRE` / bölüm düşüşü.
- Aynı tur `HIRE` edilen el o tur oynamaz (market unit'ten sonra). `actor_id`
  eşlemesi `hires_today` sırası + sonraki tur `hands[i]`; buna test şart.
- `TRANSPORT` (DROP) §3'te farm kârına giriyor, §1'de ayrı rol. Span loss'u
  `L_farm`'da mı netleştirilmeli.
- Shaping açıkken boş HIRE'ı hem Φ nakit düşüşü hem HIRE v1 cezalar. Shaping
  kapalı smoke şart.
- Müfredat / `full` unmask bu planın dışında; credit assignment düzelir, 401
  tipi DIG çöküşü ayrı kalır.
- Diyagramdaki "unit / market policy branch" encoder veya head bölmek zorunda
  değil. Ayrım loss ve span'de; `model.py`'ye farm/market dal eklemeyin.

### Uygulama sırası düzeltmesi

Aşama 1 → 2 → 4 (HIRE maliyet) → 5 doğru. Aşama 3 ikiye bölünsün: önce span
advantage + scale-only + tek critic; sonra dört head / dört GAE.

Aşama 1 miktar şişmesini, Aşama 2 probe trade churn'unu, Aşama 4 boş HIRE'ı
kapatır. Dört value head olmadan `long_v6` semptomlarının çoğu gider.

### Başarı ölçütü (v1 için daraltma)

Win ≥ baseline v1 için ağır. İlk kabul:

1. Trading-only trajectory'de farmer span'ı yerel trade kredisi almıyor.
2. Boş HIRE negatif yerel kredi taşıyor.
3. Üç tokenlı emir ile tek tokenlı emir aynı PPO ağırlığında.
4. Kullanılmayan tohum / ürün maliyeti düşüyor.
5. Smoke'da win baseline'dan kötüleşmiyor.

"İşçi tut ki ekilsin" ve starter win artışı kâr-transferi + uzun ufuk
ablation'ından sonra.

## Nihai revizyon önerileri

Bu bölüm önceki taslak ve inceleme notları arasındaki çelişkileri çözen normatif
v1 kararlarını içerir. Uygulama sırasında bir önceki bölümle çelişen noktada bu
bölüm esas alınmalıdır.

### 1. V1 kapsamı

İlk sürüm yalnızca şu kararları yerel ekonomik krediyle ayrıştıracak:

- Crop seed satın alma ve crop üretim zinciri.
- `BUY_PRODUCT` ile başlayan saf trading zinciri.
- `HIRE` maliyeti.

`BUY_ANIMAL`, `BUILD_COOP`, `BUILD_PASTURE`, `FEED`, `CARE`, `BUY_LAND` ve
navigasyon hareketleri v1'de yeni bir yerel ekonomik defter almayacak. Bunlar
global terminal hedefiyle öğrenmeye devam edecek ve ayrı bir sonraki aşamada
ele alınacak.

### 2. Tek ekonomik tutar, tek yerel dağıtım

V1'in temel invariant'ı:

```text
sum(local_credits) = realized_revenue - executed_raw_material_cost - hire_cost
```

Bu eşitlik yalnızca v1 kapsamındaki kapanmış lotlar ve HIRE maliyetleri için
hesaplanır. Açık lotların gerçekleşmiş maliyeti de sağ tarafta maliyet olarak
kalır. Aynı satış geliri veya hammadde maliyeti iki farklı yerel loss'a tam
değeriyle tekrar yazılamaz.

Mevcut şema 2'deki aşağıdaki ikili aynı anda kullanılmayacak:

```text
purchase_credit = descendant_sales - full_order_cost
plant_credit    = plant_sales - one_seed_cost
```

Çünkü aynı satış ve tohum maliyetini PURCHASE ile üretim zincirinde yeniden
saymaktadır.

### 3. Düzeltilmiş crop kredi algoritması

Her execute edilmiş seed order bir `seed_lot_id` oluşturur. Order'ın maliyeti
birim seed maliyetlerine ayrılır. Satılmış her plant için:

```text
plant_chain_net = plant_realized_sale_revenue - one_seed_cost
```

Bu net tutar aşağıdaki benzersiz semantik action span'ların birleşimine bir kez
dağıtılır:

- Seed'i sağlayan `BUY_SEED` span'ı.
- `PLANT` span'ı.
- İlgili `WATER` ve `FERTILIZE` span'ları.
- İlgili `HARVEST` span'ları.
- Ürünü shed'e getiren `DROP` veya `PLACE` span'ları.
- Ürünün gelirini gerçekleştiren `SELL` span'ları.

```text
action_credit = plant_chain_net / unique_chain_action_count
```

Bir seed order birden fazla satılmış plant'i besliyorsa BUY_SEED span'ı her
plant zincirinden kendi payını biriktirir. Episode sonunda kullanılmamış veya
satışa dönüşmemiş seed miktarının maliyeti BUY_SEED span'ına ayrıca negatif
yazılır:

```text
unused_seed_credit = -unused_quantity * executed_unit_cost
```

Satılmamış bir plant'in üretim aksiyonları gerçekleşmiş kâr kredisi almaz.
Ancak onun seed maliyeti yukarıdaki kullanılmamış/geri dönmemiş maliyet olarak
BUY_SEED span'ında kalır.

Bu yöntem sonunda bütün crop span kredilerinin toplamı, crop satış geliri eksi
execute edilmiş toplam seed maliyetine eşit olmalıdır.

### 4. Düzeltilmiş trading kredi algoritması

Her execute edilmiş `BUY_PRODUCT` emri kaynak kimliğini koruyan product lotları
oluşturur. SELL sırasında miktar FIFO ile kaynak lotlara ayrılır.

Satılan her lot parçası için:

```text
sold_part_net = allocated_sale_revenue - allocated_purchase_cost
```

Bu tutar yalnızca ilgili BUY_PRODUCT ve SELL action span'ları arasında bir kez
paylaştırılır. İlk sürümde eşit paylaşım kullanılabilir:

```text
buy_credit  += sold_part_net / 2
sell_credit += sold_part_net / 2
```

Episode sonunda elde kalan miktarın maliyeti ilgili BUY_PRODUCT span'ına
negatif yazılır:

```text
open_lot_credit = -remaining_quantity * executed_unit_cost
```

Bir SELL emri aynı item'ın hem hasat hem market-purchase kaynaklı miktarını
satabilir. Bu durumda tek SELL span'ına farklı kaynaklardan birden fazla
`CreditAssignment` bağlanır; action kaydının tek bir ekonomik role sahip olduğu
varsayılmaz.

FIFO oyun motorunun gerçek item kimliği olmadığı için deterministik bir sentetik
attribution kuralıdır. Loglarda bu ayrım `attribution_method="fifo"` olarak
açıkça belirtilmelidir.

### 5. Action kaydı ile kredi kaydını ayırma

`ActionSpanRecord` ekonomik olarak tek bir `role` taşımamalıdır. Önerilen yapı:

```python
@dataclass(frozen=True, slots=True)
class ActionSpanRecord:
    action_id: str
    episode_step: int
    section: ActionSection
    operation: str
    actor_id: str | None
    token_start: int
    token_end: int
    old_log_probability: float
    entropy: float


@dataclass(frozen=True, slots=True)
class SpanCreditAssignment:
    assignment_id: str
    action_id: str
    credit_role: CreditRole
    source_id: str
    amount: float
    reason: str
```

Bir action span sıfır, bir veya birden fazla kredi kaydı alabilir. SELL gibi
karma kaynaklı aksiyonlar bu sayede hem FARM hem TRADE attribution'ı taşıyabilir.

Bir span içinde stochastic karar tokenı yoksa kredi loglanır ancak o span için
PPO actor terimi üretilmez.

### 6. V1 HIRE davranışı

HIRE ayrı bir varlık olarak potential içinde değerlenmese de maliyet cash'i
azalttığı için mevcut shaping üzerinden zaten küçük negatif sinyal alır. Bu
nedenle HIRE v1 uygulanırken potential shaping kapatılmalıdır.

V1 HIRE kredisi:

```text
hire_credit = -executed_hire_cost
```

Maliyet, istenen HIRE sayısından değil environment state farkından ve gerçekten
oluşan elden doğrulanmalıdır. Günlük Fibonacci maliyeti ve `hires_today` sırası
loglanmalıdır.

Market unit işlemlerinden sonra çalıştığı için bu tur kiralanan işçi aynı tur
unit aksiyonu yapamaz. `hire_id`, bir sonraki turdaki `hands[index]` kaydıyla
eşleştirilmelidir. Gün sonunda hands silindiğinde hire yaşam döngüsü kapanır.

V1'in amacı HIRE win rate'ini optimize etmek değil, ekonomik karşılığı olmayan
işçi kiralamayı azaltmaktır. Daha sonraki sürümde işçinin üretim aksiyonuna düşen
kredinin bir kısmı HIRE span'ına **transfer** edilebilir:

```text
worker_action_credit_after = (1 - alpha) * original_worker_action_credit
hire_benefit_credit        = alpha * original_worker_action_credit
```

Bu bir bonus olarak eklenmemeli; aksi halde çift sayım oluşur. `alpha` yalnız
ablation ile açılmalı ve toplam-kredi invariant'ı değişmemelidir.

### 7. V1 reward ve advantage düzeni

İlk sürümde iki ayrı öğrenme kanalı kullanılacak:

1. Global kanal: terminal gold farkı için mevcut global critic ve GAE.
2. Yerel kanal: episode sonunda çözülmüş action-span ekonomik kredileri.

V1'de şunlar yapılmayacak:

- Farm/trade/hire için transition reward dizisi üretilmeyecek.
- Farm/trade/hire için ayrı GAE çalıştırılmayacak.
- Dört yeni state-only value head eklenmeyecek.
- Yerel realized kâr hem transition reward hem span credit olarak
  tekrarlanmayacak.

Yerel span kredisi doğrudan bir Monte Carlo action return olarak kullanılacak.
İşareti koruyan running scale ile ölçeklenecek:

```text
scaled_span_credit = span_credit / max(running_rms[role], epsilon)
```

Minibatch ortalaması çıkarılmayacak. Böylece tüm PURCHASE veya HIRE örnekleri
zararlıyken daha az zararlı olan örnek yapay olarak pozitif hale gelmez.

Yerel action-conditioned critic ancak bu sürüm doğrulandıktan sonra Aşama 3b'de
tasarlanacaktır. Sadece state gören bir `farm_value_head(state)`, aynı turdaki
farmer ve farklı işçi span'larının farklı getirilerine uygun baseline değildir.

### 8. V1 PPO loss tanımı

Yerel her semantik aksiyon için span içindeki stochastic token log-probability
değerleri toplanır:

```python
new_span_logp = new_token_logp[span_decision_mask].sum()
old_span_logp = old_token_logp[span_decision_mask].sum()
ratio = torch.exp(new_span_logp - old_span_logp)
```

Yerel loss:

```text
L_local = role_balanced_mean(PPOClip(ratio, scaled_span_credit))
```

Roller önce kendi içinde ortalanır, sonra konfigüre edilmiş sabit role weights
ile birleştirilir. Böylece binlerce trading emri farmer veya HIRE loss'unu sırf
sayısı fazla olduğu için bastırmaz.

Global terminal sinyali de uzun joint action ratio'suna geri dönmemelidir. Aynı
global advantage rol dengeli span ratio'larına düşük, konfigüre edilebilir bir
katsayıyla uygulanmalıdır:

```text
L_actor = L_local + global_actor_coef * L_global_span
```

Bu, tam joint-policy PPO oranının birebir aynısı değil; autoregressive karar
paketi için bilinçli bir factorized surrogate'tır. Bu yaklaşık seçim kod ve
config dokümantasyonunda açıkça belirtilmelidir.

`global_actor_coef` için `0.1` yalnız başlangıç hipotezidir; `0.0`, `0.1` ve
`1.0` sabit-seed ablation ile karşılaştırılmadan varsayılanlaştırılmamalıdır.

### 9. Potential shaping kararı

İlk action-span smoke ve karşılaştırma eğitimlerinde:

```text
shaping_beta = 0.0
```

kullanılmalıdır. Böylece HIRE maliyeti veya plant değeri hem potential hem
yerel realized krediyle iki kez verilmez. Sistem doğrulandıktan sonra shaping
kapalı/açık ayrı deney olarak denenebilir; aynı run içinde sonradan açılmamalıdır.

### 10. Entropy ve KL

Entropy önce head içinde, sonra yalnız aktif head'ler arasında ağırlıklı
ortalama alınmalıdır. PPO loss'taki role balancing ile entropy head balancing
ayrı kavramlardır ve ayrı loglanmalıdır.

KL aşağıdaki seviyelerde raporlanmalıdır:

- Stochastic token/head başına conditional KL.
- Semantik action span başına log-ratio/KL yaklaşımı.
- Rol başına ortalama KL.

Joint action token sayısıyla büyüyen tek KL değeri, tek başına early stop ölçütü
olarak kullanılmamalıdır. Early stop öncelikle rol/head conditional KL
eşiklerine dayanmalıdır.

### 11. Zorunlu invariant ve testler

Kod training'e bağlanmadan önce aşağıdaki testlerin tamamı geçmelidir:

1. Kapanmış crop lotunda toplam span kredisi tam olarak satış eksi seed
   maliyetine eşit olmalı.
2. Kapanmış trading lotunda BUY + SELL kredisi tam olarak realized trading
   kârına eşit olmalı.
3. Kısmen satılmış lotta satılan kısım neti ile açık miktar maliyeti ayrı
   korunmalı.
4. Aynı SELL içindeki hasat ve satın-alım kaynakları ayrı assignment üretmeli.
5. Üç stochastic tokenlı emir ile tek stochastic tokenlı emir, aynı span
   credit için aynı toplam PPO ağırlığını almalı.
6. Deterministic veya sıfır-stochastic-token span actor loss'a girmemeli.
7. PURCHASE/HIRE running-RMS ölçekleme kredi işaretini değiştirmemeli.
8. Başarısız HIRE tam gerçekleşen maliyeti kadar negatif kredi almalı.
9. HIRE edilen el aynı tur değil, bir sonraki turdaki doğru `hands[index]` ile
   eşleşmeli.
10. Yerel kredi transition reward'a tekrar eklenmemeli.
11. Aynı ekonomik kaynak iki `SpanCreditAssignment` içinde toplam ekonomik
    değerini aşmamalı.
12. Sabit action span sayısıyla token padding veya grammar tokenı eklemek loss'u
    değiştirmemeli.

### 12. Revize uygulama sırası

1. Mevcut reward davranışını değiştirmeden `ActionSpanRecord` ve span-level
   log-probability altyapısını ekle.
2. Mevcut lifecycle token loss'unu span loss'a geçir ve token uzunluğu eşitlik
   testini tamamla.
3. Kaynak-korumalı crop ve BUY_PRODUCT lot defterlerini ekle.
4. Şema 2 double-counting davranışını korunumlu crop/trade dağıtımıyla değiştir;
   credit scheme version değerini artır.
5. HIRE gerçekleşen maliyet kaydını ve doğrudan negatif span kredisini ekle.
6. Running-RMS sign-preserving ölçekleme, role-balanced loss, head-balanced
   entropy ve rol/head KL loglamasını bağla.
7. `shaping_beta=0`, yeni run dizini ve temiz optimizer ile smoke training yap.
8. Joint baseline, yalnız action-span ve tam ayrıştırılmış v1 koşularını aynı
   environment seed'lerinde karşılaştır.
9. V1 başarılı olduktan sonra action-conditioned local critic ve HIRE katkı
   transferini ayrı ablation aşamalarında değerlendir.

### 13. V1 kabul ölçütleri

V1 aşağıdaki koşullar birlikte sağlandığında kabul edilir:

- Ekonomik kredi invariant testlerinin tamamı geçer.
- Saf trading trajectory'sinde farmer ve farm-hand span'ları yerel TRADE
  kredisi almaz.
- Saf crop trajectory'sinde ilgisiz market trading span'ları FARM kredisi
  almaz.
- Boş HIRE tam maliyetini negatif yerel kredi olarak taşır.
- Action token uzunluğu toplam PPO ağırlığını değiştirmez.
- Kullanılmayan seed/product maliyeti ve oyun başına gereksiz HIRE baseline'a
  göre düşer.
- Sabit-seed smoke değerlendirmesinde NaN, KL patlaması veya baseline'a göre
  açıklanamayan ciddi win-rate çöküşü görülmez.
- Aktif `long_v6` checkpoint'i yeni optimizer/value/buffer semantiğiyle resume
  edilmez; yeni deney farklı run dizininde başlar.

## İkinci inceleme (2026-08-30)

Üst taslak (§1–13) ile inceleme notları çelişiyordu. **Nihai revizyon** v1 için
bağlayıcı kabul edilmeli; kod o bölümden yazılmalı. Revizyon önceki çatlağı
(dört GAE + shaping + span üçlüsü) kapatıyor. Aşağıdaki kararlar v1
için bağlayıcıdır.

### Revizyonun kilitlediği (doğru)

- Invariant: `sum(local) = satış − hammadde − HIRE`. Şema 2 çift sayımı
  (`alış = satış−tam maliyet` ve `bitki = satış−1 tohum`) kasten yok.
- 50 al / 1 sat ($51): `plant_chain_net = 41` zincire bir kez, `unused = −490`
  yalnız `BUY_SEED`, toplam `−449 = 51 − 500`.
- Yerel kâr yalnız episode sonu span kredisi. Farm/trade/hire GAE ve dört
  value head yok. Critic yalnız global.
- Smoke'da `shaping_beta = 0`. HIRE maliyeti Φ + yerel eksi olmaz.
- HIRE v1 = `−gerçek maliyet`, sonraki tur `hands[i]`, hedef win değil.
- Span tek ekonomik rol taşımaz. SELL hem FARM hem TRADE assignment alabilir.
- Scale-only RMS; `long_v6` resume yok; v1 kabul win dayatmaz.

### Kilitlenen dört karar

1. **Aynı span, birden fazla kredi.** `L_local` `scaled_span_credit` diyor.
   SELL'e +20 farm ve −5 trade yazılırsa toplanıp tek PPO terimi mi, yoksa
   aynı `ratio` ile iki rol terimi mi? Invariant tek terimi tercih eder:
   `span_credit = sum(assignments)`. Rol dengesi operasyona veya en büyük
   mutlak krediye bağlansın.
2. **PURCHASE kovası.** Tohum alış farm zincirinde; `role_balanced_mean`
   `PURCHASE` ile `FARM`'ı ayırmalı. Aksi halde binlerce `WATER` alış
   israfını bastırır. `BUY_SEED` payı çok `WATER`'da küçük kalır; miktar
   öğrenmesi `unused` teriminden gelir. Logda `buy_share` ve `unused_credit`
   ayrı dursun.
3. **`global_actor_coef=0` smoke.** Factorized surrogate joint `log π(a|s)`
   değildir. `0.1` ablation'sız varsayılan olmasın. İlk smoke yalnız yerel
   sinyali izlemeli.
4. **İlk koşu maskeli.** V1 + `β=0` iken DIG/BUILD/arazi/hayvan yalnız kısa
   GAE'li terminal görür. `full` veya probesız DIG kör kalır. Smoke
   `wheat_loop` / `wheat_hire` maskesiyle. Bunu kabul maddesine ekleyin.

### Belge hijyeni

§6–7 hâlâ dört GAE anlatıyor; eski Aşama 3 duruyor. Üstüne *superseded*
yazılmazsa biri yanlış implement eder. Diyagramdaki “iki policy dalı” ağ
bölüyormuş gibi; v1 ayrımı yalnız loss ve span'de.

### Uygulama

Sıra doğru: span altyapısı → eşit ağırlık → lot defteri → invariant
crop/trade → HIRE −maliyet → RMS/role/entropy → `β=0` yeni run. Yukarıdaki
dört karar yazılmadan Aşama 4'e (HIRE) geçilmesin.

## Uygulama durumu (economic-span v4)

Bağlayıcı dört karar şu kesin davranışlarla uygulanmıştır:

1. Attribution kayıtları kaynak bazında ayrı kalır; training'de aynı
   `action_id` tutarları toplanır ve tam bir PPO ratio'su üretir. Karma SELL'in
   denge rolü, mutlak toplam katkısı en büyük roldür; eşitlik enum sırasıyla
   deterministik çözülür.
2. `BUY_SEED` payı ve kullanılmayan tohum cezası `PURCHASE` kovasındadır.
   Ayrı assignment reason değerleri
   `seed_purchase_share_of_plant_realized_net` ve `unused_seed_credit` olarak
   loglanır. `BUY_PRODUCT`/SELL arbitrajı `TRADE` kovasında kalır.
3. CLI varsayılanı `global_actor_coef=0.0`, `shaping_beta=0.0`'dır.
4. CLI varsayılanı maskeli curriculum ve `probe_probability=0.0`'dır; ilk
   aşamalar `wheat_loop` ve `wheat_hire` olarak ilerler.

Kod yolları ayrıdır: legacy `causal_chain` şema 2 korunur; yeni
`economic_span` şema 4, `market_execution.py` ve `span_credit.py` üzerinden
çalışır. Checkpoint'te training semantics v4 ve RunningRoleRMS v2 zorunlu
olduğundan `long_v6` bu hatla resume edilemez.

Doğrulama: 88 unit/entegrasyon testi geçti. İki episode'luk gerçek CLI smoke
update'inde her episode için ekonomik invariant hatası `0.0`, checkpoint RMS
state sürümü `2` ve bütün update metrikleri sonlu kaldı. Bu smoke yalnızca
mekanik kabul testidir; davranışsal başarı veya baseline üstünlüğü iddiası
değildir.

## v5 — terminal hedefi ve bağımsız nakit muhasebesi

Güncel sözleşme `TRAINING_V5.md` içindedir. Terminal actor varsayılan 1,
lifecycle yardımcı actor varsayılan 0; STOP ayrı bir gerçek karar span'ıdır.
Tam iki taraflı kasa mutabakatı `cash_accounting.py` üzerinden her tur çalışır.
HIRE/arazi terminal gradyan yönü ve gerçek motor maliyeti testleri eklendi.
Hediyesiz, tam-aksiyon, eşleştirilmiş seed/side değerlendirmesi
`compare_checkpoints.py` ile yürütülür. Önceki v4 optimizer semantiğiyle
sessiz resume yasaktır; navigation ve hayvan lifecycle ataması tamamlandı
iddiası yoktur.
