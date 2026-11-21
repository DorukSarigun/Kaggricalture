# Görev kapanışı: gerçekleşen üretim, karşılanmayan talep ve kredi DAG'ı

`issue_completion.py`, `resource_evidence.py`, `issue_outcomes.py`.
Manager-only ve ortak eğitim aynı kapanış yolunu kullanır. Yeni bir action head,
kapasite maskesi, kaynak rezervasyonu veya crew cezası yoktur.

## EXPIRED neden tek başına başarısızlık değil?

Üreticinin yaşam süresi, istenen teslimat kotası ve satış farklı kavramlardır.
Canlı hayvanın kaydı sezon sonunda EXPIRED olabilir, ama daha önce kota kadar
ürünü teslim etmiş veya başka üretimde kullandırmış olabilir. Kaydın tarihsel
durumu değiştirilmez; `demand_fulfilled` ayrı ve miktardan hesaplanır.

Her ürün için:

- `harvested`: gerçekleşmiş HARVEST/COLLECT_FERTILIZER miktarı.
- `sold`, `consumed`, `held`, `lost`: bu miktarın birbirini dışlayan son durumları.
- `fulfilled_unique`: ilk depo teslimatı, ara ürün tüketimi veya satışa ulaşan
  benzersiz birimler. Depo → taşı → yem/satış aynı birimi tekrar saymaz.
- `ready_unharvested`: son observation'da hasada hazır görünen miktar.
  Bitkide yalnız yield_units yeterli değildir; ilk hasat yaşı da kontrol edilir.
- `requested`, `fulfilled`, `unmet`: issue'nun gerçek ürün kotası.

Denetim: `harvested == sold + consumed + held + lost`.
`ready_unharvested` bu eşitliğe eklenmez: henüz envantere girmiş değildir.
Kota dışı yan ürün, örneğin istenmemiş gübre, yumurta/süt kotasını kapatamaz.

Kaynaklar gerçek lot ID'lerinden izlenir. Ham alınmış ürünün tekrar satılması
bir üretici issue'ya yazılmaz. Kaybolan mal, gerçekleşen üretimi silmez.
`actual_inputs_consumed` yalnız gerçekten ekilen tohum/yerleştirilen hayvan ve
tüketilen yem/gübreyi sayar. `startup_resource_windows`, başlangıç girdisinin
hangi tur aralıklarında mevcut olduğunu gösterir; bir FIFO diagnostic, rezervasyon değil.

## Ceza bütçesi

Yeni bir sabit EXPIRED cezası ya da hayali kaçırılmış satış geliri eklenmez.
Önce mevcut bekleme/bakım olayları ham bütçeyi oluşturur. Varsayılan:

```text
eksik oran = ürün kotalarında toplam karşılanmayan birim / toplam istenen birim
nihai görev maliyeti = ham görev maliyeti × eksik oran
```

Örneğin 10 hedefin 6'sı gerçekten teslim/kullanıma ulaşmışsa, −0.02 ham bütçe
−0.008 olur. Tamamlanan görevde görev maliyeti sıfırdır; hiç üretmeyen görevde
oran 1'dir. Sıfır ham bütçeden sırf EXPIRED diye yeni ceza üretilmez.

`--task-cost-settlement unfulfilled_fraction` bu davranışı seçer.
`--task-cost-settlement preserve` ham görev maliyetini aynen korur.
Bu seçim checkpoint config'inde saklanır; resume sırasında değiştirilemez.

Birden çok türde kota varsa adetler üzerinden ağırlıklandırılır; piyasa değeri
ve gelecekteki tahmini getiri kullanılmaz. Bu açık bir eğitim hedefi seçimidir.

## Kanıtlı satış → yem açığı → aksayan görev

```text
                     Karşılanmayan talep maliyeti
                            /          \
                     ISSUE rolü      Kaynak çekme rolü
                                         |
                                    gerçek SELL
```

Yalnız o anda aç ve gerçekten mevcut hayvanların buğday ihtiyacını azaltan
gerçekleşmiş satışlar adaydır. İhtiyacın üzerindeki stok satışı kanıt sayılmaz.
Crew aşamasında gelen buğday (hasat vb.) eski çekilme kanıtını FIFO kapatır.
Market alışları ünitelerden sonra gelir; o turdaki yem fırsatını geri getirmez
ve önceki satışın sorumluluğunu silmez. Kanıt ancak bir sonraki crew
aşamasında, stok ünitelerin önünde dururken kapanır. Beslenen/kapanan issue
için kanıt emekliye ayrılır. Sonradan açılan issue önceki satışa bağlanmaz.

Üniteler marketten önce çalışır. Bu nedenle aynı turdaki SELL, aynı turun
daha önce aksayan FEED işleminin nedeni yapılamaz. Aynı turdaki BUY da o
turun kaçırılmış yemini telafi edemez; önceki satışın kanıtını silmez.
Kanıt en erken sonraki turun bakım aksamasına bağlanabilir. Bir çekilmiş
kaynak birimi aynı turda iki ayrı hayvanın açığını açıklamak için kullanılamaz.

Kanıt varsa tek maliyet bütçesi ISSUE rolü ve kaynak çekme rolüne yarı yarıya
dağıtılır; çekme rolündeki satışlar miktarla ağırlıklandırılır. Rol içi ve DAG
payları toplamı bütçeyi korur. Bu kesin karşı-olgusal neden tahmini değil,
gözlenen kaynak kısıtına dayalı, açıkça tanımlanmış kredi paylaşımıdır.

Kaynak hareketleri gözlemlerle uyuşmazsa satış atfı bırakılır ve
`unexplained_stock_change` kaydedilir. Kanıt yoksa ISSUE sorumluluğu korunur.
Kaynak hiç alınmadığında gerçekte verilmemiş bir BUY/STOP kararı üretilmez.
Başka issue'nun kaynak tüketimi gerçek malzeme geçmişinde görülür; açık bir
tahsis kararı olmadan otomatik olarak o issue'nun suçu ilan edilmez.

## Ayrı tutulan muhasebe

Ekonomik DAG `issue-sale-dag-v2`: her sink'in net altını aksiyon yolundaki
manager atalarına (BUY / ISSUE / SELL) dağıtır. Kapanış DAG'ı ikinci bir satın
alma kaybı veya sahte gelir oluşturmaz. Terminal hedef de altındır.

`local_credit`: yalnız gerçek ekonomik kredi.
`task_credit`: kapanış DAG'ından manager kararlarına dağıtılmış görev maliyeti (eksi; nakit değil).
`throughput_credit`: kota birimi cinsinden ISSUE kredisidir, nakit değildir.
`+ fulfill_reward × fulfilled` ve `− unmet_reward × unmet`. Varsayılan her ikisi 1.
Crew teslim/yem/satış kotayı karşılayınca ISSUE artı alır; boş talep eksi alır.
Bu, crew'un iş bitirmesini manager'ın kaldırılabilir yük öğrenmesine bağlar.
Bekleme/bakım cezası ayrıca durur; tam kotada yine sıfırlanır.

Satış DAG critic'i `local_credit + task_credit` öğrenir. Throughput ayrı critic
ve ayrı PPO terimidir (`--throughput-policy-coef`). Crew teslimat ödülü değişmez.

`--fulfill-reward` / `--unmet-reward` 0 ise bu sinyal kapanır. Resume config'inde
saklanır. Checkpoint: manager `feudal-manager-issue-gold-dag-v7`,
ortak eğitim `feudal-joint-issue-gold-dag-v9`. Eski v4/v5/v6/v8 resume edilmez.

`issue_task_cost.json` ham maliyeti, miktar raporunu, kanıtları, DAG düğümlerini,
kapanış sink'lerini ve karar paylarını içerir. Aynı grafiğe
`issue_production_graph.json.task_outcome_graph` üzerinden de ulaşılır.
Gerçek material graph lot ID'leri kanıt kayıtlarıyla bağlanır; parasal kenarlar
ile ceza kenarları birbirine karıştırılmaz.

## Sınırlar

Bu sistem gerçekleşen üretimi ve kaynakların gözlenen zamanlamasını denetler.
"Optimal rota/başka kararlarla ne üretilebilirdi?" hesabı yapmaz. Hazır ama
hasat edilmemiş ürünü kâr saymaz. Kısmi üretimi silmez, ancak teslim/kullanım
hedefini de yalnız hasat edilmiş olmakla tamamlanmış göstermez.
