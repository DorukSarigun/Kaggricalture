# Crew ajanı — alınan kararlar

> **Güncel sürüm: [FEUDAL_CREW.md](FEUDAL_CREW.md).** Ana çiftçi artık crew'a
> dahildir. Yeni `FeudalCrew`, manager görevleriyle koşullanan tek recurrent
> ekip politikasıdır. Yeni eğitim işlem-başı fiyat/grow veya yürüme potansiyeli
> kullanmaz; gerçek ilk teslimat üzerinden görev grafiği kredisi kullanır.
> Giriş: `python -m worker_grow.train_crew`. Aşağıdaki metin önceki tasarımın
> arşividir; özellikle farmer sahipliği ve grow ödülü artık geçerli değildir.

Mevcut `train_realized_chains` / production-DAG eğitimine dokunulmaz. Crew ayrı pakette (`worker_grow/`). Aşağıdakiler konuşulup kilitlenen kararlardır; eski tek-politika replay’indeki X’li arazide dolaşma bu modele ait değildir (o ajan henüz eğitilmedi).

## İki ajan, iki hedef

- **Manager:** farmer + market. Ekim planı, HIRE, arazi, al/sat. İşçinin karesinde bakım yapmaz; işi açar.
- **Crew:** o turdaki **tüm** `hands` listesini üreten **tek** politika. Her el ayrı ajan değildir; aynı ağırlıkla bağımsız N işçi de değildir.
- İki PPO terimi karışmaz. Crew kaybı manager nakit/terminalini görmez; manager, crew’un yetiştirme payını kendi ödülü saymaz.

## Crew ne kontrol eder

- Yalnız unit. `HIRE`, `BUY_*`, `SELL`, `BUY_LAND` yok.
- Motor sırası korunur: `hands[0] → hands[1] → …` gölge durumda. Birinci el ekerse ikincisi o kareyi dolu görür.
- Bir turda her el **tek** işlem. `SOUTH` o turu bitirir; aynı turda yürü + `WATER` yok. Hasat/sulama yeni karede, sonraki turda.
- HIRE edilen el aynı tur oynamaz; gece `hands` silinir.

## Gözlem

- CNN **yalnız kendi tarlası**. Rakip farm yok.
- Haritada bütün eller görünür; crew hepsini komuta eder.
- Tile kanallarında tür, yaş, `first_yield`, `yield_fill`, bugün sulandı/yemendi, ihmal streak, fiyat oranı, ömür, gübre, CARE vardır. Tek kanal `kalan_gün_hasat` yoktur; ağ yaş ile ilk hasat gününü birleştirir.
- 10×10 spatial yığın tahtayı kapsar. Bu, doğru yörüngeyi otomatik çizmez; “nerede / ne zaman iş var”ı okunur kılar.

## Crew’un yasal hamleleri (el başına, duruma göre)

Her zaman: `PASS`, tahta içi `N/S/E/W` (kilitli kareye yürümek motoren yasal; orada tarla işi no-op).

Shed kenarı: elde bir şey varsa `DROP`; depoda izinli eşya varsa `PICKUP`. `PLACE`: shed’e bırakma veya boş yapıya hayvan.

Boş kilitli-olmayan: tohum varsa `PLANT`; `BUILD_COOP` / `BUILD_PASTURE`.

Bitki: bugün sulanmadıysa `WATER`; olgun + `yield>0` ise `HARVEST`; elde gübre varsa `FERTILIZE`.

Hayvan: `yield>0` ise `HARVEST`; elde WHEAT ve bugün yemediyse `FEED`; gübre birikmişse `COLLECT_FERTILIZER`; bugün care yoksa `CARE`.

`DIG`: hayvanlı yapı hariç dolu kare. `PLANT` / `PICKUP` / `PLACE` item; `PICKUP` / `PLACE` miktar.

## Yetiştirme ödülü (her adım)

Market emri yok. Hamle yetiştirme değilse (`PASS`, yürüme, `DIG`) grow payı 0.

Yetiştirme hamlesinde, o hamlenin her ürünü için:

`güncel market fiyatı / o türün ilk döngü unit aksiyon sayısı`

Paylar toplanır, `/ 3000` ile ölçeklenir. BUY/SELL paydada yoktur.

İlk döngü = shed’e ilk satılabilir birim (market hariç): günlük WATER veya FEED (yaş 0 … `first_yield` dahil) + `PLANT`/`BUILD+PLACE` + `HARVEST` + `DROP`.

| Ürün | Payda |
|---|---|
| WHEAT, CARROT | 6 |
| TOMATO | 12 |
| STRAWBERRY, MELON | 14 |
| EGG | 9 |
| MILK | 13 |
| WOOL | 11 |
| FERTILIZER | 1 |

`BUILD_PASTURE` süt ve yünü ayrı böler. İsteğe bağlı `FERTILIZE` / `CARE` aynı payda ile yine grow sayılır.

Ödül **mürettebat adımına** yazılır (o turdaki tüm ellerin grow payları aynı crew kaybına), işçi-başı HIRE mahsuplaşması crew hedefi değildir.

## Yürüme ödülü (karar, henüz kodda değil)

Düz “ürüne yakın dur” yok: yanında bekleme ve eşmesafe daire ödül yer.

- Hedef kare: herhangi bir ürün değil, **şu an işi olan** kare (veya eldeki tohum → boş kare, ürün → shed, yem → aç hayvan).
- `Φ = −(hedefe Manhattan)`, `r_yürü = ε (Φ′ − Φ)`. Yaklaşma +, uzaklaşma −, yerinde 0.
- `ε`, bir grow payının altında (öneri: `1 / (10 × 3000)`).
- `PASS` 0. Kilitli kare: ek eksi veya crew maskesi ayrıca seçilir; potansiyel tek başına X’leri garanti kesmez.

## Bilinçli limitler

- CNN zamanlamayı gösterir; grow-only ödülle uzun yürüme yörüngesi pratikte zayıf kalır — yürüme potansiyeli bunun için.
- İki el aynı kareye aday olabilir; crew sıralı gölge ile ikincisi birincinin işini görür, rezervasyon yok.
- Pahalı uzak ürün vs ucuz yakın: fiyat kanalı var, tercih garanti değil.

## Kod durumu

Yazılı (`worker_grow/`): tarif, adım grow ödülü, yalnız-kendi-tarla encode, unit politikası iskeleti.

Yazılmadı: tek crew turu (tüm eller bir ajan), yürüme `Φ`, manager ayrımı eğitim döngüsü, kilitli-adım maskesi/cezası.
