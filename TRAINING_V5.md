# Terminal hedefli eğitim — v5

## Hedef ve kapsam

Global reward yalnız terminalde `(own_gold - opponent_gold) / starting_money`.
Bu skor, mevcut global critic/GAE üzerinden bütün stochastic action span'larına
ulaşır: farmer, her hand, her market emri ve market STOP. Deterministik grammar
tokenları gradyan hedefi değildir. Global yol rol dengeli factorized PPO
surrogate'ıdır; joint-action PPO ile matematiksel olarak aynı olduğu iddia edilmez.

Varsayılan `--global-actor-coef 1 --span-policy-coef 0 --shaping-beta 0`.
Lifecycle/FIFO kayıtları korunur, fakat varsayılan actor loss'a girmez.
`--span-policy-coef` pozitif verilirse eksik kapsamdaki yerel yardımcı hedef
bilinçli bir ablation olarak açılır; kazanmayı iyileştirme garantisi yoktur.

HIRE/BUY_LAND gibi yatırımlar yalnız anlık maliyetle yargılanmaz: sonraki
terminal getiri onların kararlarına da ulaşır. Economic-span collector ve PPO
update, global actor katsayısı sıfır/negatif/sonlu değilse başlamaz. Bu güvenlik
kontrolü bir performans yeterlilik testi değildir. Müfredat hâlâ episode
sayısına göre ilerler; güçlü bir politika öğrenildiği otomatik varsayılmaz.

## Kasa defteri, lifecycle değildir

Her episode'da `cash_ledger.json` iki tarafın bütün gerçekleşen market emirlerini
kaydeder: SELL, BUY_SEED, BUY_PRODUCT, BUY_ANIMAL, HIRE, BUY_LAND.
Unit fazı ardından marketin lockstep sırası kopyalar üzerinde replay edilir.
Her tur gerçek `after_money - before_money`, işaretli execution toplamıyla
karşılaştırılır. 1e-7 üzerindeki fark hatadır; gizli residual geliri üretilmez.
Bu defter reward eklemez ve observation'a rakibin private bilgisini sızdırmaz.

Episode summary'de `cash_accounting`, `opponent_cash_accounting` ve
`cash_minus_lifecycle_credit` vardır. Sonuncusunun sıfır olması beklenmez:
lifecycle defteri bütün ekonomiyi kapsamıyor ve ayrıca katsayıyla ölçeklenebilir.
Kasa mutabakatı ile lifecycle kredi korunumu farklı invariant'lardır.

## Kontrol ve yeni eğitim

```bash
.venv/bin/python -m unittest discover -s tests -q

.venv/bin/python train_scripted.py train \
  --run-dir runs/terminal_v5 --opponent starter \
  --episodes 10000 --episodes-per-update 16 \
  --global-actor-coef 1 --span-policy-coef 0 --shaping-beta 0 \
  --curriculum --curriculum-starting-wheat 0 \
  --save-replay-every 100 --save-trace-every 100
```

Bu komut otomatik başlatılmamıştır. Curriculum sonuçları tam oyun benchmark'ı
değildir. Eski v4 optimizer/RMS durumu v5'e resume edilemez. Yalnız mimarisi
uyumlu ağırlıkları taşımak için yeni bir run dizininde
`--initialize-from-checkpoint PATH` kullanılabilir; kötü checkpoint'i taşımak
öğrenmeyi bozabilir, varsayılan öneri değildir.

## Aynı koşullarda checkpoint karşılaştırması

```bash
.venv/bin/python compare_checkpoints.py \
  --checkpoints \
    runs/span_v1/checkpoints/episode_002000_update_000125.pt \
    runs/span_v1/checkpoints/episode_004000_update_000250.pt \
    runs/span_v1/checkpoints/latest.pt \
  --seeds 31001 31002 31003 31004 31005 \
  --opponent starter --output-dir runs/paired_v5_evaluation
```

Her seed iki oyuncu tarafında da, 720-step konfigürasyonla oynanır. Başlangıç
tohum hediyesi ve curriculum maskesi yoktur; yalnız yasal aksiyon maskesi vardır.
Her checkpoint/case başlangıcında Python, NumPy, Torch ve policy RNG sıfırlanır.
Varsayılan argmax değerlendirmedir; `--stochastic` ayrı bir dizinde denenebilir.
Policy rakibin farklı yanıtlarını üretebileceğinden aynı seed, sonraki market
fiyatlarının bütün checkpoint'lerde aynı kalacağını garanti etmez.

`manifest.json` seed/side, model ve kaynak hash'leri, motor hash'i, inference
modu ve ortam ayarlarını içerir. `comparison.json` checkpoint özetlerini ve ilk
checkpoint'e karşı seed-blok eşleştirilmiş para farklarını verir. Her oyun için
replay, kasa defteri ve lifecycle/action logları saklanır. Mevcut çıktı dizini
üzerine yazılmaz. Az seed ile sonuçlar yalnız tanı amaçlıdır; leaderboard veya
genel üstünlük iddiası değildir. `evaluate_network.py` içindeki tarihsel sabit
coverage listesi bu davranış benchmark'ının yerine kullanılamaz.

## Regresyon testlerinin garantisi

- Her stochastic token tam bir action span'ında, STOP dahil.
- Yerel reward/entropy/value etkisi sıfırken terminal avantaj işareti,
  STOP/HIRE/BUY_LAND seçiminin policy gradyan yönünü değiştirir.
- Motor içinde gelirsiz HIRE ve BUY_LAND terminal gold farkını düşürür.
- Bütün market işlem türleri iki taraflı gerçek bakiye ile mutabıktır;
  başarısız emir maliyet üretmez; eksik trace hatadır.
- Yerel krediler loglansa da katsayı sıfırken actor loss değişmez.
- Karşılaştırma iki tarafı ve aynı seed kümesini zorunlu tutar.

Bunlar öğrenme başarısının kanıtı değil, eğitim sözleşmesinin regresyon testleridir.
Navigation kredisi bu sürüme dahil değildir.
