# Scripted rakibe karşı PPO eğitimi

> **Güncel eğitim sözleşmesi: [TRAINING_V5.md](TRAINING_V5.md).**
> Aşağıdaki metin eski shaping/token-credit koşularının tarihsel dokümanıdır;
> varsayılanlar ve eski komutlar v5 için geçerli değildir. Yeni eğitimde
> terminal actor=1, yardımcı lifecycle actor=0 ve shaping=0 kullanılır.

Komutları proje klasöründe çalıştır:

```bash
cd "/home/doruk_sarigun/Çalışmalar/Kaggricalture"
source .venv/bin/activate
```

## Kredi ataması varsayılanları

Terminal hedefi (rakibe karşı gold farkı) 719 adımda tek bir skalerdir. Tek
başına kullanıldığında bir episode'un tüm adımları neredeyse aynı avantajı alır
ve eğitim episode-seviyesinde REINFORCE'a dönüşür. Varsayılanlar bunu üç yerden
kırar:

- `--shaping-beta 0.25`: ekonomik potential shaping. Ekim, ihmal ve sezon
  sonunda olgunlaşamayacak tohum alımı gibi kararlar gerçekleştikleri adımda
  sinyal alır. Terminalde `Phi=0` olduğu için episode boyunca teleskoplanır ve
  optimal politikayı değiştirmez. `0` tamamen kapatır.
- `--gae-lambda 0.95`: `1.0` terminal ağırlıklı hedefte adım ayrımını yok eder.
- `--episodes-per-update 16`: bir episode yaklaşık tek bir bağımsız örnektir;
  düşük değerler gradyan varyansını çok yükseltir.

Update sağlığını log satırındaki `mb=UYGULANAN/TOPLAM ep=TAMAMLANAN/HEDEF`
alanından izleyin. Tek bir aykırı minibatch KL eşiğini aşarsa yalnızca o adım
atlanır; update ancak epoch **ortalaması** `--target-kl`'i aşarsa durur
(`--kl-skip-multiplier` atlama eşiğini belirler). `mb` oranı sürekli düşüyorsa
öğrenme oranını düşürün — eskiden bu durumda update'ler tamamen iptal oluyor ve
model sessizce donuyordu.

`--normalize-token-advantages` (varsayılan açık) lifecycle token kredilerini
batch içinde merkezler ve ana avantajlarla aynı ölçeğe taşır. Kapatılırsa kredi
"kâr pozitif miydi" sorusunu yanıtlar (baseline yok) ve
`--lifecycle-policy-coef`'in efektif ağırlığı kredilerin büyüklüğüyle sürüklenir.

## Kısa kontrol koşusu

Bu komut 24 adımlı iki oyun oynar ve tüm kayıt türlerini test eder:

```bash
python train_scripted.py train \
  --opponent starter \
  --episodes 2 \
  --episodes-per-update 1 \
  --episode-steps 24 \
  --ppo-epochs 1 \
  --minibatch-size 32 \
  --checkpoint-every 1 \
  --save-replay-every 1 \
  --save-trace-every 1 \
  --run-dir runs/smoke_starter
```

## 720 hamlelik ilk eğitim

```bash
python train_scripted.py train \
  --opponent starter \
  --episodes 1000 \
  --episodes-per-update 16 \
  --episode-steps 720 \
  --ppo-epochs 3 \
  --minibatch-size 256 \
  --learning-rate 3e-4 \
  --gamma 0.999 \
  --gae-lambda 0.95 \
  --shaping-beta 0.25 \
  --entropy-coef 0.01 \
  --checkpoint-every 25 \
  --save-replay-every 10 \
  --save-trace-every 10 \
  --log-step-every 100 \
  --run-dir runs/starter_v1
```

Her oyunun tam state/action kaydı istenirse `--save-trace-every 1`, her oyunun
izlenebilir replay'i istenirse `--save-replay-every 1` kullan. 720 adımlı uzun
koşularda bunlar ciddi disk alanı kullanabilir.

## Eğitime kaldığı yerden devam etme

`--episodes` toplam hedef episode sayısıdır. Diğer eğitim ayarlarını ilk
koşuyla aynı tut:

```bash
python train_scripted.py train \
  --resume \
  --opponent starter \
  --episodes 2000 \
  --episodes-per-update 16 \
  --episode-steps 720 \
  --ppo-epochs 3 \
  --minibatch-size 256 \
  --learning-rate 3e-4 \
  --gamma 0.999 \
  --gae-lambda 0.95 \
  --shaping-beta 0.25 \
  --entropy-coef 0.01 \
  --checkpoint-every 25 \
  --save-replay-every 10 \
  --save-trace-every 10 \
  --run-dir runs/starter_v1
```

Model, optimizer ve Python/NumPy/PyTorch/aksiyon-sampling RNG durumları
`checkpoints/latest.pt` içinden geri yüklenir.

`SpatialEncoder` tek katmanlı 10x10 conv yerine 3x3 residual yığın kullandığı
için checkpoint formatı v2'ye yükseltildi. v1 checkpoint'leri (`runs/` altındaki
eski koşular) yüklenemez; bunlar için yeni bir `--run-dir` ile sıfırdan eğitim
başlatın.

## Checkpoint değerlendirme

Evaluation varsayılan olarak deterministik aksiyon seçer:

```bash
python train_scripted.py evaluate \
  --checkpoint runs/starter_v1/checkpoints/latest.pt \
  --opponent starter \
  --episodes 20 \
  --episode-steps 720 \
  --save-replay-every 1 \
  --save-trace-every 1 \
  --output-dir runs/starter_v1/evaluations/eval_001
```

Stokastik evaluation için `--stochastic` eklenebilir.

## Oyunu sonradan izleme

Bir episode replay'ini bağımsız HTML dosyasına çevir:

```bash
python train_scripted.py render \
  runs/starter_v1/episodes/episode_000010_seed_2035_p1/replay.json.gz \
  --output runs/starter_v1/episode_10.html \
  --open
```

`--open` verilmezse yalnızca HTML yazılır. Episode klasörü de doğrudan
girdi olarak verilebilir:

```bash
python train_scripted.py render \
  runs/starter_v1/episodes/episode_000010_seed_2035_p1
```

## Log ve artifact yapısı

```text
runs/starter_v1/
├── config.json
├── train.log
├── metrics.jsonl
├── events.jsonl
├── checkpoints/
│   ├── latest.pt
│   └── episode_......pt
└── episodes/
    └── episode_......_seed_..._p./
        ├── summary.json
        ├── complete.json
        ├── replay.json.gz
        ├── trajectory.jsonl.gz
        ├── lifecycle_events.jsonl.gz
        ├── lifecycle_summary.json
        ├── lifecycle_token_credits.json
        └── plant_lifecycles.json
```

`trajectory.jsonl.gz` her oyun adımında şunları saklar:

- ham observation ve next observation,
- bizim ve scripted rakibin aksiyonu,
- üretilen token dizisi,
- her token için head, geçerli token maskesi ve log-probability,
- value tahmini, entropy ve reward bileşenleri,
- oyuncu durumları ve terminal bilgisi.

Lifecycle artifact'leri satın alınan sentetik seed lot'unu ekildiği tile,
sulama/gübreleme, hasat product lot'u ve FIFO ile atanan satışla bağlar.
Belirli bir crop'un zincirini görmek için:

```bash
python inspect_lifecycle.py runs/starter_v1/episodes/EPISODE --crop WHEAT --events
```

Yalnızca hasat edilen plantları görmek için:

```bash
python inspect_lifecycle.py runs/starter_v1/episodes/EPISODE --only-harvested
```

Sıkıştırılmış trace'i terminalden incelemek için:

```bash
gzip -cd runs/starter_v1/episodes/EPISODE/trajectory.jsonl.gz | less
```

## Deneysel lifecycle reward

Varsayılan `--lifecycle-credit-mode off` terminal gold farkı ve potential
shaping hedefini aynen korur. Hasat anındaki ürün×piyasa değerini ayrı bir ara
reward olarak denemek için:

```bash
.venv/bin/python train_scripted.py train \
  --lifecycle-credit-mode harvest_value \
  --lifecycle-credit-coef 0.1 \
  ...
```

Bu mod hedef fonksiyonunu değiştirir; bu nedenle yeni bir run directory ile
`off` baseline'a karşı karşılaştırılmalıdır. Episode `summary.json` dosyasında
`lifecycle_credit_sum`, her trace satırında ise `lifecycle_credit_assignments`
ve ilgili `event_id` bulunur. Bu ilk strateji krediyi yalnızca HARVEST
transition'ina verir; geçmişteki BUY_SEED/PLANT/WATER aksiyonlarına geriye
dağıtım yapmaz.

Token'a özel nedensel zincir eğitimi için:

```bash
.venv/bin/python train_scripted.py train \
  --lifecycle-credit-mode causal_chain \
  --lifecycle-credit-coef 1.0 \
  --lifecycle-policy-coef 1.0 \
  ...
```

Bu mod lifecycle credit'i global transition reward'ina eklemez. Episode sonunda
satın alma maliyeti ve aynı seed lot'tan gerçekleşen satış sonucu
`lifecycle_token_credits.json` dosyasına yazılır. PPO bunu ayrı token-level
clipped loss ile yalnızca ilgili BUY_SEED/PLANT/WATER/FERTILIZE/HARVEST/DROP/SELL
span'larına uygular. Kullanılmayan fazla seed satın alımı net negatif kalır.
SELL kredisi yalnızca allocation kaynağı izlenen bir plant olan ürünlere
verilir; `BUY_PRODUCT -> SELL` veya kaynağı bilinmeyen inventory lifecycle bonusu
alamaz.

`jq` kuruluysa tek bir adımı daha okunur görmek için:

```bash
gzip -cd runs/starter_v1/episodes/EPISODE/trajectory.jsonl.gz | head -n 1 | jq
```
