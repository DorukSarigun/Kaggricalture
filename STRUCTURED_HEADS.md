# İşlem head'leri ve işlem başına advantage

Bu yeni yol `train_structured.py` ile çalışır. Eski `train_scripted.py`,
token modeli ve eski run'lar değiştirilmez. Checkpoint formatı
`structured-operation-v1`; token checkpoint'i doğrudan yüklenemez.

## Ağ

- Paylaşımlı StateEncoder kendi/rakip tarla ve globalleri kodlar.
- Her tam işlemden önce shadow observation yeniden encode edilir.
- Girdi hem turun orijinal observation'ını hem güncel shadow'yu içerir.
- Geçerli unit'in kendi konumu, taşıdığı ürünler, işlem fazı, market emir
  indeksi ve hires_today doğrudan bağlama eklenir.
- Tamamlanan işlemler sırayla prefix GRU'dan geçer. Bu GRU token veya aksiyon
  üretmez; aynı shadow sonucu veren farklı emir sıralarını bağlama taşır.
- Unit operation head / market operation head -> koşullu item head -> koşullu
  quantity head. BOS/EOS, END_ORDER ve rakam tokenları yoktur.
- Quantity head seçilen operation ve item'a koşulludur; 1..geçerli limit
  arasındaki bütün tam sayılara skor verir. Sabit bucket veya ALL kısıtı yoktur.
  Aday özellikleri normalize miktar, karesi, karekökü ve log miktarıdır.
  Büyük limitte parametre sayısı sabit, aday hesap/bellek maliyeti O(limit)'tir.
  Miktar dağılımının şekli bu düşük boyutlu skor ailesiyle sınırlıdır.
- İşlem öncesi shaped value ve ayrı realized-return value head vardır.

`ActionGrammar` burada yalnız yasal operasyon kuralları ve motorun shadow
işlemleri için adapter olarak kullanılır. Token `push`, tokenizasyon ve eski
decoder bu yolda çağrılmaz. Maske kuralları bağımsız kopyalanıp çatallaşmaz.
Kural adapter'ı eski modülün private yardımcılarına bağlıdır; testler bu sınırı
korur. Bu bağımlılığı sıfırlamak istenirse ortak legality modülü ayrı refactor'dür.

## Gerçek ortam ile shadow arasındaki fark

Oyun iki oyuncunun bütün emirlerini birlikte işler. Kendi market prefix'inin
shadow fiyatı gerçek satış fiyatı değildir. Rakip aksiyonu veya private verisi
policy/critic girdisine verilmez. Tam emir paketinden sonra gerçek env.step
çağrılır; iki oyuncunun tüm nakit execution'ları CashLedger ile ayrıca doğrulanır.

İç işlem ödülü = kendi shadow geçişinin cash-progress tahmini.
Tur-sonu settlement ödülü = gerçek tur reward'ı - iç ödüllerin toplamı.

Her tur toplamı tam olarak gerçek reward'a eşittir. Sezon başı hediye varlık
olmadığında gamma=1 ile bütün episode toplamı `(son para - rakip son para)/3000`.
Settlement hiçbir aksiyon seçmeyen ayrı bir value geçişidir: rakip, piyasa,
büyüme, gece ve shadow hatası burada uzlaşır. STOP seçimi varsa kendi span'idir;
zorunlu market üst sınırı ayrı bir stochastic STOP üretmez.

Bu iç ödüller nedensel gerçekleşmiş aksiyon kârı değildir; deterministik bir
tahminle ödülün karar adımlarına yeniden dağıtımıdır. Toplamın korunması tek
tek dağıtımın kaliteli olduğunu kanıtlamaz. Maskeler kendi shadow prefix'ine
göredir; rakibin market etkisi nedeniyle gerçek miktar kısmen uygulanabilir.

## Advantage

Her işlem için kendi V, r, A ve log_probability bulunur. Bir işlemin
`logp = logp(operation) + logp(item|operation) + logp(quantity|operation,item)`.
PPO oranı işlem bazında hesaplanır; aynı turdaki bütün işlemlere tek A kopyalanmaz.
Bir işlemin kendi argümanları hâlâ aynı işlem advantage'ını paylaşır.

İç işlem geçişlerinde gamma=1 ve lambda=1. GAE lambda yalnız fiziksel tur-sonu
geçişinde uygulanır. Daha çok işlem seçmek, kendi başına daha çok iskonto
üretmez. Tam episode gereklidir. Settlement value loss'a girer, actor loss'a
girmez. Forced işlemler de actor/entropy loss'undan çıkarılır.

Ayrı realized_value_head yalnız gerçek tur base reward-to-go etiketiyle
eğitilir; shadow'daki tahmini satışlar etiket değildir. Mevcut turdaki gerçekleşmiş
base sonuçları settlement satırında kaydedilir. Sonraki satırlara gerçek
gelecek observation aktarılmaz; gelecek yalnız supervised label'dır.

İşlem-MDP PPO eski span PPO'dan ayrıdır: işlem oranı clip=0.2, global gradient
norm clip=1.0, target KL=0.02 ve skip threshold=0.04. KL hâlâ önemlidir;
head değişimi aşırı policy update problemini otomatik çözmez. Value loss bu
ilk sürümde MSE'dir, eski value-clipping kullanılmaz.

## Çalıştırma

Önce kısa kontrollü deney:

```bash
.venv/bin/python train_structured.py \
  --run-dir runs/structured_trial \
  --device cuda --opponent starter \
  --episodes 100 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --learning-rate 0.00003 \
  --shaping-beta 0.25 --realized-value-coef 0.5 \
  --gae-lambda 0.95 --target-kl 0.02 --save-every 10
```

Bu ayrı CLI'da curriculum ve self-play henüz bağlı değildir; bütün yasal
aksiyonlar açıktır. Varsayılan 720 kayıt/719 ortam geçişidir. Gamma daima 1.
Oyunlar seri toplanır, PPO batch kullanır; paralel rollout/AMP hızlandırması yoktur.
Her işlemde yeniden encoding yapıldığı için hızlanma garantisi verilmez.

`--resume` aynı model/optimizer/RNG ile latest checkpoint'ten devam eder.
Yalnız episodes ve save-every değiştirilebilir. Checkpoint sonrası kısmi episode
kayıtları varsa sessizce ezmek yerine durur. Aynı run'a iki süreç başlatmayın.

Her episode: summary, turn_rewards, cash_ledger, advantages ve FIFO/lifecycle
audit dosyaları. advantages.json: step + unit/order index + işlem + reward +
value + ayrı advantage + gerçek getiri hedefi. `--save-every N`: her N oyunda
replay.json.gz ve tam observation/ara encoded state içeren trace.jsonl.gz.
Trace büyük olabilir. FIFO audit korunur ama token_span yoktur; PPO işlemleri
step/section/unit_index/market_order_index ile eşleştirir.

## Sınırlar

Bu nedensel kredi için bir garanti değildir; doğru zaman ölçeğinde ayrı
işlem geçişleri ve ayrı beklentiler kurar. Hareket verimliliği için özel
mesafe shaping yoktur. Başarı ölçütleri gerçek hasat, satış, final nakit,
galibiyet, KL ve örnek kullanımıdır; yalnız loss düşmesi yeterli değildir.
