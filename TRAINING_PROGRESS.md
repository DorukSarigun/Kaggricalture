# Cash-progress ve gerçekleşmiş getiri öğrenimi

## Miktar grameri: checkpoint v3

Eski 1/2/5/10/ALL kısıtlaması kaldırıldı. Tüm geçerli pozitif tam sayılar
ondalık rakamlarla seçilir: 3 = `QTY:3 <END_QUANTITY>`,
137 = `QTY:1 QTY:3 QTY:7 <END_QUANTITY>`. Başta sıfır ve bütçe/kapasiteyi
aşan rakamlar maskelenir. Miktarı bitirmek de quantity head kararıdır;
tek geçerli seçenek olduğunda zorunlu token sayılır. Vocabulary 55 token,
quantity head 11 seçenek kullanır. Bu, satın alma, satış ve miktarlı unit
işlemlerinin tamamında geçerlidir. ALL yerine maksimum sayının rakamları yazılır.

Eski v2 checkpoint'leri doğrudan resume/initialize edilemez; token embedding
ve quantity head değişti. Yanlış ağırlık eşleşmesi yerine açık hata verilir.
Çalışan eski süreç otomatik değişmez. Aşağıdaki resume KL yönergesi yalnız
yeni v3 checkpoint'leri arasında geçerlidir. Eski ağırlıkların kısmi transferi
bu değişikliğe dahil değildir; yeni eğitim için boş run klasörü kullanın.

Bu deneysel mod eski `terminal` varsayılanını değiştirmez. Mevcut run üzerinde
reward değiştirerek resume yapılmaz; ayrı run gerekir. Uzun eğitim otomatik başlamaz.

## Ekonomik hedef ve ara ilerleme

`--reward-mode cash_progress --gamma 1`:

```
base_t = (bizim nakit değişimimiz - rakibin nakit değişimi) / reward_scale
Phi_t = nakit HARİÇ mevcut varlıkların tahmini değeri / reward_scale
shaping_t = beta * (Phi_next - Phi_t)
reward_t = base_t + shaping_t
Phi_terminal = 0
```

Alım maliyeti hemen negatif, gerçek satış geliri pozitif base'dir. Terminal para
farkı yeniden eklenmez. Shaping toplamı `-beta*Phi_initial` olur. Başlangıçta
varlık yoksa toplam reward tam olarak son para farkındaki değişimdir. Hediye
tohumla başlatılan oyunlarda başlangıç sabiti ayrıca loglanır.

Satış sırasında ürün potansiyeli düşer ve gerçek nakit gelir: tahmin ve gelir
iki kez tam ödüllendirilmez. Satılmayan varlıkların geçici kredisi terminalde
kapanır; tohumun gerçek maliyeti kalır. Beta=0.25 örnek deney katsayısıdır,
performansla doğrulanmış bir optimum değildir.

Potansiyel mevcut `reward.py` hesaplarını kullanır: tohumun yetişmeye vakti,
ürünün mevcut satış eğrisi ve bitkinin büyüme takvimi değerlendirilir.
Cash-progress'te henüz hasat edilmemiş bitki tahmini, sulanmadıysa 0.50,
bugün sulandıysa 0.75 ile çarpılır. Bunlar başarı olasılığı değil, açıkça
seçilmiş ilerleme katsayılarıdır. Tekrar WATER state'i değiştirmediği için
bonus vermez. Bitki kaybı potansiyeli düşürür. Büyüme ve fiyat değişimi de
potansiyeli etkileyebilir: shaping tek başına nedensel aksiyon kredisi değildir.

Hareket/taşıma için henüz mesafe potansiyeli veya sabit adım cezası yoktur.
Bu sürüm ürün zincirine eşit-pay token kredisi uygulamaz; FIFO kayıtları audit'tir.
Ortak yolların izlenmesi ayrı tasarım gerektirir. Ekonomik tahminler gelecekteki
bakım, kapasite ve rakibin fiyat etkilerini kesin olarak öngörmez.

## Gerçekleşmiş getiriyi öğrenen critic

`--realized-value-coef 0.5` tamamlanmış episode'dan iskonto edilmiş gerçek
`base` reward-to-go hedefini hesaplar. Tahmini varlık kârı bu etikete girmez.

```
G_realized(t) = sum_k gamma**k * base_(t+k)
prediction_realized(t) = V_shaped(s_t) + beta * Phi(s_t)
auxiliary_loss = 0.5 * mean((prediction_realized - G_realized)**2)
total_loss += realized_value_coef * auxiliary_loss
```

Ayrı ağ/head yerine mevcut critic kullanılır: terminal potansiyel sıfırken
bu dönüşüm tam getiri eşitliğinden gelir. Shaped critic hâlâ GAE/value loss ile
eğitilir, ek Monte Carlo loss onu gerçekten gerçekleşen sonuçla kalibre eder.
Gelecek yalnız eğitim etiketidir; policy girdisine veya action seçimine sızmaz.
Yarım episode hedefi reddedilir. Bu yardımcı hedef örnek başına gürültülü
olabilir; başarının veya nedensel credit assignment'ın garantisi değildir.
Lambda=1 olduğunda aynı tam getiriye ikinci value-loss ağırlığı eklenmiş olur;
ayrı ve bağımsız bir ekonomik bilgi kaynağı değildir.

Kayıtlar: `metrics.jsonl` içinde `realized_value_loss`; episode summary içinde
`realized_return_mae` ve `discounted_shaping_conservation_error`; episode
klasöründe adım bazlı `realized_value_targets.json`; trace'te gerçek nakit,
potansiyel ve tahmin ayrıdır. Checkpoint mevcut critic/optimizer parametrelerini
ve yeni config alanlarını saklar; model boyutu değişmez.

## Deney komutu (otomatik çalıştırılmaz)

KL hedefinin yeni varsayılanı 0.020; skip çarpanı 2.0 ile minibatch atlama
eşiği 0.040. Gradient clipping hâlâ 1.0; `gradient_norm` kırpma ÖNCESİ normdur.
`approximate_kl` yalnız uygulanan minibatch'leri özetler;
`last_epoch_attempted_kl` atlananlar dahil son epoch ortalamasıdır ve erken
durdurma kararını açıklar. KL sınırını genişletmek daha iyi öğrenme garantisi değildir.

Çalışan süreç config dosyasını tekrar okumaz. Eski run'a yeni KL ile devam
etmek için süreç durduktan sonra aynı eğitim argümanlarına
`--resume --override-kl-on-resume --target-kl 0.020` eklenir. Bu yalnız KL
değişimine izin verir; optimizer/RNG korunur ve değişiklik `events.jsonl`'a
yazılır. Eski `config.json` başlangıç kaydıdır; yeni checkpoint güncel ayarı
taşır. Flag olmadan eski run'ı aynen sürdürmek için eski `--target-kl 0.015`
açıkça verilmelidir. Aynı run üzerinde ikinci bir süreç başlatılmamalıdır.

```bash
.venv/bin/python train_scripted.py train \
  --run-dir runs/cash_progress_trial \
  --opponent starter --episodes 200 --episodes-per-update 16 \
  --reward-mode cash_progress --gamma 1 --gae-lambda 0.95 \
  --shaping-beta 0.25 --realized-value-coef 0.5 \
  --global-actor-coef 1 --span-policy-coef 0 \
  --curriculum --curriculum-starting-wheat 0 \
  --save-replay-every 10 --save-trace-every 10 --device cpu
```

Önce aynı seed'lerle `realized-value-coef=0` karşılaştırması, ardından beta=0
karşılaştırması yapılmalıdır. Galibiyet, son nakit, gerçek hasat/satış ve
unused-seed maliyeti ayrı değerlendirilmelidir; loss düşmesi yeterli değildir.
