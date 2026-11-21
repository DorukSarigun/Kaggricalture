# Tek ürün eğitim path'leri

Manager ISSUE ve üretim marketi bir `TASK_ITEMS` kümesine kısıtlanır.
Ağ boyutu değişmez: sekiz ürün logit'i durur, maske diğerlerini kapatır.
`--issue-items` yoksa eski sekiz ürünlük oyun aynıdır.

`feudal-manager-issue-gold-dag-v7` mimarisi. Dolu run dizinlerine yazılmaz.
Bu dosya komut listesidir; eğitim burada başlatılmaz.

## Maske

| Path | ISSUE | BUY_SEED | BUY_ANIMAL | BUY_PRODUCT | SELL |
|---|---|---|---|---|---|
| WHEAT, CARROT, TOMATO, STRAWBERRY, MELON | o ürün | o ürün | — | — | o ürün |
| GOOSE / COW / SHEEP | o hayvan | — | o hayvan | WHEAT (yem) | EGG/MILK/WOOL + FERTILIZER |

Hayvan path'inde buğday tohumu alınamaz ve buğday satılamaz; yem için yalnız `BUY_PRODUCT WHEAT` açıktır. HIRE ve BUY_LAND açık kalır. Crew scripted `tour`+`greedy` yalnız açılan ISSUE'ları yürütür.

Resume kilidi: `config.issue_items` (`null` = kısıtsız). Carrot path, kısıtsız v7 checkpoint'inden `--resume` alamaz; sıfırdan veya `--init-manager` ile **yeni boş** dizine ağırlık kopyalanır.

## Manager komutları (sonradan)

Her ürün ayrı boş dizin. Rakip BestAgent; crew scripted. İsteğe bağlı: `--init-manager runs/feudal_manager_issue_gold_dag_v7_01/latest.pt`.

```bash
ITEM=CARROT   # WHEAT CARROT TOMATO STRAWBERRY MELON GOOSE COW SHEEP
SLUG=$(echo "$ITEM" | tr '[:upper:]' '[:lower:]')

.venv/bin/python -m worker_grow.train_manager \
  --run-dir "runs/feudal_manager_path_${SLUG}_v7_01" \
  --issue-items "$ITEM" \
  --episodes 400 --episode-steps 720 --episodes-per-update 4 \
  --epochs 3 --minibatch-size 64 --learning-rate 3e-5 \
  --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 \
  --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 \
  --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 \
  --crew scripted --assignment-strategy tour --walk-policy greedy \
  --opponent best_agent --device cuda --seed 2026
```

Sekiz ayrı kopya-yapıştır:

```bash
.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_wheat_v7_01 --issue-items WHEAT --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2026

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_carrot_v7_01 --issue-items CARROT --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2027

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_tomato_v7_01 --issue-items TOMATO --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2028

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_strawberry_v7_01 --issue-items STRAWBERRY --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2029

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_melon_v7_01 --issue-items MELON --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2030

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_goose_v7_01 --issue-items GOOSE --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2031

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_cow_v7_01 --issue-items COW --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2032

.venv/bin/python -m worker_grow.train_manager --run-dir runs/feudal_manager_path_sheep_v7_01 --issue-items SHEEP --episodes 400 --episode-steps 720 --episodes-per-update 4 --epochs 3 --minibatch-size 64 --learning-rate 3e-5 --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 --max-grad-norm 1.0 --issue-wait-cost 0.0001 --missed-care-cost 0.001 --save-every 10 --crew scripted --assignment-strategy tour --walk-policy greedy --opponent best_agent --device cuda --seed 2033
```

Aynı flag `train_joint` üzerindedir. Sihirbaz: `worker_grow/run_product_paths.sh CARROT`.
