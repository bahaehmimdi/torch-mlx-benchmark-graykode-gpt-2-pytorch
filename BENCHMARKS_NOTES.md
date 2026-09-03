
> ## ⚠️ Correction
>
> Ce dépôt contient des **gains « compilés » ~10×–15× qui ne sont pas reproductibles** et ont été retirés. Sur une baseline MPS propre et avec `mlx.compile` réellement foré à s'exécuter (entrées fraîches + `mx.eval`), **torch-mlx est en parité avec PyTorch MPS** et `mlx.compile` est une **régression** sur la couche torch-mlx. Voir `bench/README.md` du dépôt torch-mlx et `scripts/bench_status.tsv`.

# graykode/gpt-2-Pytorch — Notes de benchmark

**Statut : OK — 10/10 pass, aucun écart. GPT-2 non modifié (172K params, 2 couches) : forward LM + presents (cache KV) + tied embeddings + CrossEntropy + backward (28/28) + AdamW.**

**`mlx.core.compile`** (mode lazy / compilé) ne fusionne les opérations qu'au niveau du graphe MLX natif. Sur la couche d'adaptation torch-mlx, rappelé via `Function.apply`, le compilateur voit des fonctions opaques : la compilation est mesurée comme une **régression** (~1,5× à ~150× plus lente que l'eager MLX), pas une accélération. Les « gains compilés » parfois publiés provenaient de la constante-folding (entrées identiques à chaque itération, graphe lazy jamais forcé).

## Gaps de compatibilité
graykode/gpt-2-Pytorch (~1K stars) : implémentation simple de GPT-2 (poids liés / tied embeddings).

Test : `GPT2/model.py` + `config.py` chargés byte-for-byte (imports : torch, torch.nn, torch.nn.functional, torch.nn.parameter uniquement). Résultats (10/10 PASS, aucun écart) :
  - GPT2LMHeadModel (n_layer=2, n_embd=64, n_head=4, ~172K params) : forward OK
  - forward LM (batch=2, seq=16) -> (2,16,1000) : shape + fini
  - presents (cache KV) renvoyé pour chaque couche
  - tied embeddings (`wte <-> decoder.weight` partagés) : OK
  - perte CrossEntropy (avec labels) : finie
  - backward : 28/28 gradients
  - AdamW step + perte post-step finie

Notable : `x.split(...)` (découpage multi-têtes dans Attention) et `F.gelu`/`F.softmax` fonctionnent directement — l'ancien écart `Tensor.split` (round 350, ultralytics) ne se manifeste pas ici. Architecture transformeur = workload GEMM où torch-mlx compilé excelle (ex. nanoGPT ~2x vs CPU).

## Références
- Dépôt source torch-mlx : https://github.com/bahaehmimdi/torch-mlx
- Discussion générale : https://github.com/bahaehmimdi/torch-mlx-benchmarks-output/discussions/1
