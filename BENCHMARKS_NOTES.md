# graykode/gpt-2-Pytorch — Notes de benchmark

**Statut : OK — 10/10 pass, aucun écart. GPT-2 non modifié (172K params, 2 couches) : forward LM + presents (cache KV) + tied embeddings + CrossEntropy + backward (28/28) + AdamW.**

Un point central : **`mlx.core.compile` en mode lazy / compilé** attend de connaître toute la séquence d'opérations avant de lancer les calculs. C'est particulièrement important pour les **opérations de batching** : au lieu d'exécuter chaque petite opération GPU séparément (avec son overhead de dispatch/lancement à chaque itération), le mode compilé lazy construit d'abord le graphe d'opérations de tout le batch, le fusionne en kernels optimisés, puis l'exécute d'un seul coup. Pour un batch de N échantillons, l'overhead est amorti une seule fois au lieu de N fois — d'où des gains typiques de plusieurs fois (jusqu'à ~15×) dès que le travail par étape est suffisant.

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
