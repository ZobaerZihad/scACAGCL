from time import perf_counter as t
import numpy as np
import torch
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from utils import edge2adj, EdgeDropping, GeneDropping, GraphAdversarialAttack


def _embedding_score(Z, n_clusters, loss1_val=None, loss1_ref=None,
                     pure_sil=False, sample=1024, n_runs=3):
    """
    Stratified-silhouette + adaptive hybrid quality score.

    pure_sil=True  -> raw silhouette on [-1,1].  Used only for the
                      pre-training difficulty measurement (pre_sil).
    pure_sil=False -> hybrid = 0.5*sil + 0.5*loss_signal.  Used for
                      the top-K pool on every dataset, because adversarial
                      training is always active (alpha never drops to zero),
                      so the loss signal is always informative.

    Stratified subsampling guarantees every predicted cluster is represented
    in the silhouette calculation (minimum 2 cells per cluster), solving the
    subsampling bias that hurts high-K datasets such as Worm_Neuron (K=37).
    """
    try:
        Z_np = Z.detach().cpu().numpy()
        Z_np = normalize(Z_np, norm='l2')
        n    = Z_np.shape[0]

        sil_scores = []
        for seed in range(n_runs):
            if n > sample:
                km_full     = MiniBatchKMeans(n_clusters=n_clusters, n_init=3,
                                              random_state=seed)
                full_labels = km_full.fit_predict(Z_np)
                rng         = np.random.RandomState(seed * 7)
                idx_list    = []
                for c in range(n_clusters):
                    c_idx  = np.where(full_labels == c)[0]
                    if len(c_idx) == 0:
                        continue
                    n_take = max(2, int(sample * len(c_idx) / n))
                    n_take = min(n_take, len(c_idx))
                    idx_list.append(rng.choice(c_idx, n_take, replace=False))
                if not idx_list:
                    continue
                Z_s = Z_np[np.concatenate(idx_list)]
            else:
                Z_s = Z_np

            km     = MiniBatchKMeans(n_clusters=n_clusters, n_init=5,
                                     random_state=seed)
            labels = km.fit_predict(Z_s)
            if len(np.unique(labels)) < 2:
                continue
            sil_scores.append(silhouette_score(Z_s, labels,
                                               sample_size=min(sample, len(Z_s))))

        sil = float(np.mean(sil_scores)) if sil_scores else -np.inf
        if not np.isfinite(sil):
            return -np.inf

        if pure_sil or loss1_val is None or loss1_ref is None:
            return sil

        scale       = max(loss1_ref * 0.15, 0.3)
        loss_signal = float(np.tanh((loss1_ref - loss1_val) / scale))
        return 0.5 * sil + 0.5 * loss_signal

    except Exception:
        return -np.inf


def Train(scAGCLmodel, data, cellGgraph, device, num_epochs, lam, alpha, beta, iters,
          optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size):

    start_time = t()
    scAGCLmodel.train()

    n_cells    = data.x.shape[0]
    n_clusters = int(data.y.max().item()) + 1

    ema_loss  = None
    ema_decay = 0.85

    # =========================================================================
    # PRE-TRAINING DIFFICULTY MEASUREMENT
    #
    # Compute the silhouette score of the randomly-initialised encoder before
    # any training begins.  This single value — pre_sil — quantifies how well
    # the untrained encoder already separates cell types.  It is the sole
    # signal driving the adaptive curriculum; no dataset names, no thresholds.
    #
    #   pre_sil ≈ 0  →  random noise embedding; needs maximum adversarial
    #                    pressure to escape poor local minima.
    #   pre_sil ≈ 1  →  near-perfect separation before training; strong
    #                    attacks would only disrupt existing structure.
    
    # =========================================================================
    scAGCLmodel.eval()
    with torch.no_grad():
        X_p   = data.x.to(device)
        Adj_p = edge2adj(data.x, data.edge_index).to(device)
        Z_p   = scAGCLmodel(X_p, Adj_p)
    scAGCLmodel.train()

    pre_sil = _embedding_score(Z_p, n_clusters, pure_sil=True)
    pre_sil = float(np.clip(pre_sil, 0.0, 1.0))   # guard: keep in [0, 1]

    # =========================================================================
    # ADAPTIVE CURRICULUM PARAMETERS  (Equations 1 & 2)
    #
    # Both parameters are continuous, monotone functions of pre_sil.
    # No thresholds. No dataset categories. One code path for all datasets.
    #
    # ── Equation 1: Adaptive warmup duration ─────────────────────────────────
    #
    #   WARMUP_EPOCHS = int( T × pre_sil × W_SCALE )       W_SCALE = 0.20
    #
    #   High pre_sil  → long warmup: contrastive objective consolidates the
    #                   already-good structure before adversarial pressure begins.
    #   Low pre_sil   → short warmup: adversarial perturbations needed early
    #                   to escape poor local minima.
    #
    #   Examples (T = 500):
    #     Petropoulos  pre_sil=0.03  →  WARMUP =  3 epochs
    #     Yan          pre_sil=0.05  →  WARMUP =  5 epochs
    #     Melanoma     pre_sil=0.35  →  WARMUP = 35 epochs
    #     Muraro       pre_sil=0.45  →  WARMUP = 45 epochs
    #     Limb_Muscle  pre_sil=0.82  →  WARMUP = 82 epochs
    #
    # ── Equation 2: Adaptive attack magnitude ────────────────────────────────
    #
    #   alpha_eff = alpha_0 × ( 1 − pre_sil × A_SCALE )    A_SCALE = 0.50
    #   beta_eff  = beta_0  × ( 1 − pre_sil × A_SCALE )
    #
    #   High pre_sil  → reduced alpha: tight clusters need gentler perturbations
    #                   that reinforce robustness without disrupting geometry.
    #   Low pre_sil   → full alpha: loose embeddings need strong PGD steps that
    #                   cross cluster boundaries and force better separation.
    #
    #   CRITICAL: even at pre_sil = 1.0, alpha_eff = alpha_0 × 0.50.
    #   The adversarial term is NEVER zero — loss2 always contributes to
    #   training, making this a genuine generalised adversarial method.
    #
    #   Examples (alpha_0 = 100):
    #     Petropoulos  pre_sil=0.03  →  alpha_eff = 98.5  (≈ full strength)
    #     Yan          pre_sil=0.05  →  alpha_eff = 97.5
    #     Melanoma     pre_sil=0.35  →  alpha_eff = 82.5  (17% gentler)
    #     Muraro       pre_sil=0.45  →  alpha_eff = 77.5  (22% gentler)
    #     Limb_Muscle  pre_sil=0.82  →  alpha_eff = 59.0  (41% gentler)
    # =========================================================================
    W_SCALE = 0.20   # warmup scale factor  (Equation 1 coefficient)
    A_SCALE = 0.50   # alpha attenuation    (Equation 2 coefficient)

    WARMUP_EPOCHS  = int(num_epochs * pre_sil * W_SCALE)
    alpha_adaptive = alpha * (1.0 - pre_sil * A_SCALE)
    beta_adaptive  = beta  * (1.0 - pre_sil * A_SCALE)

    # TOP_K=3: ensemble of the 3 highest-scoring checkpoints.
    # The convergence-aware selection at the end collapses to TOP_K=1
    # for monotonically-improving datasets (best epoch in final 15%).
    TOP_K = 3

    print(f'[INFO] pre_sil={pre_sil:.3f} | '
          f'WARMUP={WARMUP_EPOCHS} epochs | '
          f'alpha_eff={alpha_adaptive:.1f} (base={alpha}) | '
          f'beta_eff={beta_adaptive:.5f} (base={beta})')

    # =========================================================================
    # POOL START EPOCH — relative to warmup, not absolute
    #
    # POOL_START = WARMUP + 10% of post-warmup length.
    # Short stabilisation buffer after warmup ends, then the pool captures
    # the full remaining training trajectory.
    # =========================================================================
    post_warmup_len  = num_epochs - WARMUP_EPOCHS
    POOL_START_EPOCH = WARMUP_EPOCHS + max(1, int(post_warmup_len * 0.10))
    print(f'[INFO] Pool opens at epoch {POOL_START_EPOCH}')

    # Curriculum ramp: adversarial fraction rises from 0.15 to 1.0 over the
    # first 75% of post-warmup training.  The ramp schedule is the same for
    # every dataset; pre_sil scaling adjusts WHERE the ramp starts (warmup)
    # and HOW STRONG the ceiling is (alpha_adaptive / beta_adaptive).
    RAMP_END_FRAC = 0.75

    # Pool: each entry is (score, Z_tensor, epoch)
    top_scores       = []
    EVAL_EVERY       = 5

    LOSS_REF_EPOCHS  = 5
    loss1_ref_buffer = []
    loss1_ref        = None
    prev_loss1       = None

    Z_final = None

    for epoch in range(1, num_epochs + 2):

        # ── Full-graph embedding (no grad, eval mode) ─────────────────────────
        scAGCLmodel.eval()
        X   = data.x
        Adj = edge2adj(X, data.edge_index)
        X   = X.to(device)
        Adj = Adj.to(device)
        Z_current = scAGCLmodel(X, Adj)
        scAGCLmodel.train()

        if epoch == num_epochs + 1:
            Z_final = Z_current.detach().clone()

        # ── Top-K pool update ─────────────────────────────────────────────────
        if epoch % EVAL_EVERY == 0 and epoch >= POOL_START_EPOCH:
            score = _embedding_score(Z_current, n_clusters,
                                     loss1_val=prev_loss1,
                                     loss1_ref=loss1_ref)
            Z_det = Z_current.detach().clone()
            entry = (score, Z_det, epoch)
            if len(top_scores) < TOP_K:
                top_scores.append(entry)
                top_scores.sort(key=lambda x: x[0], reverse=True)
            elif score > top_scores[-1][0]:
                top_scores[-1] = entry
                top_scores.sort(key=lambda x: x[0], reverse=True)

        # ── Random subgraph ───────────────────────────────────────────────────
        n_nodes  = cellGgraph.number_of_nodes()
        sub_size = min(subgraph_size, n_nodes)
        subGraph = cellGgraph.subgraph(
            np.random.permutation(n_nodes)[:sub_size]
        )
        x_sub    = data.x[np.array(subGraph.nodes())].to(device)
        subGraph = nx.relabel.convert_node_labels_to_integers(
            subGraph, first_label=0, ordering='default'
        )
        edgeind = np.array(subGraph.edges()).T
        edgeind = torch.LongTensor(np.hstack([edgeind, edgeind[::-1]])).to(device)

        optimizer.zero_grad()

        x_1 = GeneDropping(x_sub, feature_r1)
        x_2 = GeneDropping(x_sub, feature_r2)

        edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
        edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
        adj_1 = edge2adj(x_1, edgeind_1)
        adj_2 = edge2adj(x_2, edgeind_2)

        z_1 = scAGCLmodel(x_1, adj_1)
        z_2 = scAGCLmodel(x_2, adj_2)
        loss1 = scAGCLmodel.loss(z_1, z_2, batch_size=0)

        loss2 = torch.tensor(0.0, device=device)

        # ── Adversarial block ─────────────────────────────────────────────────
        #
        # Active after WARMUP_EPOCHS ends for all datasets (lam > 0).
        # alpha_adaptive ensures loss2 is always non-trivially contributing —
        # alpha_adaptive >= alpha_0 * 0.5 regardless of pre_sil.
        #
        # The curriculum ramp (fraction 0.15 → 1.0) further modulates strength
        # within the post-warmup window.  Final effective alpha at any epoch:
        #
        #   alpha_final = alpha_adaptive × (0.15 + 0.85 × fraction)
        #               = alpha_0 × (1 − pre_sil/2) × (0.15 + 0.85 × fraction)
        if lam > 0 and epoch > WARMUP_EPOCHS:

            fraction = min(1.0,
                           (epoch - WARMUP_EPOCHS) / (post_warmup_len * RAMP_END_FRAC))

            current_node_ratio = 0.04 + (0.18 - 0.04) * fraction
            current_alpha      = alpha_adaptive * (0.15 + 0.85 * fraction)
            current_beta       = beta_adaptive  * (0.15 + 0.85 * fraction)
            min_iters          = 3
            current_iters      = max(min_iters, min(iters,
                                    int(min_iters + (iters - min_iters) * fraction)))

            adj_3, x_3 = GraphAdversarialAttack(
                scAGCLmodel, edgeind, edgeind_1, x_sub, x_1,
                current_iters, current_node_ratio, current_alpha, current_beta
            )
            adj_4, x_4 = GraphAdversarialAttack(
                scAGCLmodel, edgeind, edgeind_2, x_sub, x_2,
                current_iters, current_node_ratio, current_alpha, current_beta
            )
            z_3 = scAGCLmodel(x_3, adj_3)
            z_4 = scAGCLmodel(x_4, adj_4)
            loss2 = scAGCLmodel.loss(z_3, z_4, batch_size=0)

            if not torch.isfinite(loss2):
                print(f'[WARNING] Epoch {epoch}: loss2 NaN/Inf — skipping adversarial.')
                loss2 = torch.tensor(0.0, device=device)

        loss = loss1 + lam * loss2
        loss.backward()
        optimizer.step()

        l1_val     = loss1.item()
        prev_loss1 = l1_val
        if loss1_ref is None:
            loss1_ref_buffer.append(l1_val)
            if len(loss1_ref_buffer) >= LOSS_REF_EPOCHS:
                loss1_ref = float(np.mean(loss1_ref_buffer))
                print(f'[INFO] loss1_ref calibrated to {loss1_ref:.4f}')

        raw      = loss.item()
        ema_loss = raw if ema_loss is None else ema_decay * ema_loss + (1 - ema_decay) * raw

        if epoch <= num_epochs and epoch % 20 == 0:
            now_time   = t()
            loss2_val  = loss2.item() if torch.is_tensor(loss2) else float(loss2)
            warmup_tag = ' [WARMUP]' if epoch <= WARMUP_EPOCHS else ''
            best_score = top_scores[0][0] if top_scores else -np.inf
            print(
                f'Epoch={epoch:03d}{warmup_tag} | '
                f'loss1={l1_val:.4f}  loss2={loss2_val:.4f}  '
                f'total={raw:.4f}  ema={ema_loss:.4f}  '
                f'best_score={best_score:.4f} | '
                f'time={now_time - start_time:.2f}s'
            )

    if not top_scores:
        scAGCLmodel.eval()
        with torch.no_grad():
            X   = data.x.to(device)
            Adj = edge2adj(data.x, data.edge_index).to(device)
            return scAGCLmodel(X, Adj)

    # =========================================================================
    # CONVERGENCE-AWARE FINAL SELECTION
    #
    # Ensemble averaging (TOP_K=3) cancels subgraph-sampling noise for datasets
    # whose best checkpoint is found mid-training (plateau behaviour).
    # It dilutes performance for monotonically-improving datasets where the
    # best checkpoint is always the latest one.
    #
    # Detection via epoch tracking (no recomputation, no random-state risk):
    #   best_epoch > 85% of T  →  monotone  →  collapse to TOP_K=1
    #   best_epoch ≤ 85% of T  →  plateau   →  keep TOP_K=3 ensemble
    # =========================================================================
    MONOTONE_THRESHOLD = int(num_epochs * 0.85)
    best_epoch = top_scores[0][2]

    if best_epoch > MONOTONE_THRESHOLD:
        print(f'[INFO] Monotone: best epoch={best_epoch} '
              f'> {MONOTONE_THRESHOLD} (85% of {num_epochs}) '
              f'-> TOP_K=1 (pool best, no ensemble dilution)')
        top_scores = [top_scores[0]]
    else:
        print(f'[INFO] Plateau: best epoch={best_epoch} '
              f'<= {MONOTONE_THRESHOLD} '
              f'-> TOP_K={len(top_scores)} ensemble')

    Z_tensors  = [entry[1] for entry in top_scores]
    Z_ensemble = torch.stack(Z_tensors, dim=0).mean(dim=0)
    return Z_ensemble