"""
Genome-wide VNTR heritability simulation using prebuilt GRMs and mixed causal
features sampled from a pruned feature pool.

This script samples causal features only from the same feature space used to
construct the joint model, including copy-number features and pruned sequence
features. The number of causal features assigned to each feature class is
proportional to the total number of available features in that class.

Objective:
    Evaluate whether the joint model can recover unbiased heritability estimates
    when the simulated causal feature space is fully aligned with the tested
    feature space.
"""

import os, sys, subprocess, logging, glob
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# -- 1. Paths and Configuration ------------------------------------------------
BASE = "/home/s3020226030/1_rSV/01_human_chm13"
BFILE_DIR = f"{BASE}/25_hsq_real_data/bfile"

POOL_BFILE_RLAV = f"{BFILE_DIR}/rLAV_VNTR"
POOL_BFILE_NLAV = f"{BFILE_DIR}/nLAV_VNTR"
TSV_COPY_NUMBER = f"{BASE}/06_dosage/LAV_VNTR.copy_number_exact.tsv"

# Pruned sequence-feature list used for causal sampling.
WORK_DIR = f"{BASE}/25_hsq_real_data/08_Pruned_GRM_Inputs"
KEEP_LIST = f"{WORK_DIR}/pruned_rLAV_nLAV_VNTR_r2_0.05.keep.snplist"

# Output directory for the mixed-causal simulation sampled from the pruned pool.
OUT_DIR = f"{BASE}/24_hsq_simulation/real_data_simulation/results_VNTR_prebuilt_GRMs_Mixed_causal_pruned_pool_100reps"

PREBUILT_GRMS = {
    'CN': f"{BASE}/25_hsq_real_data/04_WG_hsq_comparison_ManualGRM_LAV_CN/GRMs/grm_manual_LAV_VNTR",
    'rLAV': f"{BASE}/25_hsq_real_data/01_WG_hsq_comparison_ManualGRM_raw_VNTR/GRMs/grm_manual_rLAV",
    'LAV': f"{BASE}/25_hsq_real_data/01_WG_hsq_comparison_ManualGRM_raw_VNTR/GRMs/grm_manual_LAV",
    'rLAV_nLAV': f"{BASE}/25_hsq_real_data/01_WG_hsq_comparison_ManualGRM_raw_VNTR/GRMs/grm_manual_rLAV_nLAV",
    'Joint_Pruned_05': f"{BASE}/25_hsq_real_data/08_Pruned_GRM_Inputs/GRMs/grm_joint_pruned_CN_rLAV_nLAV_VNTR_r2_0.05"
}

N_CAUSAL = 100
HSQ_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
N_REPS = 100
THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', 20))
SEED = 42

for subdir in ["", "grms", "phenos", "remls", "tmp"]:
    os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(f"{OUT_DIR}/simulation.log"), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# -- 2. Helper Functions -------------------------------------------------------
def run(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        log.error(f"Command failed: {' '.join(cmd)}\n{r.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return r


def parse_gcta_reml(hsq_file):
    try:
        with open(hsq_file) as f:
            for line in f:
                if line.startswith('V(G)/Vp'):
                    parts = line.split()
                    return float(parts[1]), float(parts[2])
    except Exception:
        pass
    return None, None


def load_raw_genotypes(raw_file):
    df = pd.read_csv(raw_file, sep=r'\s+')
    iids = df['IID'].astype(str).tolist()
    raw_cols = list(df.columns[6:])
    snp_ids = [c.rsplit('_', 1)[0] for c in raw_cols]

    X = df[raw_cols].values.astype(np.float64)
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return iids, snp_ids, X


def standardize_and_cov(X):
    stds = X.std(axis=0, ddof=1)
    nz = stds > 0
    Z = np.zeros_like(X)
    Z[:, nz] = (X[:, nz] - X[:, nz].mean(axis=0)) / stds[nz]
    P = int(nz.sum())
    G = (Z @ Z.T) / P if P > 0 else np.zeros((X.shape[0], X.shape[0]))
    return G, P, Z


def save_grm(G, P, sample_ids, out_prefix):
    n = len(sample_ids)
    rows, cols = np.tril_indices(n)
    with open(out_prefix + '.grm.id', 'w') as f:
        for s in sample_ids:
            f.write(f"{s}\t{s}\n")
    G[rows, cols].astype(np.float32).tofile(out_prefix + '.grm.bin')
    np.full(len(rows), float(P), dtype=np.float32).tofile(out_prefix + '.grm.N.bin')


def gcta_reml(grm_prefix, phen_file, out_prefix):
    cmd = [
        'gcta64', '--reml',
        '--grm', grm_prefix,
        '--pheno', phen_file,
        '--out', out_prefix,
        '--thread-num', '1'
    ]
    run(cmd, check=False)
    return parse_gcta_reml(out_prefix + '.hsq')


# -- 3. Main Simulation Workflow -----------------------------------------------
def main():
    log.info(f"VNTR mixed-causal simulation started. Sampling only from the pruned joint pool. Threads: {THREADS}")

    # 1. Load the anchor sample list.
    log.info(f"Phase 1: Loading anchor sample list from {PREBUILT_GRMS['CN']}.grm.id ...")
    if not os.path.exists(PREBUILT_GRMS['CN'] + '.grm.id'):
        log.error(f"Missing base GRM ID file: {PREBUILT_GRMS['CN']}.grm.id")
        sys.exit(1)

    df_id = pd.read_csv(PREBUILT_GRMS['CN'] + '.grm.id', sep='\t', header=None, names=['FID', 'IID'])
    valid_iids = df_id['IID'].astype(str).tolist()
    N_samples = len(valid_iids)
    log.info(f"Aligned sample count: {N_samples}")

    # 2. Load the full copy-number matrix and count available features.
    log.info("Phase 2: Loading copy-number data and counting available features...")
    cn_df = pd.read_csv(TSV_COPY_NUMBER, sep='\t')
    cn_df[valid_iids] = cn_df[valid_iids].apply(pd.to_numeric, errors='coerce')
    X_cn_full = cn_df[valid_iids].values.T.astype(np.float64)

    col_means_cn = np.nanmean(X_cn_full, axis=0)
    nan_mask_cn = np.isnan(X_cn_full)
    X_cn_full[nan_mask_cn] = np.take(col_means_cn, np.where(nan_mask_cn)[1])
    if np.isnan(X_cn_full).any():
        X_cn_full[np.isnan(X_cn_full)] = 0.0

    N_loci_cn = X_cn_full.shape[1]

    # 3. Build the pruned sequence-feature sampling pool.
    log.info("Phase 3: Reading the pruned sequence-feature list to construct the sampling pool...")
    if not os.path.exists(KEEP_LIST):
        log.error(f"Missing keep list: {KEEP_LIST}")
        sys.exit(1)

    with open(KEEP_LIST, 'r') as f:
        pruned_pool_ids = f.read().splitlines()

    pool_size = len(pruned_pool_ids)

    # 4. Calculate the feature-proportional causal allocation.
    total_variants = N_loci_cn + pool_size
    n_causal_cn = int(np.round(N_CAUSAL * (N_loci_cn / total_variants)))
    n_causal_seq = N_CAUSAL - n_causal_cn

    log.info("=" * 60)
    log.info("Feature-proportional causal sampling configuration (pruned features only):")
    log.info(f"   -> Total copy-number features: {N_loci_cn:,}")
    log.info(f"   -> Total rLAV+nLAV pruned sequence features: {pool_size:,}")
    log.info(f"   -> Target total causal features: {N_CAUSAL}")
    log.info(f"   -> Allocation: {n_causal_cn} from CN, {n_causal_seq} from pruned sequence features")
    log.info("=" * 60)

    # 5. Run simulation replicates.
    log.info("Phase 4: Sampling mixed causal features, simulating phenotypes, and queuing REML tasks...")
    reml_tasks = []
    rng = np.random.default_rng(SEED)
    tmp_dir = os.path.join(OUT_DIR, "tmp")

    for rep in range(1, N_REPS + 1):
        # ---------------------------------------------------------
        # A. Sample causal copy-number features.
        # ---------------------------------------------------------
        cn_idx = rng.choice(N_loci_cn, n_causal_cn, replace=False) if N_loci_cn >= n_causal_cn else np.arange(N_loci_cn)
        X_causal_cn = X_cn_full[:, cn_idx]

        # ---------------------------------------------------------
        # B. Sample causal sequence features from the pruned list only.
        # ---------------------------------------------------------
        seq_idx = rng.choice(pool_size, n_causal_seq, replace=False) if pool_size >= n_causal_seq else np.arange(pool_size)
        causal_snps = [pruned_pool_ids[i] for i in seq_idx]

        snp_list_file = os.path.join(tmp_dir, f"causal_snps_r{rep}.txt")
        with open(snp_list_file, 'w') as f:
            f.write('\n'.join(causal_snps))

        tmp_r = os.path.join(tmp_dir, f"tmp_r_{rep}")
        tmp_n = os.path.join(tmp_dir, f"tmp_n_{rep}")
        tmp_merge = os.path.join(tmp_dir, f"tmp_merge_{rep}")

        run(['plink', '--bfile', POOL_BFILE_RLAV, '--extract', snp_list_file, '--make-bed', '--out', tmp_r, '--allow-extra-chr', '--silent'], check=False)
        run(['plink', '--bfile', POOL_BFILE_NLAV, '--extract', snp_list_file, '--make-bed', '--out', tmp_n, '--allow-extra-chr', '--silent'], check=False)

        has_r = os.path.exists(tmp_r + ".bim") and os.path.getsize(tmp_r + ".bim") > 0
        has_n = os.path.exists(tmp_n + ".bim") and os.path.getsize(tmp_n + ".bim") > 0

        if has_r and has_n:
            run(['plink', '--bfile', tmp_r, '--bmerge', tmp_n, '--recode', 'A', '--out', tmp_merge, '--allow-extra-chr', '--silent'], check=False)
        elif has_r:
            run(['plink', '--bfile', tmp_r, '--recode', 'A', '--out', tmp_merge, '--allow-extra-chr', '--silent'], check=False)
        elif has_n:
            run(['plink', '--bfile', tmp_n, '--recode', 'A', '--out', tmp_merge, '--allow-extra-chr', '--silent'], check=False)
        else:
            raise RuntimeError(f"Rep {rep}: Both extracts failed to produce variants.")

        raw_iids, _, X_causal_seq_raw = load_raw_genotypes(tmp_merge + ".raw")

        # Force the sequence-feature matrix to follow the anchor sample order.
        idx_map = {str(iid): i for i, iid in enumerate(raw_iids)}
        aligned_idx = [idx_map[str(iid)] for iid in valid_iids]
        X_causal_seq = X_causal_seq_raw[aligned_idx, :]

        # ---------------------------------------------------------
        # C. Combine causal features and construct the oracle causal GRM.
        # ---------------------------------------------------------
        X_causal_mixed = np.hstack([X_causal_cn, X_causal_seq])

        G_causal, P_causal, Z_causal = standardize_and_cov(X_causal_mixed)
        causal_grm_pfx = f"{OUT_DIR}/grms/causal_oracle_r{rep}"
        save_grm(G_causal, P_causal, valid_iids, causal_grm_pfx)

        # Remove temporary files generated for the current replicate.
        for f in glob.glob(f"{tmp_dir}/*_{rep}*"):
            os.remove(f)

        # ---------------------------------------------------------
        # D. Simulate mixed-causal phenotypes.
        # ---------------------------------------------------------
        for hsq in HSQ_LEVELS:
            beta = rng.normal(0, 1.0 / np.sqrt(P_causal), size=P_causal)
            G_raw = Z_causal @ beta
            var_G_raw = np.var(G_raw, ddof=1)

            G_true = G_raw * np.sqrt(hsq / var_G_raw) if var_G_raw > 0 else G_raw

            e_raw = rng.normal(0, 1, size=N_samples)
            var_e_raw = np.var(e_raw, ddof=1)
            e = e_raw * np.sqrt((1.0 - hsq) / var_e_raw)

            Y = G_true + e
            emp_hsq = float(np.var(G_true, ddof=1) / np.var(Y, ddof=1))

            phen_path = os.path.join(OUT_DIR, "phenos", f"hsq{int(hsq*100)}_rep{rep}.phen")
            with open(phen_path, 'w') as f:
                for iid, y_val in zip(valid_iids, Y):
                    f.write(f"{iid}\t{iid}\t{y_val:.8f}\n")

            grms_to_test = {'0_Oracle_Causal': causal_grm_pfx, **PREBUILT_GRMS}

            for model, grm_pfx in grms_to_test.items():
                out_pfx = os.path.join(OUT_DIR, "remls", f"reml_{model}_h{int(hsq*100)}_r{rep}")
                reml_tasks.append({
                    'n_causal': P_causal, 'model': model, 'hsq': hsq, 'rep': rep,
                    'emp_hsq': emp_hsq, 'grm_pfx': grm_pfx, 'phen_path': phen_path, 'out_pfx': out_pfx
                })

    # 6. Execute REML models in parallel.
    log.info(f"Phase 5: Running {len(reml_tasks)} REML models in parallel...")
    all_results = []
    n_done = 0

    def process_reml(task):
        est, se = gcta_reml(task['grm_pfx'], task['phen_path'], task['out_pfx'])
        for ext in ['.hsq', '.log']:
            try:
                os.remove(f"{task['out_pfx']}{ext}")
            except Exception:
                pass
        return {
            'n_causal': task['n_causal'], 'model': task['model'], 'true_hsq': task['hsq'],
            'rep': task['rep'], 'emp_hsq': task['emp_hsq'], 'est_hsq': est, 'se': se,
        }

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(process_reml, t) for t in reml_tasks]
        for f in as_completed(futures):
            all_results.append(f.result())
            n_done += 1
            if n_done % 50 == 0 or n_done == len(reml_tasks):
                log.info(f"   -> Progress: {n_done}/{len(reml_tasks)}")

    # 7. Export summary reports.
    log.info("Phase 6: Summarizing and exporting simulation results...")
    out_tsv = os.path.join(OUT_DIR, "raw_simulation_results.tsv")
    df = pd.DataFrame(all_results)
    df.to_csv(out_tsv, sep='\t', index=False)

    if not df.empty:
        summary = (df.dropna(subset=['est_hsq'])
                     .groupby(['model', 'true_hsq'])
                     .agg(mean_est=('est_hsq', 'mean'),
                          mean_emp=('emp_hsq', 'mean'),
                          n=('est_hsq', 'count'))
                     .reset_index())
        summary['bias'] = summary['mean_est'] - summary['true_hsq']

        log.info("\n" + "=" * 85)
        log.info("Simulation result summary under fully aligned features: CN + pruned sequence features")
        log.info("=" * 85)

        pivot_df = summary.pivot(index='model', columns='true_hsq', values='mean_est')
        log.info("\n" + pivot_df.to_string(float_format="%.5f"))
        log.info("=" * 85)

        summary.to_csv(os.path.join(OUT_DIR, "summary_simulation_bias.tsv"), sep='\t', index=False)
        log.info(f"Simulation completed. Summary results saved to: {OUT_DIR}/summary_simulation_bias.tsv")


if __name__ == '__main__':
    main()
