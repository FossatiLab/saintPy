# saintpy/main.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

# keep your existing imports as-is
import math
import pandas as pd
from saintpy.io import build_inputs_from_files, build_output_df, UIClass
from saintpy.stats import ModelData, statModel, calculateScore
from saintpy.inference import icm_Z, wrt_MRF_gamma_0

log = logging.getLogger("saintpy")


class Options:
    def __init__(self, f=0.0, R=100, L=100):
        self.f = f
        self.R = R
        self.L = L


def run_pipeline(prey_file: str, bait_file: str, inter_file: str, opts) -> pd.DataFrame:
    I = build_inputs_from_files(prey_file, bait_file, inter_file)
    model = statModel(
        I["p2p_mapping"],
        I["ubait"],
        I["test_mat_DATA"],
        I["ctrl_mat_DATA"],
        I["ip_idx_to_bait_no"],
        I["nprey"],
        I["nbait"],
        opts
    )
    # attach fields used by build_output_df
    model.PDATA = I["PDATA"]
    model.ubait = I["ubait"]

    # inference loop (unchanged)
    for _ in range(15):
        old_ll = model.llikelihood()
        icm_Z(model)               # update Z
        wrt_MRF_gamma_0(model)     # optimize β₁
        new_ll = model.llikelihood()
        if new_ll >= old_ll and (math.exp(new_ll - old_ll) - 1) < 1e-3:
            break

    # scoring (unchanged)
    avgp, maxp, min_logodds = calculateScore(model)

    # replicate map for mask construction
    BDATA = I["BDATA"]
    ubait = I["ubait"]
    nbait = I["nbait"]

    ubait_map = {bait_id: i for i, bait_id in enumerate(ubait)}
    ip_idx_to_bait_no = []
    for bait in BDATA:
        if not bait.get_isCtrl():
            ip_idx_to_bait_no.append(ubait_map[bait.get_baitId()])

    bait_no_to_ip_idxes = [[] for _ in range(nbait)]
    for ip_idx, bait_no in enumerate(ip_idx_to_bait_no):
        bait_no_to_ip_idxes[bait_no].append(ip_idx)

    # build mask: 1 if any replicate has counts > 0
    test_mat_mask = np.zeros((I["nprey"], nbait), dtype=int)
    for j, ip_indices in enumerate(bait_no_to_ip_idxes):
        if not ip_indices:
            continue
        nz = (I["test_mat_DATA"][:, ip_indices] > 0).any(axis=1)
        test_mat_mask[nz, j] = 1

    df = build_output_df(
        model,
        avgp=np.asarray(avgp),
        maxp=np.asarray(maxp),
        min_logodds=np.asarray(min_logodds),
        ip_idx_to_bait_no=I["ip_idx_to_bait_no"],
        test_mat_DATA=I["test_mat_DATA"],
        ctrl_mat_DATA=I["ctrl_mat_DATA"],
        test_mat_mask=test_mat_mask,
        topo_avgp=None,
        topo_maxp=None,
    )
    return df


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="saintpy",
        description="Run SAINT-style scoring from fixed-format inputs and write a results CSV."
    )
    p.add_argument("--prey", required=True, help="Path to prey.dat (preyId preyLength)")
    p.add_argument("--bait", required=True, help="Path to bait.dat (ipId baitId T|C)")
    p.add_argument("--inter", required=True, help="Path to inter.dat (ipId baitId preyId quant)")
    p.add_argument("--out", default='list.csv', help="Output CSV path")

    # expose your Options
    p.add_argument("--f", type=float, default=0.0, help="Options.f")
    p.add_argument("--R", type=int, default=100, help="Options.R")
    p.add_argument("--L", type=int, default=100, help="Options.L")

    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # logging
    level = logging.WARNING - (10 * min(args.verbose, 2))  # WARNING/INFO/DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # sanity checks
    for pth in (args.prey, args.bait, args.inter):
        if not Path(pth).is_file():
            log.error("Input not found: %s", pth)
            return 2

    try:
        df = run_pipeline(
            prey_file=args.prey,
            bait_file=args.bait,
            inter_file=args.inter,
            opts=Options(f=args.f, R=args.R, L=args.L),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("Wrote %s (%d rows)", out_path, len(df))
        return 0
    except Exception as e:
        log.exception("Failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
