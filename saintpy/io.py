import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Deque, Dict, Set, Tuple
from collections import deque, defaultdict



@dataclass
class InterClass:
    ipId: str = ""
    baitId: str = ""
    preyId: str = ""
    quant: int = 0
    rowId: int = -1
    colId: int = -1
    is_ctrl: bool = False

    def set_ipId(self, v): self.ipId = v
    def set_baitId(self, v): self.baitId = v
    def set_preyId(self, v): self.preyId = v
    def set_quant(self, v): self.quant = int(v)
    def set_rowId(self, v): self.rowId = int(v)
    def set_colId(self, v): self.colId = int(v)

    def get_ipId(self): return self.ipId
    def get_baitId(self): return self.baitId
    def get_preyId(self): return self.preyId
    def get_quant(self): return self.quant
    def get_rowId(self): return self.rowId
    def get_colId(self): return self.colId

@dataclass
class PreyClass:
    rowId: int = -1
    preyId: str = ""
    preyLength: float = 0.0
    preyGeneId: str = ""

    def set_rowId(self, v): self.rowId = int(v)
    def set_preyId(self, v): self.preyId = v
    def set_preyLength(self, v): self.preyLength = float(v)
    def set_preyGeneId(self, v): self.preyGeneId = v

    def get_rowId(self): return self.rowId
    def get_preyId(self): return self.preyId
    def get_preyGeneId(self): return self.preyGeneId

@dataclass
class BaitClass:
    colId: int = -1
    ipId: str = ""
    baitId: str = ""
    isCtrl: bool = False

    def set_colId(self, v): self.colId = int(v)
    def set_ipId(self, v): self.ipId = v
    def set_baitId(self, v): self.baitId = v
    def set_isCtrl(self, v): self.isCtrl = bool(v)

    def get_colId(self): return self.colId
    def get_ipId(self): return self.ipId
    def get_baitId(self): return self.baitId
    def get_isCtrl(self): return self.isCtrl

@dataclass
class UIClass:
    baitId: str = ""
    preyId: str = ""
    preyGeneId: str = ""
    rowId: int = -1
    colIds: List[int] = field(default_factory=list)

    def set_baitId(self, v): self.baitId = v
    def set_preyId(self, v): self.preyId = v
    def set_preyGeneId(self, v): self.preyGeneId = v
    def set_rowId(self, v): self.rowId = int(v)
    def add_colId(self, v): self.colIds.append(int(v))

    def get_baitId(self): return self.baitId
    def get_preyId(self): return self.preyId
    def get_preyGeneId(self): return self.preyGeneId
    def get_rowId(self): return self.rowId
    def get_colId(self): return list(self.colIds)


def detect_line_ending(_path: str) -> str:
    # C++ code distinguishes CRLF vs LF; safe default:
    return "\n"

# ---------- ports of your I/O + mapping functions ----------
def getFileDimensions(inputFile: str) -> Tuple[int, int]:
    with open(inputFile, "r", encoding="utf-8", newline="") as inF:
        lineCtr = 0
        for line in inF:
            if len(line.strip()) > 10:
                lineCtr += 1
        numRows = lineCtr - 1  # header ignored

    with open(inputFile, "r", encoding="utf-8", newline="") as inF:
        header = inF.readline()
        v = splitString(header)
        numCols = len(v) - 1  # first element is row header

    return numRows, numCols


def get_nexpr(BDATA: Deque[BaitClass]) -> int:
    return sum(1 for m in BDATA if not m.get_isCtrl())

def get_nctrl(BDATA: Deque[BaitClass]) -> int:
    return sum(1 for m in BDATA if m.get_isCtrl())


def createList(UIDATA: Deque[UIClass],
               IDATA: Deque[InterClass],
               BDATA: Deque[BaitClass],
               PDATA: Deque[PreyClass],
               nprey: int, nbait: int,
               ubait: List[str],
               ip_idx_to_bait_no: List[int]) -> int:
    ip_idx_to_bait_no.clear()
    ubait_map = {ubait[i]: i for i in range(len(ubait))}
    for bait in BDATA:
        if not bait.get_isCtrl():
            ip_idx_to_bait_no.append(ubait_map[bait.get_baitId()])

    UIDATA.clear()
    # sparse pointer matrix UImat: (nprey x nbait)
    UImat: List[List[UIClass]] = [[None for _ in range(nbait)] for _ in range(nprey)]

    for inter in IDATA:
        if not inter.is_ctrl:
            j = inter.get_colId()
            b = ip_idx_to_bait_no[j]
            i = inter.get_rowId()
            if UImat[i][b] is None:
                UIDATA.append(UIClass())
                tmp = UIDATA[-1]
                tmp.set_baitId(inter.get_baitId())
                tmp.set_preyId(inter.get_preyId())
                tmp.set_preyGeneId(PDATA[i].get_preyGeneId())
                tmp.set_rowId(i)
                UImat[i][b] = tmp
            UImat[i][b].add_colId(j)
    nuinter = len(UIDATA)
    return nuinter


def sortBaitData(BDATA: Deque[BaitClass]) -> None:
    newBDATA = deque()
    for m in BDATA:
        if m.get_isCtrl():
            newBDATA.append(m)
    for m in BDATA:
        if not m.get_isCtrl():
            newBDATA.append(m)
    if len(BDATA) != len(newBDATA):
        print("Bait sorting error")
    BDATA.clear()
    BDATA.extend(newBDATA)


def mapRowCol(IDATA: Deque[InterClass],
              PDATA: Deque[PreyClass],
              BDATA: Deque[BaitClass],
              bait_Id_map: Dict[str, BaitClass]) -> None:
    prey_by_id = {p.get_preyId(): p for p in PDATA}
    bait_by_ip = {b.get_ipId(): b for b in BDATA}
    for m in IDATA:
        mp = prey_by_id.get(m.get_preyId())
        if mp is None:
            raise RuntimeError(f"prey {m.get_preyId()} not found")
        m.set_rowId(mp.get_rowId())

        mb = bait_by_ip.get(m.get_ipId())
        if mb is None:
            raise RuntimeError("bait not found")
        m.set_colId(mb.get_colId())
        m.is_ctrl = bait_Id_map[m.get_ipId()].get_isCtrl()



def parseInterFile(inputFile: str) -> Tuple[Deque[InterClass], int]:
    print(f"Parsing interaction file {inputFile} ...", end="")
    IDATA: Deque[InterClass] = deque()
    ninter = 0
    line_ending = detect_line_ending(inputFile)
    with open(inputFile, "r", encoding="utf-8", newline="") as inF:
        for raw in inF.read().split(line_ending):
            line = raw.strip()
            curLineVec = splitString(line)
            if len(curLineVec) < 4:
                continue
            tmp = InterClass()
            tmp.set_ipId(curLineVec[0])
            tmp.set_baitId(curLineVec[1])
            tmp.set_preyId(curLineVec[2])
            tmp.set_quant(float(curLineVec[3]))
            IDATA.append(tmp)
            ninter += 1
    print("done.")
    return IDATA, ninter


def parsePreyFile(PDATA: Deque[PreyClass], inputFile: str) -> Tuple[Set[str], int]:
    print(f"Parsing prey file {inputFile} ...", end="")
    line_ending = detect_line_ending(inputFile)
    nprey = 0
    prey_Id_set: Set[str] = set()
    with open(inputFile, "r", encoding="utf-8", newline="") as inF:
        for raw in inF.read().split(line_ending):
            line = raw.strip()
            if not line:
                continue
            curLineVec = splitString(line)

            prey_id = curLineVec[0]
            prey_length = 0.0
            gene_name = ""

            if len(curLineVec) == 2:
                # If second col is not numeric, treat as gene name
                try:
                    prey_length = float(curLineVec[1])
                    gene_name = curLineVec[0]
                except ValueError:
                    gene_name = curLineVec[1]
            elif len(curLineVec) > 2:
                prey_length = float(curLineVec[1])
                gene_name = curLineVec[2]

            tmp = PreyClass()
            tmp.set_rowId(nprey)
            tmp.set_preyId(prey_id)
            prey_Id_set.add(prey_id)
            tmp.set_preyLength(prey_length)
            tmp.set_preyGeneId(gene_name)
            PDATA.append(tmp)
            nprey += 1

    if nprey != len(prey_Id_set):
        raise RuntimeError("duplicate preys in prey file")
    print("done.")
    return prey_Id_set, nprey


def parseBaitFile(BDATA: Deque[BaitClass], inputFile: str) -> Tuple[Dict[str, BaitClass], int, int]:
    print(f"Parsing prey file {inputFile} ...", end="")
    line_ending = detect_line_ending(inputFile)
    nip = 0
    bait_Id_map: Dict[str, BaitClass] = {}
    n_test_ip = 0
    n_ctrl_ip = 0
    with open(inputFile, "r", encoding="utf-8", newline="") as inF:
        for raw in inF.read().split(line_ending):
            line = raw.strip()
            if line == "":
                continue
            curLineVec = splitString(line)
            if len(curLineVec) != 3:
                print(nip)
                print(len(curLineVec))
                print(curLineVec[0] if curLineVec else "")
                raise RuntimeError("Bait file must have 3 columns")

            node1, node2, node3 = curLineVec[0], curLineVec[1], curLineVec[2]
            if node3 not in ("T", "C"):
                raise RuntimeError("3rd column of bait file must be 'T' or 'C'")

            tmp = BaitClass()
            tmp.set_colId(n_test_ip if node3 == "T" else n_ctrl_ip)
            if node3 == "T":
                n_test_ip += 1
            else:
                n_ctrl_ip += 1
            tmp.set_ipId(node1)
            tmp.set_baitId(node2)
            tmp.set_isCtrl(node3 != "T")

            BDATA.append(tmp)
            bait_Id_map[node1] = tmp
            nip += 1
    print("done.")
    return bait_Id_map, nip, n_test_ip


def createMatrixData(test_mat_DATA: np.ndarray,
                     ctrl_mat_DATA: np.ndarray,
                     IDATA: Deque[InterClass]) -> None:
    for m in IDATA:
        r, c, q = m.get_rowId(), m.get_colId(), m.get_quant()
        if m.is_ctrl:
            ctrl_mat_DATA[r, c] = int(q)
        else:
            test_mat_DATA[r, c] = int(q)


def zero_self_interactions(test_mat_DATA: np.ndarray,
                           UIDATA: Deque[UIClass]) -> None:
    for UI in UIDATA:
        if UI.get_preyGeneId() == UI.get_baitId():
            for col in UI.get_colId():
                test_mat_DATA[UI.get_rowId(), col] = 0


def splitString(line: str) -> List[str]:
    # Original splits by whitespace/tabs; keep it simple
    return line.strip().split()


# ---------- glue to produce statModel inputs exactly like C++ ----------
def build_inputs_from_files(prey_file: str, bait_file: str, inter_file: str):
    # parse bait / prey / interactions
    BDATA: Deque[BaitClass] = deque()
    PDATA: Deque[PreyClass] = deque()
    UIDATA: Deque[UIClass]  = deque()

    bait_Id_map, nip, n_test_ip = parseBaitFile(BDATA, bait_file)
    sortBaitData(BDATA)
    prey_set, nprey = parsePreyFile(PDATA, prey_file)
    IDATA, ninter = parseInterFile(inter_file)
    mapRowCol(IDATA, PDATA, BDATA, bait_Id_map)

    # compute sizes
    nctrl = get_nctrl(BDATA)
    nexpr = get_nexpr(BDATA)
    nbait = len({b.get_baitId() for b in BDATA if not b.get_isCtrl()})
    # ubait order is the order of first appearance among test baits (like typical SAINT usage)
    ubait = []
    seen = set()
    for b in BDATA:
        if not b.get_isCtrl():
            if b.get_baitId() not in seen:
                ubait.append(b.get_baitId())
                seen.add(b.get_baitId())

    # allocate matrices (rows = preys)
    test_mat_DATA = np.zeros((nprey, nexpr), dtype=int)
    ctrl_mat_DATA = np.zeros((nprey, nctrl), dtype=int)
    createMatrixData(test_mat_DATA, ctrl_mat_DATA, IDATA)

    # unique interaction list & ip->bait mapping
    ip_idx_to_bait_no: List[int] = []
    nuinter = createList(UIDATA, IDATA, BDATA, PDATA, nprey, nbait, ubait, ip_idx_to_bait_no)

    # zero self interactions
    zero_self_interactions(test_mat_DATA, UIDATA)

    # p2p_mapping is not constructed here in C++; pass empty (no neighbors) unless you have it
    p2p_mapping = [[] for _ in range(nprey)]

    return {
        "PDATA": PDATA,
        "BDATA": BDATA,
        "UIDATA": UIDATA,
        "ubait": ubait,
        "ip_idx_to_bait_no": ip_idx_to_bait_no,
        "test_mat_DATA": test_mat_DATA,
        "ctrl_mat_DATA": ctrl_mat_DATA,
        "p2p_mapping": p2p_mapping,
        "nprey": nprey,
        "nbait": nbait,
        "nexpr": nexpr,
        "nctrl": nctrl,
        "nuinter": nuinter,
    }


def build_output_df(model,
                    avgp, maxp, min_logodds,
                    ip_idx_to_bait_no,                  # <— pass this in
                    test_mat_DATA, ctrl_mat_DATA,       # <— pass these too
                    test_mat_mask,
                    topo_avgp=None, topo_maxp=None):
    import numpy as np
    import pandas as pd

    ubait = model.ubait
    PDATA = model.PDATA
    nprey, _ = test_mat_DATA.shape
    nbait = len(ubait)

    if topo_avgp is None: topo_avgp = avgp
    if topo_maxp is None: topo_maxp = maxp

    # invert mapping: bait_no -> list of test IP indices (replicates)
    bait_no_to_ip_idxes = [[] for _ in range(nbait)]
    for ip_idx, bait_no in enumerate(ip_idx_to_bait_no):
        bait_no_to_ip_idxes[bait_no].append(ip_idx)

    rows = []
    flat_avg = avgp.flatten()

    for i in range(nprey):
        # adapt to your PDATA structure
        prey_id   = PDATA[i].get_preyId()     if hasattr(PDATA[i], "get_preyId")     else PDATA[i]["prey_id"]
        prey_gene = PDATA[i].get_preyGeneId() if hasattr(PDATA[i], "get_preyGeneId") else PDATA[i].get("prey_gene","")

        for j in range(nbait):
            if test_mat_mask is not None and not test_mat_mask[i, j]:
                continue

            bait_id = ubait[j]
            ip_idxes = bait_no_to_ip_idxes[j]
            counts = np.array([test_mat_DATA[i, k] for k in ip_idxes], dtype=float)
            if counts.size == 0:
                continue

            avg_spec  = counts.mean()
            spec_sum  = counts.sum()
            num_reps  = counts.size
            spec_str  = "|".join(str(int(c)) for c in counts)

            ctrl = np.asarray(ctrl_mat_DATA[i, :], dtype=float)
            ctrl_str = "|".join(str(int(c)) for c in ctrl)

            avg_p       = float(avgp[i, j])
            max_p       = float(maxp[i, j])
            topo_avg_p  = float(topo_avgp[i, j])
            topo_max_p  = float(topo_maxp[i, j])
            saint_score = max(topo_avg_p, avg_p)
            log_odds    = float(min_logodds[i, j])
            fold_change = avg_spec / max(float(model.eta[i]), 1e-8)

            # BFDR exactly like SAINT
            greater = flat_avg[flat_avg > avg_p]
            bfdr = 0.0 if greater.size == 0 else (1.0 - float(greater.mean()))

            rows.append({
                "Bait": bait_id,
                "Prey": prey_id,
                "PreyGene": prey_gene,
                "Spec": spec_str,
                "SpecSum": spec_sum,
                "AvgSpec": avg_spec,
                "NumReplicates": num_reps,
                "ctrlCounts": ctrl_str,
                "AvgP": avg_p,
                "MaxP": max_p,
                "TopoAvgP": topo_avg_p,
                "TopoMaxP": topo_max_p,
                "SaintScore": saint_score,
                "logOddsScore": log_odds,
                "FoldChange": fold_change,
                "BFDR": bfdr,
                "boosted_by": ""  # fill from p2p if you want
            })

    cols = ["Bait","Prey","PreyGene","Spec","SpecSum","AvgSpec","NumReplicates",
            "ctrlCounts","AvgP","MaxP","TopoAvgP","TopoMaxP",
            "SaintScore","logOddsScore","FoldChange","BFDR","boosted_by"]
    return pd.DataFrame(rows, columns=cols)
