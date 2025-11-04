# saintPy

saintPy is a Python port of SAINT (Significance Analysis of INTeractome) for
AP-MS and related interaction data.

## Install

```bash
pip install -e .
```

This installs the `saintpy` command.

## Quick start

```bash
saintpy   --prey path/to/prey.dat   --bait path/to/bait.dat   --inter path/to/inter.dat   --out results.csv
```

### Input files

Plain text, whitespace separated, no headers.

**prey.dat**
```
preyId  preyLength
```
Example:
```
ACTR5 607
RUVBL2 463
```

**bait.dat**
```
ipId  baitId  flag
```
`flag` is `T` for test or `C` for control. Example:
```
ARP5 ACTR5 T
CTRL1 CTRLBAIT C
```

**inter.dat**
```
ipId  baitId  preyId  quant
```
`quant` is integer. Example:
```
ARP5 ACTR5 ACTR5 417
ARP5 ACTR5 RUVBL2 73
```

### Output

CSV with one row per bait–prey pair. Columns:
```
Bait, Prey, PreyGene, Spec, SpecSum, AvgSpec, NumReplicates,
ctrlCounts, AvgP, MaxP, TopoAvgP, TopoMaxP,
SaintScore, logOddsScore, FoldChange, BFDR, boosted_by
```

## CLI options

```
--prey   path to prey.dat                (required)
--bait   path to bait.dat                (required)
--inter  path to inter.dat               (required)
--out    output CSV                      (required)
--f      model option f                  default 0.0
--R      model option R                  default 100
--L      model option L                  default 100
-v       increase log verbosity
```

Example:

```bash
saintpy   --prey test/TIP49/prey.dat   --bait test/TIP49/bait.dat   --inter test/TIP49/inter.dat   --out results.csv   --f 0.05 --R 150 -v
```

## Use as a library

```python
from saintpy.main import run_pipeline, Options

df = run_pipeline(
    prey_file="test/TIP49/prey.dat",
    bait_file="test/TIP49/bait.dat",
    inter_file="test/TIP49/inter.dat",
    opts=Options(f=0.05, R=150, L=150)
)
df.to_csv("results.csv", index=False)
```

## Command line usage

```
python -m saintpy.main --prey myprey.dat --bait mybait.dat --inter myinter.dat```
```

## Notes

- `prey.dat` has two columns. No gene field in file. The pipeline sets `PreyGene` to `preyId`.
- `bait.dat` assigns indices for test and control separately. Controls come first in internal storage.
- `inter.dat` rows must match `ipId` and `baitId` seen in `bait.dat`.

## License

MIT


