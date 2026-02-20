# Permeability and Efflux Prediction (GNN-MTL)

**Ersilia Model ID:** `eos0bbb`

| | |
|---|---|
| **Task** | Regression (multitask) |
| **Input** | Single SMILES |
| **Output** | 4 log10-transformed ADME scores |
| **Framework** | [Chemprop](https://github.com/chemprop/chemprop) v2.1.0 (MPNN) |
| **License** | CC-BY-4.0 |

## Description

Predicts cell membrane permeability and efflux transport using a multitask graph neural network. The model simultaneously predicts four endpoints from a single compound:

| Output | Description | Units |
|--------|-------------|-------|
| `caco2_er` | Caco-2 efflux ratio | log10(unitless) |
| `caco2_papp` | Caco-2 apparent permeability | log10(x10^-6 cm/s) |
| `mdck_er` | MDCK efflux ratio | log10(unitless) |
| `nih_mdck_er` | NIH-MDCK efflux ratio | log10(unitless) |

Built with Chemprop v2.1 using a message-passing neural network (MPNN, d_h=300, depth=3) trained on a harmonized single-laboratory dataset of over 10,000 compounds from Caco-2 and MDCK cell-line assays.

## Interpretation

Outputs are log10-transformed. Efflux ratios (Caco-2 ER, MDCK ER, NIH-MDCK ER) indicate active efflux when >0.30 (i.e. ER > 2 in linear scale). Caco-2 P_app reflects log10 of apparent permeability (x10^-6 cm/s), where higher values indicate better membrane permeation. Exponentiate (10^x) to recover linear-scale values.

## Source

- **Publication:** [Ohlsson et al., ACS Omega 2025, 10(45), 54148-54159](https://pubs.acs.org/doi/10.1021/acsomega.5c04861)
- **Source Code:** https://zenodo.org/records/16948542

## Deep Validation

| Check | Status | Details |
|-------|--------|---------|
| Distribution (50 molecules) | PASS | 100% valid, 4/4 columns non-constant, 4.5s runtime |
| Sanity (permeability) | PASS | High-perm 1.36 vs low-perm -0.95 (2.31 log-unit separation) |
| BCS classification | PASS | Class I 1.02 vs Class III -0.35 (1.37 log-unit separation) |
| P-gp efflux | PASS | Substrates 0.34 vs non-substrates -0.30 (0.63 log-unit separation) |

**Overall: PASS (4/4)**

### Highlights

- **Distribution analysis:** 50 diverse molecules processed in 4.5s. All 4 output columns are non-constant with high coefficients of variation (1.4-3.7), indicating good discrimination across chemical space.

- **Sanity check:** 8 known high-permeability drugs (ibuprofen, diazepam, sildenafil, etc.) score 2.3 log-units higher in Caco-2 P_app than 8 low-permeability compounds (mannitol, glucose, citric acid, etc.). This matches fundamental pharmacological expectations.

- **BCS classification:** BCS Class I drugs (high solubility + high permeability) correctly separated from BCS Class III drugs (high solubility + low permeability) by 1.37 log-units.

- **P-gp efflux:** Known P-glycoprotein substrates (sildenafil, cocaine) show higher Caco-2 efflux ratios than passively permeable non-substrates (benzene, ethanol), with 0.63 log-unit separation.

See [`deep_validation.ipynb`](deep_validation.ipynb) for full analysis with visualizations.

## Usage

```python
# Input CSV format
# smiles
# CC(=O)Oc1ccccc1C(=O)O
# CC(C)Cc1ccc(cc1)C(C)C(=O)O

python model/framework/code/main.py input.csv output.csv
```

## Dependencies

```yaml
python: "3.10"
commands:
    - ["pip", "chemprop", "2.1.0"]
```
