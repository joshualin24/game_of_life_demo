# Periodic Study — Game of Life Oscillators & Periodic Objects

## Goal

Study the distribution, structure, and properties of **periodic objects** in
Conway's Game of Life (B3/S23) as catalogued by
[Catagolue](https://catagolue.hatsya.com/statistics) — the largest crowdsourced
census of naturally-occurring GoL objects, with data from over
333 quadrillion asymmetric 16×16 random soups.

We focus on **oscillators** (objects that return to their original state after
a fixed number of steps) and how their properties relate to their period,
population, and frequency of natural occurrence.

## Research Questions

1. **Period distribution**: how are oscillators distributed across periods?
   Which periods are common, which are rare or absent?
2. **Population vs. period**: do higher-period oscillators tend to be larger?
3. **Natural frequency**: how does the frequency of natural occurrence decay
   with period and population?
4. **Embedding structure**: do oscillators of the same period cluster together
   in the VAE / contrastive embedding space?
5. **Omniperiodicity**: following arXiv:2312.02799, every period ≥ 1 is now
   known to be achievable — can we recover the period structure from embeddings
   alone, without knowing the period label?

## Data Source

All data is downloaded from [Catagolue](https://catagolue.hatsya.com) via its
public HTTP API:

| Endpoint | Description |
|---|---|
| `/census/b3s23/C1/xp{n}` | Census page for period-n oscillators (HTML) |
| `/object/{apgcode}/b3s23` | Individual object page with RLE pattern |
| `/statistics` | Aggregate statistics across all soups |

The `apgcode` format encodes the object type:
- `xs` — still life (xp1)
- `xp{n}` — oscillator of period n
- `xg{n}` — spaceship of period n (up to translation)

## Directory Structure

```
periodic_study/
├── README.md             # this file
├── download_data.py      # fetch oscillator census data from Catagolue
├── data/                 # downloaded data (gitignored)
│   ├── census_xp2.json   # period-2 oscillators
│   ├── census_xp3.json   # period-3 oscillators
│   └── ...
└── results/              # plots and analysis outputs
```

## Quick Start

```bash
# Download oscillator census data for periods 2–30
python download_data.py

# (analysis scripts to follow)
```
