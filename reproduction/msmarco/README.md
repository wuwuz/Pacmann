# Reproduce the MS-MARCO result from the official download

This directory reproduces the MS-MARCO experiment in Table 1 of the Pacmann
paper, starting with Microsoft's public `msmarco-docs.tsv.gz` download. The
pipeline downloads and validates the data, creates document/query embeddings,
applies the paper's fitted 768-to-192 PCA, builds the degree-32 graph, runs the
private search, and computes MRR@100.

The paper's saved run reported `MRR@100 = 0.266211`. A fresh seeded audit of
the public code produced `MRR@100 = 0.266464` (633/1,000 queries). Search entry
points and PianoPIR failures are randomized, so a run is not expected to equal
all six decimals unless its seed and every artifact are identical. Verified
runs are approximately `0.258`–`0.266`; the automated check accepts the
deliberately wider `[0.24, 0.29]` stochastic envelope.

## What must match, and what may vary

Data integrity is the primary reproduction requirement. The prepared corpus
text, row order, document-ID mapping, first 1,000 queries, qrels, encoder
semantics, fitted PCA, and graph parameters must match. The pipeline checks
these with row counts, whole-file hashes, array metadata, and independently
regenerated random samples before reporting MRR.

Dependency versions are pinned to make setup predictable, but modest timing
differences are not a data-integrity failure. Hardware, scheduling, and library
implementations can move the roughly 9–10 second PIR preparation measurement
or other wall-clock values. Judge reproduction by the integrity checks and
MRR, while retaining the raw report for performance comparisons.

## What computer do I need?

Use a 64-bit Ubuntu 22.04 Linux machine with:

- at least 40 GB of free disk space;
- at least 32 GB of RAM (64 GB is safer);
- 16 CPU threads for the paper's graph-build setting;
- preferably an NVIDIA GPU with at least 8 GB of memory and a recent driver.

The 8.45 GB corpus download is resumable. Embedding 3.2 million documents is
the longest stage and can take hours on a GPU or several days on CPU. The
paper used 16 threads for graph construction and one thread for the remaining
reported server computation.

## One-time Ubuntu setup

Copy and paste:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake curl ca-certificates gzip \
  libopenblas-dev liblapack-dev python3 python3-venv
```

The reproduction script installs private copies of Go 1.22.1, NGT 2.2.2, and
the pinned Python packages under the reproduction data directory. It does not
replace your system Go or NGT installation.

The pinned encoder environment requires Python 3.10 or 3.11. Ubuntu 22.04's
default Python 3.10 is the tested path.

If you use an NVIDIA GPU, verify that the driver works before starting:

```bash
nvidia-smi
```

## Run everything

From the root of the Pacmann repository:

```bash
./reproduction/msmarco/reproduce.sh
```

That is the complete command. It is safe to run it again after an interruption:
completed downloads are reused, document embedding resumes from a checkpoint,
and an existing graph is loaded instead of rebuilt.

Generated data goes under `reproduction-data/msmarco/`, which Git ignores.
Each search execution gets a new UTC-named directory under `runs/`. At the end,
the script prints the result directory and creates `latest-run` pointing to it.

Expected final output resembles:

```text
Ranked / Total = 633 / 1000
MRR@100: 0.266464
Expected stochastic range: [0.240, 0.290]
```

The exact hit count varies. The command exits with an error if MRR is outside
that range, rather than silently presenting a mismatched experiment.

## Run or repeat one stage

```bash
./reproduction/msmarco/reproduce.sh --stage download
./reproduction/msmarco/reproduce.sh --stage prepare
./reproduction/msmarco/reproduce.sh --stage embed
./reproduction/msmarco/reproduce.sh --stage search
./reproduction/msmarco/reproduce.sh --stage evaluate
```

Useful options:

```bash
# Use a different disk:
./reproduction/msmarco/reproduce.sh --work-dir /large-disk/pacmann-msmarco

# Reduce GPU memory use:
./reproduction/msmarco/reproduce.sh --batch-size 8

# Explicitly allow the very slow CPU encoder:
./reproduction/msmarco/reproduce.sh --device cpu

# Re-evaluate a named run:
PACMANN_RUN_ID=20260820T120000Z \
  ./reproduction/msmarco/reproduce.sh --stage evaluate
```

## Exact data preparation performed

The official document release has 3,213,835 rows with fields:

```text
docid<TAB>url<TAB>title<TAB>body
```

The historical paper input is reconstructed as follows:

1. preserve official row order and raw TSV quote characters;
2. exclude the 12,014 rows whose `body` field is empty;
3. concatenate `title + " " + body` and remove trailing whitespace;
4. keep the first 512 Unicode characters;
5. write `docid<TAB>text`.

This produces exactly 3,201,821 rows and SHA-256:

```text
8c1f99f1ddb68bd6e12d05d7f829e8635c200a29371e32412bf49aa315565e0c  msmarco-prefix.tsv
```

The first 1,000 official dev-query rows are used, in order. The official files
used here have these decompressed hashes:

```text
bb97b748dda44cf2352fb128efcbcc0ad157f7daaa572c2f5d4cdc3f0191e47b  msmarco-docdev-queries.tsv
28e2e5ccc17cd507875ded42c77434129694594d2cd2ec48f58ec1abc13cee43  msmarco-docdev-qrels.tsv
29f31149bbb653cbcba7544f12faca207de0abfb56e22eac7bf41b6b0ffa2fba  msmarco-queries-1000.tsv
```

The preparation stage refuses to continue if any count or hash differs.

## Embedding and PCA provenance

Documents and queries use:

```text
model: sentence-transformers/msmarco-distilbert-base-tas-b
revision: 996dfc6404137c6d89c7bf647a4bae62fdf8dd9a
pooling: CLS
maximum sequence length: 512
normalization: none
document/query prompts: none
encoder output: 768-dimensional float32
```

The original cuML `IncrementalPCA(n_components=192, batch_size=1024,
whiten=False)` was fitted sequentially over all 3,201,821 document embeddings.
To avoid requiring an obsolete RAPIDS/cuML environment, this repository ships
its portable fitted `components_` and `mean_` arrays in `assets/pca_192.npz`.
The transform is exactly:

```python
(embedding.astype(numpy.float64) - mean) @ components.T
```

PCA asset provenance:

```text
0fc7bcfd693498d08c9ccfe98f3cb2d9d78ae345e9d8777000700a75fd704e77  pca_192.npz
a33c1929169a437da9c514fdcf9490ab81b8761ec12c77908e3370890a5fac97  components raw bytes
1424e900221d9d002ed8d311ab61d32465fad0495b307b641e58cc527b3578b2  mean raw bytes
```

Before full document encoding, the pipeline automatically checks three
deterministic random document samples (32 rows each, seeds 1, 2024, and 1600)
and all 1,000 query vectors against the surviving paper artifacts. This makes a
data/model/PCA mismatch fail before the dominant preprocessing cost is paid.
After encoding, it repeats the check against the arrays actually written to
disk. The compressed validation asset is `assets/validation_reference.npz`
with SHA-256:

```text
3421d5fbbb4a3d17194e99fc44bcaea27564cffb115262254b6967e2f7907a39
```

The check fails if a document ID/text differs or if any generated coordinate
differs by more than `1e-5`. This catches incorrect corpus parsing, model
revision, pooling, prompts, normalization, row order, or PCA orientation before
the expensive graph build starts.

Small floating-point differences across GPU models and CUDA/PyTorch versions
are normal. Across the three audited 32-document samples, regenerated reduced
vectors differed from the historical vectors by at most `2.67e-6`.
The script records hashes of every newly generated array and graph in
`msmarco-dataset/artifact-manifest.json` so runs can be compared unambiguously.

For reference, the surviving exact historical artifacts have these hashes:

```text
63e9c13be1d8cbfeb92f6ba5cd9cff97120f1aa05f33f7db0991da6cf5e1ef48  reduced documents
ff9c383c59cad2f59624c708c73648b74ad7bfee83d1d0232008d59b16fc47c9  document IDs
da00b64ff727a4e9573ef0571e65350fa78c5e4d7fb68f83c0bc5392a9e6c541  reduced queries
335d6a415ce5f39ccfa3f7283174ec7c1e27f5075f1c8adbb8ef4d532a9e2bef  degree-32 graph
```

## Search parameters

The automated command records and uses:

```text
n=3,201,821  dimensions=192  degree=32  top-k=100
queries=1,000  rounds=20  parallel exploration=3  modeled RTT=50 ms
graph construction threads=16  random seed=1
```

The private-search report contains raw protocol accounting. In particular, it
reports 3,150 KB online communication per query. The paper table's 1.5 MB was
obtained by dividing this counter by two, but the implementation does not
perform or require that division. The reproducible aggregate value is therefore
3,150 KB (about 3.08 MiB) per query; do not divide it when comparing raw logs.

## If your result is outside the expected range

Do not change parameters until you compare `artifact-manifest.json`. Check, in
this order:

1. graph hash and shape `(3201821, 32)`;
2. document/query embedding hashes and row order;
3. document-ID mapping;
4. first-1,000 query hash and qrels hash;
5. model revision, lack of prompts/L2 normalization, and PCA asset hash.
