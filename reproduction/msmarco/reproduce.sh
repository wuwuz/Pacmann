#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

stage="all"
work_dir="$repo_dir/reproduction-data/msmarco"
device="auto"
batch_size=32
graph_threads=16
seed=1
rebuild_graph=false
run_id=${PACMANN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

usage() {
  cat <<'EOF'
Reproduce the paper's MS-MARCO experiment from the official dataset.

Usage:
  ./reproduction/msmarco/reproduce.sh [options]

Options:
  --stage STAGE         all, download, prepare, embed, search, or evaluate
  --work-dir PATH       generated data directory (default: reproduction-data/msmarco)
  --device DEVICE       auto, cuda, or cpu (default: auto)
  --batch-size NUMBER   encoder batch size (default: 32)
  --graph-threads N     graph construction threads (default: 16)
  --seed N              graph-search random seed (default: 1)
  --rebuild-graph       force a cold graph and NGT-index rebuild
  --run-id ID           name for this run (default: current UTC timestamp)
  --help                show this message

The script is resumable. Re-running it skips completed downloads and artifacts.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage=$2; shift 2 ;;
    --work-dir) work_dir=$2; shift 2 ;;
    --device) device=$2; shift 2 ;;
    --batch-size) batch_size=$2; shift 2 ;;
    --graph-threads) graph_threads=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --rebuild-graph) rebuild_graph=true; shift ;;
    --run-id) run_id=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$stage" in
  all|download|prepare|embed|search|evaluate) ;;
  *) echo "Invalid stage: $stage" >&2; usage >&2; exit 2 ;;
esac
case "$device" in
  auto|cuda|cpu) ;;
  *) echo "Invalid device: $device" >&2; exit 2 ;;
esac
if [[ $rebuild_graph == true && $stage != all && $stage != search ]]; then
  echo "--rebuild-graph is valid only with --stage all or --stage search" >&2
  exit 2
fi

mkdir -p "$work_dir"
work_dir=$(cd "$work_dir" && pwd)
downloads_dir="$work_dir/downloads"
dataset_dir="$work_dir/msmarco-dataset"
model_cache="$work_dir/model-cache"
venv_dir="$work_dir/venv"
run_dir="$work_dir/runs/$run_id"
tools_dir="$work_dir/tools"
go_dir="$tools_dir/go-1.22.1"
ngt_dir="$tools_dir/ngt-2.2.2"
graph_path="$dataset_dir/msmarco_embeddings_3201821_192_32_graph.npy"
ngt_path="$dataset_dir/msmarco_embeddings_3201821_192_32.ngt"

documents_url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"
queries_url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz"
qrels_url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"

download_file() {
  local url=$1
  local destination=$2
  local expected_size=$3
  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]] && [[ $(stat -c '%s' "$destination") -eq $expected_size ]]; then
    echo "Already downloaded: $destination"
    return
  fi
  echo "Downloading $url"
  echo "This download is resumable; rerun the same command if it is interrupted."
  curl --location --fail --retry 10 --retry-all-errors --continue-at - \
    --output "$destination" "$url"
  local actual_size
  actual_size=$(stat -c '%s' "$destination")
  if [[ $actual_size -ne $expected_size ]]; then
    echo "Wrong download size for $destination: got $actual_size, expected $expected_size" >&2
    exit 1
  fi
}

download_inputs() {
  mkdir -p "$downloads_dir"
  download_file "$documents_url" "$downloads_dir/msmarco-docs.tsv.gz" 8446274598
  download_file "$queries_url" "$downloads_dir/msmarco-docdev-queries.tsv.gz" 91837
  download_file "$qrels_url" "$downloads_dir/msmarco-docdev-qrels.tsv.gz" 38553
}

prepare_inputs() {
  mkdir -p "$dataset_dir"
  python3 "$script_dir/prepare_corpus.py" \
    --documents-gz "$downloads_dir/msmarco-docs.tsv.gz" \
    --queries-gz "$downloads_dir/msmarco-docdev-queries.tsv.gz" \
    --qrels-gz "$downloads_dir/msmarco-docdev-qrels.tsv.gz" \
    --output-dir "$dataset_dir"
}

prepare_python() {
  python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 11), "Pacmann reproduction requires Python 3.10 or 3.11"'
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    python3 -m venv "$venv_dir"
  fi
  local requirements_hash
  requirements_hash=$(sha256sum "$script_dir/requirements.txt" | cut -d' ' -f1)
  if [[ ! -f "$venv_dir/.pacmann-requirements-$requirements_hash" ]]; then
    "$venv_dir/bin/python" -m pip install --upgrade pip
    "$venv_dir/bin/python" -m pip install --requirement "$script_dir/requirements.txt"
    touch "$venv_dir/.pacmann-requirements-$requirements_hash"
  fi
}

validate_generated_inputs() {
  "$venv_dir/bin/python" "$script_dir/verify_artifacts.py" --dataset-dir "$dataset_dir"
  "$venv_dir/bin/python" "$script_dir/validate_samples.py" \
    --corpus "$dataset_dir/msmarco-prefix.tsv" \
    --queries "$dataset_dir/msmarco-queries-1000.tsv" \
    --generated-documents "$dataset_dir/msmarco_embeddings.npy" \
    --generated-queries "$dataset_dir/msmarco_queries.npy" \
    --pca "$script_dir/assets/pca_192.npz" \
    --reference "$script_dir/assets/validation_reference.npz" \
    --model-cache "$model_cache" \
    --output "$dataset_dir/sample-validation.json" \
    --batch-size "$batch_size" \
    --device "$device"
}

embed_inputs() {
  prepare_python
  echo "Running the fast integrity preflight before full document encoding"
  "$venv_dir/bin/python" "$script_dir/validate_samples.py" \
    --corpus "$dataset_dir/msmarco-prefix.tsv" \
    --queries "$dataset_dir/msmarco-queries-1000.tsv" \
    --pca "$script_dir/assets/pca_192.npz" \
    --reference "$script_dir/assets/validation_reference.npz" \
    --model-cache "$model_cache" \
    --output "$dataset_dir/preflight-sample-validation.json" \
    --batch-size "$batch_size" \
    --device "$device"
  "$venv_dir/bin/python" "$script_dir/embed_and_reduce.py" \
    --corpus "$dataset_dir/msmarco-prefix.tsv" \
    --queries "$dataset_dir/msmarco-queries-1000.tsv" \
    --pca "$script_dir/assets/pca_192.npz" \
    --output-dir "$dataset_dir" \
    --model-cache "$model_cache" \
    --batch-size "$batch_size" \
    --device "$device"
  validate_generated_inputs
}

prepare_native() {
  "$script_dir/bootstrap_native.sh" "$work_dir"
  export PATH="$go_dir/bin:$PATH"
  export GOCACHE="$work_dir/go-cache"
  export GOPATH="$work_dir/go-path"
  export GOENV=off
  export GOTOOLCHAIN=local
  export CGO_CFLAGS="-I$ngt_dir/include ${CGO_CFLAGS:-}"
  export CGO_LDFLAGS="-L$ngt_dir/lib -Wl,-rpath,$ngt_dir/lib ${CGO_LDFLAGS:-}"
  export LD_LIBRARY_PATH="$ngt_dir/lib:${LD_LIBRARY_PATH:-}"
  mkdir -p "$GOCACHE" "$GOPATH"
}

prepare_graph_rebuild() {
  if [[ $rebuild_graph != true ]]; then
    return
  fi

  if [[ ! -e "$graph_path" && ! -L "$graph_path" && ! -e "$ngt_path" && ! -L "$ngt_path" ]]; then
    echo "No existing graph or NGT index found; proceeding with a cold graph build."
    return
  fi

  local backup_dir="$dataset_dir/graph-backups/$run_id"
  if [[ -e "$backup_dir" || -L "$backup_dir" ]]; then
    echo "Graph backup path already exists: $backup_dir" >&2
    echo "Choose a different --run-id before forcing another rebuild." >&2
    exit 1
  fi
  mkdir -p "$backup_dir"

  if [[ -e "$graph_path" || -L "$graph_path" ]]; then
    mv "$graph_path" "$backup_dir/"
  fi
  if [[ -e "$ngt_path" || -L "$ngt_path" ]]; then
    mv "$ngt_path" "$backup_dir/"
  fi
  echo "Moved the previous graph artifacts to $backup_dir"
  echo "The search stage will rebuild both the NGT index and degree-32 graph."
}

run_search() {
  if [[ $stage == search ]]; then
    prepare_python
    echo "Validating existing generated inputs before graph construction/search"
    validate_generated_inputs
  fi
  prepare_graph_rebuild
  prepare_native
  mkdir -p "$work_dir/bin" "$run_dir"
  echo "Building the Pacmann executable"
  (cd "$repo_dir" && "$go_dir/bin/go" build -o "$work_dir/bin/private-search" private-search.go)
  echo "Running private MS-MARCO search. The graph is built automatically on the first run."
  PACMANN_GRAPH_THREADS="$graph_threads" "$work_dir/bin/private-search" \
    -n 3201821 -d 192 -m 32 -k 100 -q 1000 \
    -input "$dataset_dir/msmarco_embeddings.npy" \
    -graph "$graph_path" \
    -query "$dataset_dir/msmarco_queries.npy" \
    -step 20 -parallel 3 -rtt 50 -seed "$seed" \
    -output "$run_dir/private-search-output.txt" \
    -report "$run_dir/private-search-report.txt" \
    2>&1 | tee "$run_dir/private-search.log"
  prepare_python
  "$venv_dir/bin/python" "$script_dir/verify_artifacts.py" \
    --dataset-dir "$dataset_dir" --include-graph
  ln -sfn "$run_dir" "$work_dir/latest-run"
}

evaluate_run() {
  prepare_python
  if [[ ! -f "$run_dir/private-search-output.txt" ]] && [[ -L "$work_dir/latest-run" ]]; then
    run_dir=$(readlink -f "$work_dir/latest-run")
  fi
  if [[ ! -f "$run_dir/private-search-output.txt" ]]; then
    echo "No raw result found in $run_dir. Run the search stage first." >&2
    exit 1
  fi
  "$venv_dir/bin/python" "$script_dir/evaluate.py" \
    --raw-results "$run_dir/private-search-output.txt" \
    --docids "$dataset_dir/msmarco_docid_permuted.npy" \
    --queries "$dataset_dir/msmarco-queries-1000.tsv" \
    --qrels "$dataset_dir/msmarco-docdev-qrels.tsv" \
    --output-docids "$run_dir/private-search-output-docids.txt" \
    --output-json "$run_dir/evaluation.json" \
    | tee "$run_dir/evaluation.txt"
  echo "Complete reproduction output: $run_dir"
}

case "$stage" in
  download) download_inputs ;;
  prepare) prepare_inputs ;;
  embed) embed_inputs ;;
  search) run_search ;;
  evaluate) evaluate_run ;;
  all)
    download_inputs
    prepare_inputs
    embed_inputs
    run_search
    evaluate_run
    ;;
esac
