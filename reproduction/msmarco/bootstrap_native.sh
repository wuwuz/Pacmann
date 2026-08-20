#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 WORK_DIRECTORY" >&2
  exit 2
fi

work_dir=$(cd "$1" && pwd)
tools_dir="$work_dir/tools"
downloads_dir="$tools_dir/downloads"
go_dir="$tools_dir/go-1.22.1"
ngt_dir="$tools_dir/ngt-2.2.2"
go_marker="$go_dir/.pacmann-install-complete"
ngt_marker="$ngt_dir/.pacmann-install-complete"

go_url="https://go.dev/dl/go1.22.1.linux-amd64.tar.gz"
go_sha256="aab8e15785c997ae20f9c88422ee35d962c4562212bb0f879d052a35c8307c7f"
ngt_url="https://github.com/yahoojapan/NGT/archive/refs/tags/v2.2.2.tar.gz"
ngt_sha256="cad1d2dfd58f9267580e2de0c9617c312a3ca082d1ce3e5f82aaa54ae5bf9470"

for command in curl sha256sum tar cmake c++; do
  if ! command -v "$command" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Missing a system build tool. On Ubuntu, run this once and retry:

  sudo apt-get update
  sudo apt-get install -y build-essential cmake curl ca-certificates gzip libopenblas-dev liblapack-dev python3 python3-venv
EOF
    exit 1
  fi
done

mkdir -p "$downloads_dir"

download_and_check() {
  local url=$1
  local destination=$2
  local expected_sha=$3
  if [[ ! -f "$destination" ]]; then
    echo "Downloading $url"
    curl --location --fail --retry 10 --retry-all-errors --output "$destination" "$url"
  fi
  echo "$expected_sha  $destination" | sha256sum --check --status || {
    echo "Checksum failed for $destination. Move that file away and retry." >&2
    exit 1
  }
}

if [[ ! -f "$go_marker" ]]; then
  go_archive="$downloads_dir/go1.22.1.linux-amd64.tar.gz"
  download_and_check "$go_url" "$go_archive" "$go_sha256"
  go_staging=$(mktemp -d "$tools_dir/.go-1.22.1-install.XXXXXX")
  tar -xzf "$go_archive" --strip-components=1 -C "$go_staging"
  if [[ ! -x "$go_staging/bin/go" ]] || \
     [[ ! -f "$go_staging/src/encoding/binary/binary.go" ]]; then
    echo "The verified Go archive did not produce a complete toolchain." >&2
    exit 1
  fi
  if [[ -e "$go_dir" ]]; then
    mv "$go_dir" "$go_dir.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mv "$go_staging" "$go_dir"
  touch "$go_marker"
fi

if [[ ! -f "$ngt_marker" ]]; then
  ngt_archive="$downloads_dir/NGT-v2.2.2.tar.gz"
  ngt_source="$tools_dir/NGT-2.2.2-source"
  ngt_build="$tools_dir/NGT-2.2.2-build"
  download_and_check "$ngt_url" "$ngt_archive" "$ngt_sha256"
  if [[ ! -d "$ngt_source" ]]; then
    mkdir -p "$ngt_source"
    tar -xzf "$ngt_archive" --strip-components=1 -C "$ngt_source"
  fi
  cmake -S "$ngt_source" -B "$ngt_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_PREFIX="$ngt_dir"
  cmake --build "$ngt_build" --parallel 16
  cmake --install "$ngt_build"
  if [[ ! -f "$ngt_dir/lib/libngt.so" ]] || \
     [[ ! -f "$ngt_dir/include/NGT/Index.h" ]]; then
    echo "NGT installation is incomplete." >&2
    exit 1
  fi
  touch "$ngt_marker"
fi

echo "Native tools are ready:"
echo "  Go:  $go_dir/bin/go"
echo "  NGT: $ngt_dir"
