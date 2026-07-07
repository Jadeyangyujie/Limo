#!/usr/bin/env bash

# Source after `conda activate limo`:
#   source scripts/limo_env.sh
#
# This file is intentionally repo-local. It derives paths from its own location
# so the same conda env can be reused with different Limo checkouts.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced: source scripts/limo_env.sh" >&2
  exit 1
fi

_limo_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export LIMO_REPO_ROOT="$(cd -- "${_limo_script_dir}/.." >/dev/null 2>&1 && pwd)"

export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1

export CACHE_DIR="${LIMO_REPO_ROOT}/cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

case ":${PYTHONPATH:-}:" in
  *":${LIMO_REPO_ROOT}:"*) ;;
  *) export PYTHONPATH="${LIMO_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

echo "Configured LIMO repo: ${LIMO_REPO_ROOT}"

unset _limo_script_dir
