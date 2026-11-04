set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:?Usage: bash environment/install.sh --conda [wheelhouse] | /new/venv [wheelhouse]}"
wheelhouse="${2:-${XDG_CACHE_HOME:-$HOME/.cache}/tempfit/env-cu118-py310-v1}"
use_conda=false
if [[ "$target" == "--conda" ]]; then
    use_conda=true
    target="${CONDA_PREFIX:?Activate a Python 3.10 Conda environment first}"
    python_bin="$target/bin/python"
    if [[ ! -d "$target/conda-meta" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
        echo "Create and activate a project Conda environment: conda create -n tempfit python=3.10 pip" >&2
        exit 2
    fi
else
    python_bin="${PYTHON:-python3.10}"
    if [[ -e "$target" ]]; then
        echo "Environment already exists: $target" >&2
        exit 2
    fi
fi
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$python_bin" -c 'import sys, platform; assert sys.version_info[:2] == (3, 10); assert sys.platform == "linux" and platform.machine() == "x86_64"'
if [[ $# -lt 2 ]]; then
    "$python_bin" "$root/environment/download.py" --destination "$wheelhouse"
fi
if [[ "$use_conda" == false ]]; then
    "$python_bin" -m venv "$target"
fi
"$target/bin/python" -m pip install --no-index --no-cache-dir --only-binary=:all: --require-hashes --find-links "$wheelhouse" -r "$root/environment/requirements.txt"
"$target/bin/python" "$root/environment/apply_patch.py"
"$target/bin/python" -m pip check
"$target/bin/python" "$root/environment/smoke_test.py" --output "$target/ENVIRONMENT_CHECK.json"
