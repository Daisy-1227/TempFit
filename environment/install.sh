set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:?Usage: bash environment/install.sh /new/venv /path/to/wheelhouse}"
wheelhouse="${2:?Provide the verified Linux CUDA 11.8 wheelhouse}"
if [[ -e "$target" ]]; then
    echo "Environment already exists: $target" >&2
    exit 2
fi
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
python -c 'import sys, platform; assert sys.version_info[:2] == (3, 10); assert sys.platform == "linux" and platform.machine() == "x86_64"'
python -m venv "$target"
"$target/bin/python" -m pip install --no-index --no-cache-dir --require-hashes --find-links "$wheelhouse" -r "$root/environment/requirements.txt"
"$target/bin/python" "$root/environment/apply_patch.py"
"$target/bin/python" -m pip check
cd "$root"
"$target/bin/python" -m util.doctor --gpu --output "$target/DOCTOR.json"
