# shared by the job scripts: python env on node local disk, built once per node
SCR=""
for base in /scratch /ssd_scratch /tmp; do
    if [ -n "$(df -P $base 2>/dev/null | awk 'NR == 2 && $4 > 20 * 1024 * 1024')" ]; then SCR=$base/$USER/moe; break; fi
done
[ -n "$SCR" ] || { echo "no local disk"; exit 3; }
mkdir -p $SCR
export UV_CACHE_DIR=$SCR/cache/uv UV_PYTHON_INSTALL_DIR=$SCR/python
# several seeds can land on one node at once: only one of them builds the env
(
    flock 9
    [ -x $SCR/tools/bin/uv ] || { python3 -m venv $SCR/tools && $SCR/tools/bin/pip install -q uv; }
    [ -x $SCR/env/bin/python ] || $SCR/tools/bin/uv venv -q --python 3.12 $SCR/env
    $SCR/tools/bin/uv pip install -q --python $SCR/env/bin/python --torch-backend=cu126 \
        torch numpy pandas scipy yfinance ta pyyaml torchdiffeq pytest
) 9>$SCR/env.lock
PY=$SCR/env/bin/python
