#!/usr/bin/env bash
# Repeatable check of a CLEAN start: a copy of the repository without .venv, models, data, reports → docker compose build →
# wait for «healthy» (finite timeout) → scripts/smoke_test.py from the host → the same image started WITHOUT a network and checked from inside →
# a JSON record in reports/clean_run/ (image id, CPU architecture, size, what was checked, platform).
#
# The BUILD needs the network (PyPI, Hugging Face). The check without a network proves that the RUNNING service does not need it;
# it does not prove that the build works offline. Other platforms stay «not checked» until this script is run on them.
#
# Run:  scripts/clean_run_check.sh            (APP_PORT=8766 by default; CLEAN_TIMEOUT=600 seconds to become healthy)
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
PORT="${APP_PORT:-8766}"
TIMEOUT="${CLEAN_TIMEOUT:-600}"
PROJECT="analogclean${STAMP}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/analog-search-clean.XXXXXX")"
OUT="$SRC/reports/clean_run"
mkdir -p "$OUT"
status=0

cleanup() {
  (cd "$WORK" && docker compose -p "$PROJECT" down --remove-orphans >/dev/null 2>&1) || true
  docker rm -f "${PROJECT}-offline" >/dev/null 2>&1 || true
  rm -rf "${WORK:?}"
}
trap cleanup EXIT

echo "== чистая копия: $WORK"
rsync -a --exclude .venv --exclude models --exclude data --exclude reports --exclude .git --exclude __pycache__ \
      --exclude .pytest_cache --exclude .DS_Store --exclude notebooks --exclude Questions --exclude dataset ./ "$WORK"/ 2>/dev/null || \
rsync -a --exclude .venv --exclude models --exclude data --exclude reports --exclude .git --exclude __pycache__ \
      --exclude .pytest_cache --exclude .DS_Store --exclude notebooks --exclude Questions --exclude dataset "$SRC"/ "$WORK"/

cd "$WORK"
echo "== сборка и запуск (порт $PORT)"
APP_PORT="$PORT" docker compose -p "$PROJECT" up --build -d

echo "== ожидание healthy (не более ${TIMEOUT} с)"
cid="$(APP_PORT="$PORT" docker compose -p "$PROJECT" ps -q analog-search)"
deadline=$(( $(date +%s) + TIMEOUT ))
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null)" = "healthy" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "НЕ стал healthy за ${TIMEOUT} с"; docker logs --tail 30 "$cid" || true; status=1; break; fi
  sleep 3
done

HOST_REPORT="$OUT/${STAMP}-host.json"
OFFLINE_REPORT="$OUT/${STAMP}-offline.json"
host_rc=1; offline_rc=1
if [ "$status" -eq 0 ]; then
  echo "== smoke с хоста"
  python3 scripts/smoke_test.py --base-url "http://localhost:${PORT}" --expected eval/smoke_expected.json --report "$HOST_REPORT" && host_rc=0 || host_rc=$?

  echo "== тот же образ без сети: проверка изнутри"
  image="$(docker inspect -f '{{.Image}}' "$cid")"
  docker run -d --network none --name "${PROJECT}-offline" "$image" >/dev/null
  docker cp eval/smoke_expected.json "${PROJECT}-offline":/tmp/smoke_expected.json
  docker exec "${PROJECT}-offline" python /app/scripts/smoke_test.py --base-url http://127.0.0.1:8000 --expected /tmp/smoke_expected.json \
      --wait 120 --report /tmp/offline.json && offline_rc=0 || offline_rc=$?
  docker cp "${PROJECT}-offline":/tmp/offline.json "$OFFLINE_REPORT" 2>/dev/null || true
fi

image_id="$(docker inspect -f '{{.Image}}' "$cid" 2>/dev/null || echo unknown)"
python3 - "$OUT/${STAMP}.json" "$image_id" "$host_rc" "$offline_rc" "$status" "$HOST_REPORT" "$OFFLINE_REPORT" <<'PY'
import json, platform, subprocess, sys
out, image_id, host_rc, offline_rc, status, host_report, offline_report = sys.argv[1:8]
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, check=True).stdout.strip()
    except Exception: return None
info = {"image_id": image_id, "architecture": sh("docker", "image", "inspect", "-f", "{{.Architecture}}", image_id),
        "os": sh("docker", "image", "inspect", "-f", "{{.Os}}", image_id), "size_bytes": int(sh("docker", "image", "inspect", "-f", "{{.Size}}", image_id) or 0),
        "docker": sh("docker", "--version"), "host_platform": platform.platform(), "host_machine": platform.machine(),
        "became_healthy": status == "0", "smoke_from_host_exit": int(host_rc), "smoke_offline_runtime_exit": int(offline_rc),
        "reports": {"host": host_report, "offline_runtime": offline_report},
        "not_checked": ["другая ОС и архитектура процессора", "сборка образа без доступа к PyPI и Hugging Face"]}
open(out, "w", encoding="utf-8").write(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
print(json.dumps(info, ensure_ascii=False, indent=1))
PY
[ "$status" -eq 0 ] && [ "$host_rc" -eq 0 ] && [ "$offline_rc" -eq 0 ]
