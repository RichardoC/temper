#!/usr/bin/env bash
# M0 cycle measurement: incremental rebuild cost of the emitted demo crate
# (touch candidate.rs + cargo build --offline), measured with hyperfine in
# both sandbox modes: seatbelt (default) and TEMPER_SANDBOX=none.
#
# Prereqs: repo committed rust-template (warm target/), hyperfine on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/opt/homebrew/opt/rustup/bin:/opt/homebrew/bin:$PATH"

cd "$REPO_ROOT"

# (Re)emit the demo crate and prime the hardlink cache so the measured
# cycle is the steady-state incremental build, not a cold one.
python3 - <<'PY'
from pathlib import Path
from temper.harness.emit import emit
from temper.harness.cache import prime_target_cache
from temper.harness.sandbox import write_seatbelt_profile

props = Path("rust-template/src/properties.rs").read_text()
cand = Path("rust-template/src/candidate.rs").read_text()
e = emit(cand, props)
links = prime_target_cache(e.workdir)
profile = write_seatbelt_profile(e.workdir)
(e.workdir / ".tmp").mkdir(exist_ok=True)
print(f"emitted {e.workdir} (cache links: {links}, profile: {profile})")
PY

WORKDIR="$REPO_ROOT/temper/.work/gen"
PROFILE="$REPO_ROOT/temper/.work/seatbelt.sb"

# One warm-up build per mode (outside hyperfine) so dep artifacts exist.
(cd "$WORKDIR" && TMPDIR="$PWD/.tmp" cargo build --offline -q)
(cd "$WORKDIR" && touch src/candidate.rs && \
  TMPDIR="$PWD/.tmp" sandbox-exec -f "$PROFILE" -D "WORKDIR=$WORKDIR" cargo build --offline -q)

cd "$WORKDIR"
hyperfine --warmup 3 --style basic --shell=sh \
  -n "seatbelt" \
  "touch src/candidate.rs && TMPDIR='$PWD/.tmp' sandbox-exec -f '$PROFILE' -D 'WORKDIR=$WORKDIR' cargo build --offline" \
  -n "none" \
  "touch src/candidate.rs && TMPDIR='$PWD/.tmp' cargo build --offline"
