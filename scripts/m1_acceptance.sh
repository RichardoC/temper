#!/bin/sh
# M1 acceptance driver: end-to-end gen on bench/sort (K=4, steps=96) plus the
# steps sweep (K=2 at steps 64/96/128) that fixes the M1 default.
# Run: nohup scripts/m1_acceptance.sh > /tmp/m1_acceptance.log 2>&1 &
cd "$(dirname "$0")/.."
echo "=== acceptance: bench/sort K=4 steps=96 ==="
./bin/temper gen --task bench/sort -K 4 --steps 96 --canvas 384 --seed 11
echo "=== sweep: bench/sort K=2 steps=64 ==="
./bin/temper gen --task bench/sort -K 2 --steps 64 --canvas 384 --seed 21
echo "=== sweep: bench/sort K=2 steps=96 ==="
./bin/temper gen --task bench/sort -K 2 --steps 96 --canvas 384 --seed 22
echo "=== sweep: bench/sort K=2 steps=128 ==="
./bin/temper gen --task bench/sort -K 2 --steps 128 --canvas 384 --seed 23
echo "=== acceptance driver done ==="
