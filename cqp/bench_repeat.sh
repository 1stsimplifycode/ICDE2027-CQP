# Repeated equivalent-functionality benchmark: 5 reps x 3 modes (60 days), mode order rotated per rep (Latin square, seed-free),
# each run in a fresh process with a fresh in-memory database (cold start; no warm-up run).
P=${PY:-python}
ORDERS=("inc_scan inc full" "inc full inc_scan" "full inc_scan inc" "inc_scan inc full" "inc full inc_scan")
for r in ${REPS:-0 1 2 3 4}; do for m in ${ORDERS[$r]}; do REP=$r $P cqp/bench.py 60 $m > results/log_rep${r}_$m.txt 2>&1; echo "rep $r $m exit $?" >> results/bench_repeat_log.txt; done; done
echo DONE >> results/bench_repeat_log.txt
