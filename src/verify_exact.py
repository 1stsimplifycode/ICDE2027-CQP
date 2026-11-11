"""Exact-arithmetic audit of Theorem 2 (value-indexed DP) with Fractions, including adversarial boundary cases:
zero-cost slots, zero budget, empty decision sets, identical and nested ranges, single-slot ranges, costs with
large/small magnitudes, and source-level costs (slot cost = sum of its uncertain items' costs; we check that
probing sub-slot item sets never beats slot-atomic plans by brute force over ITEMS)."""
import itertools, json, random
from fractions import Fraction as Fr

def dp_value(c, ranges):
    m = len(c); nd = len(ranges); INF = None
    G = [[0] * (m + 2) for _ in range(m + 2)]
    for i in range(m + 1):
        for j in range(i + 1, m + 2):
            G[i][j] = sum(1 for p, q in ranges if i < p and q < j)
    pc = [Fr(0)]
    for x in c: pc.append(pc[-1] + x)
    best = [[INF] * (nd + 1) for _ in range(m + 2)]; best[0][0] = Fr(0)
    for j in range(1, m + 2):
        for i in range(j):
            cost = pc[j - 1] - pc[i]; g = G[i][j]
            for v in range(g, nd + 1):
                b = best[i][v - g]
                if b is not None and (best[j][v] is None or b + cost < best[j][v]):
                    best[j][v] = b + cost
    out = best[m + 1][:]
    for v in range(nd - 1, -1, -1):
        if out[v + 1] is not None and (out[v] is None or out[v + 1] < out[v]): out[v] = out[v + 1]
    return out

def answer(g, B):
    return max([v for v, x in enumerate(g) if x is not None and x <= B] or [0])

def brute_items(item_cost, slot_of, ranges, m, B):
    """Brute force over individual ITEMS (source-level probes). A range is guaranteed iff all items in its slots probed."""
    n = len(item_cost); best = 0
    for r in range(n + 1):
        for P in itertools.combinations(range(n), r):
            if sum(item_cost[i] for i in P) > B: continue
            Ps = set(P)
            full = {s for s in range(1, m + 1) if all(i in Ps for i in range(n) if slot_of[i] == s)}
            best = max(best, sum(all(x in full for x in range(p, q + 1)) for p, q in ranges))
    return best

random.seed(0); n = bad = 0; cases = {"zero_cost": 0, "zero_budget": 0, "empty": 0, "identical": 0, "nested": 0}
for t in range(1500):
    m = random.randint(1, 5)
    items_per = [random.randint(1, 2) for _ in range(m)]
    item_cost, slot_of = [], []
    for s in range(1, m + 1):
        for _ in range(items_per[s - 1]):
            item_cost.append(random.choice([Fr(0), Fr(1, 3), Fr(1), Fr(7, 2), Fr(10**6), Fr(1, 10**6)])); slot_of.append(s)
    c = [sum(item_cost[i] for i in range(len(item_cost)) if slot_of[i] == s) for s in range(1, m + 1)]
    R = []
    for _ in range(random.randint(0, 6)):
        p = random.randint(1, m); q = random.randint(p, m); R.append((p, q))
    if R and random.random() < 0.3: R.append(R[0]); cases["identical"] += 1
    if not R: cases["empty"] += 1
    if any(a[0] <= b[0] and b[1] <= a[1] and a != b for a in R for b in R): cases["nested"] += 1
    if any(x == 0 for x in item_cost): cases["zero_cost"] += 1
    B = random.choice([Fr(0), sum(c, Fr(0)) * Fr(random.randint(0, 10), 10)])
    if B == 0: cases["zero_budget"] += 1
    n += 1; bad += int(answer(dp_value(c, R), B) != brute_items(item_cost, slot_of, R, m, B))
out = {"instances": n, "mismatches": bad, "boundary_case_counts": cases, "arithmetic": "exact (fractions.Fraction)",
       "brute_force_granularity": "individual source items (verifies slot atomicity is without loss of optimality)"}
print(out); json.dump(out, open(r"C:\Users\mailt\Downloads\ICDE-2027\results\verify_exact.json", "w"), indent=1)
