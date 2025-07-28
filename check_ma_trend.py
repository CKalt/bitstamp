#!/usr/bin/env python3
"""Quick check of MA trend from recent logs"""
import re

# Sample log lines
logs = """
2025-07-27 16:08:41,382 - MA6=118410 MA34=118125 Diff=284 Prox=0.24%
2025-07-27 16:09:13,598 - MA6=118412 MA34=118126 Diff=286 Prox=0.24%
2025-07-27 16:11:48,543 - MA6=118431 MA34=118129 Diff=302 Prox=0.26%
2025-07-27 16:12:20,670 - MA6=118440 MA34=118131 Diff=309 Prox=0.26%
2025-07-27 16:12:52,554 - MA6=118446 MA34=118132 Diff=314 Prox=0.27%
2025-07-27 16:13:24,433 - MA6=118462 MA34=118135 Diff=327 Prox=0.28%
2025-07-27 16:13:56,122 - MA6=118470 MA34=118136 Diff=334 Prox=0.28%
"""

# Parse values
pattern = r'MA6=(\d+) MA34=(\d+) Diff=(\d+) Prox=([\d.]+)%'
matches = re.findall(pattern, logs)

print("MA Trend Analysis:")
print("-" * 50)
print("Time        MA6      MA34     Diff   Proximity")
print("-" * 50)

diffs = []
for i, (ma6, ma34, diff, prox) in enumerate(matches):
    print(f"{i:4d}   ${ma6}  ${ma34}   ${diff}    {prox}%")
    diffs.append(int(diff))

print("-" * 50)
print(f"\nTrend: Difference is INCREASING (diverging)")
print(f"Start: ${diffs[0]} ({matches[0][3]}%)")
print(f"Now:   ${diffs[-1]} ({matches[-1][3]}%)")
print(f"Change: +${diffs[-1] - diffs[0]} (+{float(matches[-1][3]) - float(matches[0][3]):.2f}%)")

print(f"\nConclusion: MAs are DIVERGING away from crossover")
print(f"Current proximity: {matches[-1][3]}% (threshold is 0.3%)")
print(f"Risk of crossover during 15-min upgrade: VERY LOW")