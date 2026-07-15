"""
Complete the qlib-store extension: update SP500 constituents' listing END-DATE in
instruments/all.txt from the old boundary (2020-11-10) to the calendar end.

extend_qlib_data.py appended price bins + calendar dates to 2026 for the SP500
constituents but never updated their entries in instruments/all.txt, which still
say they delisted 2020-11-10. qlib's backtest exchange defaults to the `all`
universe, so in the 2023-2026 window it treats every constituent as delisted →
the strategy can trade nothing (100% cash, identical portfolio every loop).
sp500.txt already has these open-ended (2099-12-31), which is why market="sp500"
feature queries worked but the backtest did not.

This updates only tickers that are active in sp500.txt AND currently end at the
boundary in all.txt, setting their end-date to the calendar's last day.
"""
from pathlib import Path

QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "us_data"
BOUNDARY = "2020-11-10"

cal = [l.strip() for l in open(QLIB_DIR/"calendars"/"day.txt") if l.strip()]
cal_end = cal[-1]
print(f"calendar end: {cal_end}")

# active SP500 constituents (open-ended in sp500.txt)
active = set()
for ln in open(QLIB_DIR/"instruments"/"sp500.txt"):
    p = ln.split("\t")
    if len(p) >= 3 and "2099-12-31" in ln:
        active.add(p[0].strip().upper())
print(f"active SP500 constituents in sp500.txt: {len(active)}")

all_path = QLIB_DIR/"instruments"/"all.txt"
lines = all_path.read_text().splitlines()
updated = 0
samples = []
for i, ln in enumerate(lines):
    p = ln.split("\t")
    if len(p) < 3:
        continue
    tkr, start, end = p[0].strip(), p[1].strip(), p[2].strip()
    if tkr.upper() in active and end == BOUNDARY:
        p[2] = cal_end
        lines[i] = "\t".join(p)
        updated += 1
        if len(samples) < 6:
            samples.append(f"{tkr}: {end} -> {cal_end}")

print(f"updated {updated} constituent listings in all.txt")
for s in samples:
    print("  ", s)

# backup then write
bak = all_path.with_suffix(".txt.bak")
if not bak.exists():
    bak.write_text(all_path.read_text())
    print(f"backup written: {bak.name}")
all_path.write_text("\n".join(lines) + "\n")
print("all.txt updated.")
