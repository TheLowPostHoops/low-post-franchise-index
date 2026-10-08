"""Refresh the data from Basketball Reference, re-run the model, rebuild the dashboard.

First time only:   pip3 install requests beautifulsoup4 lxml pandas numpy matplotlib
Run:               open this file in IDLE and press F5   (takes about 6 minutes; it waits between page loads on purpose)
Options (Terminal): python3 update.py --full     re-download every season (about 1.5 hours)
                    python3 update.py --push     also commit and push to GitHub after building (needs a git repo here)

What it refreshes: this season and last season of player stats, team ratings, standings, playoffs and
transactions, the last two drafts, and every team's executive history.
What it does NOT refresh: salaries (nba_salaries_master.csv stops at 2024) and the penalties list
(edit data/penalties.csv by hand when the league announces a new one).
"""
import os, re, sys, json, time, datetime, subprocess
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

BASE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(BASE, "data")
HOST = "https://www.basketball-reference.com"
DELAY = 4.0      # seconds between page loads; Basketball Reference asks for 20 per minute or fewer
FULL = "--full" in sys.argv
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"}
S = requests.Session(); S.headers.update(UA)

def get(path):
    time.sleep(DELAY)
    r = S.get(HOST + path, timeout=60)
    if r.status_code == 429: raise SystemExit("Basketball Reference says slow down (429). Wait 15 minutes and run again.")
    if r.status_code == 403: raise SystemExit("Basketball Reference blocked this request (403). Open the site in your browser, then run again; or use the browser fallback described in the README.")
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def rows_of(t): return [tr for tr in t.select("tbody tr") if "thead" not in (tr.get("class") or []) and tr.select_one("[data-stat]")]
def cells(tr):
    o = {c["data-stat"]: c.get_text(strip=True) for c in tr.select("[data-stat]")}
    a = tr.select_one('[data-stat="name_display"] a, [data-stat="player"] a')
    if a and a.get("href"):
        m = re.search(r"/players/\w/(\w+)\.html", a["href"])
        if m: o["pid"] = m.group(1)
    return o

now = datetime.date.today()
season_end = now.year + 1 if now.month >= 10 else now.year          # season ending year that is current (2026-27 -> 2027)
FIRST = 1990
years = list(range(FIRST, season_end + 1)) if FULL else [season_end - 1, season_end]

raw_path = os.path.join(DATA, "bbref_raw.json")
raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {"pl": [], "tm": [], "st": [], "po": []}
tx_path = os.path.join(DATA, "transactions_1991_2026.csv")
tx_old = pd.read_csv(tx_path) if os.path.exists(tx_path) and not FULL else pd.DataFrame(columns=["y", "txt", "links"])

new = {"pl": [], "tm": [], "st": [], "po": []}; new_tx = []
for y in years:
    print("Season", y)
    d = get(f"/leagues/NBA_{y}_advanced.html"); pl = []
    for tid, ph in [("advanced", "R"), ("advanced_post", "P")]:
        t = d.select_one(f"table#{tid}")
        if t:
            for tr in rows_of(t):
                o = cells(tr); o["year"] = y; o["phase"] = ph; pl.append(o)
    if not pl: print("  no player rows yet, skipping"); continue
    d = get(f"/leagues/NBA_{y}.html"); tm = []; st = []
    t = d.select_one("table#advanced-team")
    if t:
        for tr in rows_of(t): o = cells(tr); o["year"] = y; tm.append(o)
    for tid in ["divs_standings_E", "divs_standings_W"]:
        t = d.select_one(f"table#{tid}")
        if t:
            for tr in rows_of(t):
                a = tr.select_one("th a"); m = re.search(r"teams/([A-Z]{3})/", a["href"]) if a and a.get("href") else None
                o = cells(tr); o["tm"] = m.group(1) if m else ""; o["year"] = y; st.append(o)
    wins = sum(float(o.get("wins") or 0) for o in tm if o.get("team") != "League Average")
    if wins == 0: print("  season has no games yet, skipping"); continue
    d = get(f"/playoffs/NBA_{y}.html"); po = []
    t = d.select_one("table#all_playoffs")
    if t:
        for tr in t.select("tbody tr"):
            po.append({"year": y, "txt": re.sub(r"\s+", " ", tr.get_text(" ", strip=True)), "links": " ".join(a.get("href", "") for a in tr.select("a"))})
    d = get(f"/leagues/NBA_{y}_transactions.html")
    for p in d.select("#content p"):
        txt = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if len(txt) >= 12: new_tx.append({"y": y, "txt": txt, "links": " ".join(a.get("href", "") for a in p.select("a"))})
    for k, v in [("pl", pl), ("tm", tm), ("st", st), ("po", po)]: new[k] += v
    print(f"  players {len(pl)}, teams {len(tm)}, playoff rows {len(po)}")

done_years = {r["year"] for r in new["pl"]}
for k in raw: raw[k] = [r for r in raw[k] if r["year"] not in done_years] + new[k]
json.dump(raw, open(raw_path, "w"))
if new_tx:
    tx = pd.concat([tx_old[~tx_old.y.isin(done_years)], pd.DataFrame(new_tx)]); tx.to_csv(tx_path, index=False)

# drafts: this calendar year and last (or everything with --full)
dr_path = os.path.join(DATA, "drafts_1990_2025.csv"); dr = pd.read_csv(dr_path)
for y in (range(1990, now.year + 1) if FULL else [now.year - 1, now.year]):
    try: d = get(f"/draft/NBA_{y}.html")
    except Exception as e: print("draft", y, "not available:", e); continue
    t = d.select_one("table#stats"); out = []
    if t:
        for tr in rows_of(t):
            g = lambda s: (tr.select_one(f'[data-stat="{s}"]').get_text(strip=True).replace(",", "") if tr.select_one(f'[data-stat="{s}"]') else "")
            if g("pick_overall"): out.append((y, g("pick_overall"), g("team_id"), g("player")))
    if out: dr = pd.concat([dr[dr.year != y], pd.DataFrame(out, columns=["year", "pick", "tm", "player"])])
    print("Draft", y, len(out), "picks")
dr.sort_values(["year", "pick"], key=lambda s: pd.to_numeric(s, errors="coerce")).to_csv(dr_path, index=False)

# executives: current-franchise pages (Nets = NJN, Hornets = CHA, Pelicans = NOH on Basketball Reference)
rows = []
for t in "ATL BOS NJN CHI CHA CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOH NYK OKC ORL PHI PHO POR SAC SAS TOR UTA WAS".split():
    d = get(f"/teams/{t}/executives.html")
    for tr in [x for x in d.select("table tbody tr") if "thead" not in (x.get("class") or [])]:
        o = {c["data-stat"]: c.get_text(strip=True) for c in tr.select("[data-stat]")}
        if o.get("exec"): rows.append({"team": t, "exec": o["exec"], "date_start": o.get("date_start", ""), "date_end": o.get("date_end", "")})
if rows: pd.DataFrame(rows).to_csv(os.path.join(DATA, "executives.csv"), index=False); print("Executives:", len(rows), "rows")

# latest offseason: dated transactions from the draft until opening night (feeds the Offseason tab)
off_end = now.year + 1 if now.month >= 6 else now.year
MON = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
try:
    d = get(f"/leagues/NBA_{off_end}_transactions.html"); off_rows = []
    for li in d.select("ul.page_index li"):
        sp = li.select_one("span"); m = re.search(r"([A-Z][a-z]+) (\d+), (\d{4})", sp.get_text() if sp else li.get_text()[:40])
        if not m or m.group(1) not in MON: continue
        dd = f"{m.group(3)}-{MON[m.group(1)]:02d}-{int(m.group(2)):02d}"
        if dd > f"{off_end - 1}-10-21": continue
        for p in li.select("p"):
            txt = re.sub(r"\s+", " ", p.get_text()).strip()
            if len(txt) >= 12: off_rows.append({"d": dd, "txt": txt, "links": " ".join(a.get("href", "") for a in p.select("a"))})
    if off_rows: pd.DataFrame(off_rows).to_csv(os.path.join(DATA, "offseason.csv"), index=False); print("Offseason:", len(off_rows), "transactions")
except Exception as e: print("offseason page not available:", e)

print("Running the model...")
subprocess.run([sys.executable, os.path.join(BASE, "franchise_model.py")], check=True, cwd=BASE)
subprocess.run([sys.executable, os.path.join(BASE, "extras.py")], check=True, cwd=BASE)      # player stats, needs profiles, offseason
import build_dashboard; build_dashboard.build()
if "--push" in sys.argv:
    subprocess.run(["git", "add", "franchise_dashboard.html"], cwd=BASE); subprocess.run(["git", "commit", "-m", "Update dashboard " + str(now)], cwd=BASE); subprocess.run(["git", "push"], cwd=BASE)
print("All done. Open franchise_dashboard.html")
