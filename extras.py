"""extras.py - season detail for the dashboard: player stats, "what the team lacked" profile, and the latest offseason.
Run after franchise_model.py (update.py does this). Reads data/ and output/, writes output/extras.json."""
import os, re, json, datetime
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(BASE, "data"); OUT = os.path.join(BASE, "output")
FR = {"NJN": "BKN", "BRK": "BKN", "CHH": "CHA", "CHO": "CHA", "CHA": "CHA", "NOH": "NOP", "NOK": "NOP", "NOP": "NOP", "PHO": "PHX", "SEA": "OKC", "VAN": "MEM", "WSB": "WAS", "KCK": "SAC", "SDC": "LAC", "CHP": "WAS"}
fr = lambda t: FR.get(t, t)
num = lambda x: pd.to_numeric(x, errors="coerce")
def nn(v, nd=1):
    return None if v is None or pd.isna(v) else round(float(v), nd)

D = json.load(open(os.path.join(DATA, "bbref_raw.json")))

# ---------------- player stats per franchise-season ----------------
P = pd.DataFrame(D["pl"]); P = P[P.phase == "R"].copy()
P = P[~P.team_name_abbr.str.match(r"^\d+TM$|^TOT$")]; P["tm"] = P.team_name_abbr.map(fr)
for c in ["age", "games", "mp", "per", "ts_pct", "usg_pct", "ast_pct", "trb_pct", "stl_pct", "blk_pct", "tov_pct", "ws", "bpm", "vorp"]: P[c] = num(P[c])
P["wa"] = (0.5 * 2.7 * P.vorp + 0.5 * (P.ws - 0.040 * P.mp / 48)).fillna(0)
P = P.sort_values(["tm", "year", "mp"], ascending=[True, True, False])
extra = {}
rp = os.path.join(OUT, "roster_by_season.csv")
if os.path.exists(rp):
    R = pd.read_csv(rp)
    for r in R.itertuples(): extra[(r.tm, int(r.year), r.player)] = (None if pd.isna(r.salary) else float(r.salary), None if pd.isna(r.how) else r.how)
players = {}
for (tm, y), g in P.groupby(["tm", "year"]):
    rows = []
    for r in g.head(15).itertuples():
        sal, how = extra.get((tm, int(y), r.name_display), (None, None))
        rows.append([r.name_display, nn(r.age, 0), r.pos, nn(r.games, 0), nn(r.mp, 0), nn(r.wa), nn(sal), how, nn(r.per), nn(r.ts_pct * 100), nn(r.usg_pct), nn(r.ast_pct), nn(r.trb_pct), nn(r.stl_pct), nn(r.blk_pct), nn(r.tov_pct), nn(r.bpm), nn(r.ws), nn(r.vorp)])
    players[f"{tm}|{int(y)}"] = rows

# ---------------- "what they lacked": measurable profile vs the league, season by season ----------------
ST = pd.DataFrame(D["st"]); ST["team_name"] = ST.team_name.str.replace("*", "", regex=False).str.strip(); ST = ST[ST.tm != ""]; ST["tm"] = ST.tm.map(fr)
name2tm = ST.drop_duplicates(["year", "team_name"]).set_index(["year", "team_name"]).tm
T = pd.DataFrame(D["tm"]); T["team"] = T.team.str.replace("*", "", regex=False).str.strip(); T = T[T.team != "League Average"]
T["tm"] = [name2tm.get((y, t)) for y, t in zip(T.year, T.team)]; T = T.dropna(subset=["tm"]).drop_duplicates(["tm", "year"])
for c in ["wins", "losses", "off_rtg", "def_rtg", "efg_pct", "tov_pct", "orb_pct", "ft_rate", "opp_efg_pct", "opp_tov_pct", "drb_pct", "opp_ft_rate", "age"]: T[c] = num(T[c])
def roster_feats(g):
    g = g.sort_values("mp", ascending=False); w = g.wa.sort_values(ascending=False)
    top3 = g.loc[w.index[:3]]
    return pd.Series({"star": w.iloc[0] if len(w) else np.nan, "top3": w.iloc[:3].sum(), "depth": g.wa.iloc[5:10].sum() if len(g) >= 6 else np.nan, "avail_games": top3.games.sum()})
RF = P.groupby(["tm", "year"]).apply(roster_feats, include_groups=False).reset_index()
T = T.merge(RF, on=["tm", "year"], how="left"); T["avail"] = T.avail_games / (3 * (T.wins + T.losses))
# key, label, group, higher_is_better, decimals, unit
METRICS = [("off_rtg", "Offensive rating", "Offense", 1, 1, ""), ("efg_pct", "Shot making (eFG%)", "Offense", 1, 3, ""), ("tov_pct", "Turnover rate", "Offense", -1, 1, "%"),
           ("orb_pct", "Offensive rebounding", "Offense", 1, 1, "%"), ("ft_rate", "Free-throw rate", "Offense", 1, 3, ""),
           ("def_rtg", "Defensive rating", "Defense", -1, 1, ""), ("opp_efg_pct", "Opponent eFG%", "Defense", -1, 3, ""), ("opp_tov_pct", "Forced turnovers", "Defense", 1, 1, "%"),
           ("drb_pct", "Defensive rebounding", "Defense", 1, 1, "%"), ("opp_ft_rate", "Opponent free-throw rate", "Defense", -1, 3, ""),
           ("star", "Best player (wins added)", "Roster", 1, 1, ""), ("top3", "Top three (wins added)", "Roster", 1, 1, ""), ("depth", "Depth, 6th-10th men (wins added)", "Roster", 1, 1, ""),
           ("avail", "Top-three availability", "Roster", 1, 2, "")]
needs = {}
for y, g in T.groupby("year"):
    cols = {}
    for k, lab, grp, d, dec, u in METRICS:
        s = g[k]; z = (s - s.mean()) / s.std(ddof=0) * d if s.notna().sum() >= 20 and s.std(ddof=0) > 0 else pd.Series(np.nan, index=g.index)
        rank = (s * d).rank(ascending=False, method="min")
        cols[k] = (z, rank, s)
    for i, r in zip(g.index, g.itertuples()):
        needs[f"{r.tm}|{int(y)}"] = {"z": [nn(cols[k][0][i], 2) for k, *_ in METRICS], "r": [None if pd.isna(cols[k][1][i]) else int(cols[k][1][i]) for k, *_ in METRICS],
                                     "v": [nn(cols[k][2][i], m[4]) for k, m in zip([m[0] for m in METRICS], METRICS)], "age": nn(r.age, 1)}
# who eliminated each team (the series winner in the round they lost); champions get none
rounds = []
for r in D["po"]:
    t = r["txt"]
    if "over" in t and "Series Stats" in t and not t.startswith("Game"):
        links = re.findall(r"/teams/([A-Z]{3})/", r["links"]); rd = 4 if t.startswith("Finals") else 3 if "Conference Finals" in t[:40] else 2 if "Semifinals" in t[:40] else 1
        if len(links) >= 2: rounds.append((int(r["year"]), rd, fr(links[0]), fr(links[1])))
for y, rd, w, l in sorted(rounds):
    k = f"{l}|{y}"
    if k in needs: needs[k]["by"] = w          # later rounds overwrite earlier, so the last series lost wins

# ---------------- latest offseason ----------------
def classify(txt):
    if re.search(r"Head Coach|General Manager|President|Assistant Coach|Executive|hired|fired|promoted|resigns|named", txt) and not re.search(r"signed|waived|traded", txt): return "Staff"
    if " traded " in txt or re.search(r"-team trade", txt): return "Trade"
    if " signed " in txt:
        if re.search(r"two-way|Exhibit 10|10-day", txt): return "Camp deal"
        return "Extension" if "extension" in txt else "Signing"
    if re.search(r"waived|released", txt): return "Waived"
    return "Other"
today = datetime.date.today(); off_end = today.year + 1 if today.month >= 6 else today.year
off = {"year": off_end - 1, "label": f"{off_end - 1} offseason", "moves": [], "picks": []}
op = os.path.join(DATA, "offseason.csv")
if os.path.exists(op):
    O = pd.read_csv(op).fillna("")
    for r in O.itertuples():
        teams = list(dict.fromkeys(fr(t) for t in re.findall(r"/teams/([A-Z]{3})/", r.links)))
        if not teams: continue
        k = classify(r.txt); off["moves"].append({"d": r.d, "k": k, "t": teams, "x": r.txt[:900 if k == "Trade" else 320]})
    off["moves"].sort(key=lambda m: m["d"], reverse=True)
fp = os.path.join(DATA, "frontoffice.json")      # hand-compiled coaching and front-office changes (like penalties.csv)
if os.path.exists(fp):
    for m in json.load(open(fp)):
        if m["d"] >= f"{off_end - 1}-04-01": off["moves"].append({"d": m["d"], "dl": m.get("dl"), "k": "Staff", "t": m["t"], "x": m["x"], "s": m.get("src"), "u": m.get("url")})
    off["moves"].sort(key=lambda m: m["d"], reverse=True)
dp = os.path.join(DATA, "drafts_1990_2025.csv")
if os.path.exists(dp):
    dr = pd.read_csv(dp); dr = dr[dr.year == off_end - 1]
    off["picks"] = [{"n": int(r.pick), "t": fr(r.tm), "p": str(r.player)} for r in dr.itertuples() if str(r.player) != "nan"]
sp = os.path.join(DATA, "storylines.json")
stories = [s for s in json.load(open(sp)) if s.get("year", off_end - 1) == off_end - 1] if os.path.exists(sp) else []
# ---------------- aging curve: how much a player's wins added changes from one season to the next, by age ----------------
A = P.groupby(["pid", "year"]).agg(wa=("wa", "sum"), mp=("mp", "sum"), age=("age", "min")).reset_index()
last_y = int(A.year.max()); A = A[A.mp >= 1000]
nxt = P.groupby(["pid", "year"]).wa.sum().rename("wa_next").reset_index(); nxt["year"] -= 1
A = A[A.year < last_y].merge(nxt, on=["pid", "year"], how="left"); A["wa_next"] = A.wa_next.fillna(0)     # left the league = 0
A["d"] = A.wa_next - A.wa; A["age"] = A.age.clip(19, 38).round()
cv = A.groupby("age").d.agg(["mean", "count"]); cv = cv[cv["count"] >= 15]
curve = {int(a): round(float(v), 2) for a, v in cv["mean"].rolling(3, center=True, min_periods=1).mean().items()}

json.dump({"players": players, "needs": needs, "metrics": [[m[0], m[1], m[2], m[4], m[5]] for m in METRICS], "off": off, "stories": stories, "curve": curve}, open(os.path.join(OUT, "extras.json"), "w"), separators=(",", ":"), ensure_ascii=False)
print(f"extras.json written: {len(players)} rosters, {len(needs)} profiles, {len(off['moves'])} offseason moves, {len(off['picks'])} picks")
