#!/usr/bin/env python3
"""Run: open in IDLE, press F5. Needs: pip3 install pandas numpy matplotlib
Franchise history model 1991-2026 v2: draft, trades, free agency, contracts, fringe pipeline, GMs, penalties, play-in, style. Leak-free backtest."""
import json, re, unicodedata, sys
import numpy as np, pandas as pd
import os
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data") + os.sep      # input files
OUT = os.path.join(BASE, "output") + os.sep     # results go here
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

FR = {"NJN": "BKN", "BRK": "BKN", "CHH": "CHA", "CHO": "CHA", "CHA": "CHA", "NOH": "NOP", "NOK": "NOP", "NOP": "NOP", "PHO": "PHX",
      "SEA": "OKC", "VAN": "MEM", "WSB": "WAS", "KCK": "SAC", "SDC": "LAC", "CHP": "WAS"}
def fr(t): return FR.get(t, t)
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()
def num(x): return pd.to_numeric(x, errors="coerce")

D = json.load(open(DATA + "bbref_raw.json"))
# ---------------- players ----------------
P = pd.DataFrame(D["pl"])
for c in ["mp", "ws", "bpm", "vorp", "games", "age"]: P[c] = num(P[c])
P["nm"] = P.name_display.map(norm)
P["tm"] = P.team_name_abbr
reg = P[P.phase == "R"].copy()
# per-player-season: use per-team rows (skip TOT/2TM rows) for team attribution; TOT row for totals
agg = reg[reg.tm.str.match(r"^\d+TM$|^TOT$") == False].copy()
agg["tm"] = agg.tm.map(fr)
agg["wins_added"] = 0.5 * 2.7 * agg.vorp + 0.5 * (agg.ws - 0.040 * agg.mp / 48)   # blend of VORP and win-shares-over-replacement, in wins
agg["wins_added"] = agg.wins_added.fillna(0)
pid_year = agg.groupby(["pid", "year"], as_index=False).agg(wa=("wins_added", "sum"), mp=("mp", "sum"))
pid_name = agg.drop_duplicates("pid").set_index("pid").nm

# ---------------- salaries / cap ----------------
S = pd.read_csv(DATA + "nba_salaries_master.csv"); S["nm"] = S.player.map(norm); S["team"] = S.team.map(fr)
cap = pd.read_csv(DATA + "cap.csv")[["year", "Salary Cap"]].rename(columns={"Salary Cap": "cap"}).drop_duplicates("year").set_index("year").cap
S = S[S.year.isin(cap.index)]; S["share"] = S.salary / S.year.map(cap)
S = S.groupby(["nm", "year", "team"], as_index=False).share.sum()

# join player-seasons (by name+year+team) to salary share
agg2 = agg.copy(); agg2["team"] = agg2.tm
agg2 = agg2.merge(S, on=["nm", "year", "team"], how="left")
agg2["has_sal"] = agg2.share.notna()
# replacement pay = 10th pct share among players with >=200 min that year
repl = agg2[(agg2.mp >= 200) & agg2.has_sal].groupby("year").share.quantile(0.10)
agg2["excess"] = (agg2.share - agg2.year.map(repl)).clip(lower=0)
# price of a win (share of cap) per year = total excess pay / total positive wins added among salaried
tot = agg2[agg2.has_sal].groupby("year").agg(ex=("excess", "sum"), w=("wins_added", lambda s: s.clip(lower=0).sum()))
price = (tot.ex / tot.w)
agg2["price"] = agg2.year.map(price)
agg2["surplus_w"] = np.where(agg2.has_sal, agg2.wins_added - agg2.excess / agg2.price, np.nan)   # in wins; salary known only <=2024

# ---------------- draft ----------------
Dr = pd.read_csv(DATA + "drafts_1990_2025.csv"); Dr["nm"] = Dr.player.map(norm); Dr["tm"] = Dr.tm.map(fr)
Dr["pick"] = num(Dr.pick)
LAST = int(P.year.max())
pn = agg.groupby(["nm", "year"], as_index=False).wins_added.sum()
def firstk(row):
    return None
byplayer = {n: g.set_index("year").wins_added for n, g in pn.groupby("nm")}
rows = []
for r in Dr.itertuples():
    k = min(5, LAST - r.year)
    s = byplayer.get(r.nm)
    vals = [float(s.get(r.year + j, 0.0)) if s is not None else 0.0 for j in range(1, k + 1)]
    cum = np.cumsum(vals) if k > 0 else np.array([])
    rows.append(dict(year=r.year, pick=r.pick, tm=r.tm, nm=r.nm, player=r.player, k=k, **{f"c{j+1}": (cum[j] if j < k else np.nan) for j in range(5)}))
DR = pd.DataFrame(rows)
# expected value by pick for each horizon, from mature classes (<=LAST-5)
mat = DR[DR.year <= LAST - 5]
def smooth_curve(x, y, bw=3):
    out = {}
    for p in range(1, 61):
        w = np.exp(-0.5 * ((x - p) / (bw * (1 + p / 20))) ** 2); out[p] = (w * y).sum() / w.sum()
    return out
EXP, SD = {}, {}
for j in range(1, 6):
    c = smooth_curve(mat.pick.values, mat[f"c{j}"].values)
    resid2 = (mat[f"c{j}"].values - mat.pick.map(c).values) ** 2
    v = smooth_curve(mat.pick.values, resid2, bw=4)
    EXP[j] = c; SD[j] = {p: max(np.sqrt(v[p]), 0.5) for p in v}
def pick_eval(r):
    if r.k <= 0: return pd.Series({"exp": np.nan, "act": np.nan, "z": np.nan, "over": np.nan})
    p = int(min(r.pick, 60)); e = EXP[r.k][p]; a = r[f"c{r.k}"]
    return pd.Series({"exp": e, "act": a, "z": float(np.clip((a - e) / SD[r.k][p], -3, 5)), "over": a - e})
DR = pd.concat([DR, DR.apply(pick_eval, axis=1)], axis=1)
DR.loc[DR.pick > 60, ["z", "over"]] = np.nan
# draft credit: z-score weighted so a 30th-pick hit beats a #1 hit; pro-rate immature classes by k/5 confidence
DR["draft_score"] = DR.z * np.minimum(DR.k, 5) / 5
draft_team = DR.dropna(subset=["z"]).groupby(["tm", "year"], as_index=False).agg(draft_z=("draft_score", "sum"), draft_over=("over", "sum"))

# ---------------- outside acquisitions (FA + trades, not separable without a transaction log) ----------------
prev = agg2[["pid", "year", "tm", "mp"]].assign(year=lambda x: x.year + 1).rename(columns={"tm": "tm_prev", "mp": "mp_prev"})
A = agg2.merge(prev.drop_duplicates(["pid", "year", "tm_prev"]), on=["pid", "year"], how="left")
# arrival = had a prior season in NBA with a different team; draft-year rookies (no prior) excluded
A["arrival"] = A.tm_prev.notna() & (A.tm != A.tm_prev)
A = A.drop_duplicates(["pid", "year", "tm"])
arr = A[A.arrival & (A.mp >= 300)]
acq_team = arr[arr.has_sal].groupby(["tm", "year"], as_index=False).agg(acq_surplus=("surplus_w", "sum"), acq_n=("pid", "count"))
# not-rookie-scale proxy not needed: arrivals exclude draftees
# departures: wins_added produced elsewhere by players who left (next season) with >=800 min -- opportunity lost
dep = A[A.arrival & (A.mp_prev >= 800)].groupby(["tm_prev", "year"], as_index=False).wins_added.sum().rename(columns={"tm_prev": "tm", "wins_added": "wa_lost"})
nxt = arr.groupby(["tm", "year"], as_index=False).wins_added.sum().rename(columns={"wins_added": "wa_gain"})
flow = nxt.merge(dep, on=["tm", "year"], how="outer").fillna(0); flow["net_flow"] = flow.wa_gain - flow.wa_lost

# diamonds in the rough
dia = arr[arr.has_sal & (arr.share <= 2.5 * arr.year.map(repl)) & (arr.mp_prev.fillna(0) < 1000) & (arr.mp >= 500) & (arr.surplus_w >= 3)]
dia = dia.sort_values("surplus_w", ascending=False)[["year", "tm", "name_display", "tm_prev", "mp_prev", "mp", "wins_added", "surplus_w"]]
dia.round(2).to_csv(OUT + "diamonds_1991_2024.csv", index=False)

# contract surplus per team-season (all rostered players, ex-ante not available -> realized)
con_team = agg2[agg2.has_sal].groupby(["tm", "year"], as_index=False).agg(contract_surplus=("surplus_w", "sum"), wins_added=("wins_added", "sum"))

# ---------------- results ----------------
ST = pd.DataFrame(D["st"]); ST["team_name"] = ST.team_name.str.replace("*", "", regex=False).str.strip()
ST = ST[ST.tm != ""]; ST["tm"] = ST.tm.map(fr)
name2tm = ST.drop_duplicates(["year", "team_name"]).set_index(["year", "team_name"]).tm
TM = pd.DataFrame(D["tm"]); TM["team"] = TM.team.str.replace("*", "", regex=False).str.strip()
TM = TM[TM.team != "League Average"]
TM["tm"] = [name2tm.get((y, t), None) for y, t in zip(TM.year, TM.team)]
TM = TM.dropna(subset=["tm"])
for c in ["wins", "losses", "srs", "mov", "pace", "off_rtg", "def_rtg", "fg3a_per_fga_pct", "age"]: TM[c] = num(TM[c])
TM["win_pct"] = TM.wins / (TM.wins + TM.losses)
TM = TM.drop_duplicates(["tm", "year"])
# playoffs: parse series results "<Round> <Winner> over <Loser> (a-b)"
rounds = {"Finals": 4, "Conference Finals": 3, "Eastern Conference Finals": 3, "Western Conference Finals": 3, "Conference Semifinals": 2, "Eastern Conference Semifinals": 2,
          "Western Conference Semifinals": 2, "First Round": 1, "Eastern Conference First Round": 1, "Western Conference First Round": 1}
PO = []
for r in D["po"]:
    t = r["txt"]
    m = re.match(r"^(.*?) ([A-Z][\w\.\' ]+?) over ([A-Z][\w\.\' ]+?) \((\d)-(\d)\)", t)
    if "over" in t and "Series Stats" in t and not t.startswith("Game"):
        PO.append(r)
RK = []
for r in PO:
    links = re.findall(r"/teams/([A-Z]{3})/", r["links"])
    txt = r["txt"]; rd = 4 if txt.startswith("Finals") else 3 if "Conference Finals" in txt[:40] else 2 if "Semifinals" in txt[:40] else 1
    if len(links) >= 2: RK.append(dict(year=r["year"], rd=rd, win=fr(links[0]), lose=fr(links[1])))
RK = pd.DataFrame(RK)
# playoff points: made playoffs 1, + 1 per series won, + champion bonus
pp = {}
for r in RK.itertuples():
    pp[(r.win, r.year)] = max(pp.get((r.win, r.year), 0), r.rd + 1 + (1 if r.rd == 4 else 0))
    pp.setdefault((r.lose, r.year), r.rd)
playoff = pd.Series(pp).rename("playoff_pts"); playoff.index.names = ["tm", "year"]
playoff = playoff.reset_index()
TM = TM.merge(playoff, on=["tm", "year"], how="left"); TM["playoff_pts"] = TM.playoff_pts.fillna(0)
TM["champ"] = (TM.playoff_pts == 6).astype(int)

# ================= NEW IN V2: transactions (trades, free agents, two-way/pipeline), executives, penalties, play-in, style =================
TX = pd.read_csv(DATA + "transactions_1991_2026.csv").fillna("")
TX["pids"] = TX.links.map(lambda s: re.findall(r"/players/\w/(\w+)\.html", s))
TX["teams"] = TX.links.map(lambda s: [fr(t) for t in re.findall(r"/teams/([A-Z]{3})/", s)])
wa_by = {}                                        # (pid, year) -> wins_added
for r in agg.groupby(["pid", "year"]).wins_added.sum().reset_index().itertuples(): wa_by[(r.pid, r.year)] = r.wins_added
first_year = agg.groupby("pid").year.apply(sorted).to_dict()
def horizon_value(pid, start, n=3):
    """wins added over the player's first n NBA seasons at or after `start` (production credited to whoever holds him)."""
    ys = [y for y in first_year.get(pid, []) if y >= start][:n]
    return sum(wa_by.get((pid, y), 0.0) for y in ys)

# ---- free-agent / fringe signings
sg = TX[TX.txt.str.contains(" signed ") & ~TX.txt.str.contains("traded")].copy()
def sign_type(t):
    if "two-way" in t: return "two_way"
    if "Exhibit 10" in t: return "exh10"
    if "10-day" in t: return "ten_day"
    if "extension" in t: return "extension"
    return "fa"
sg["stype"] = sg.txt.map(sign_type)
sg = sg[(sg.pids.map(len) >= 1) & (sg.teams.map(len) >= 1)]
sg["pid"] = sg.pids.map(lambda x: x[0]); sg["tm"] = sg.teams.map(lambda x: x[0]); sg["year"] = sg.y
fa_keys = set(zip(sg[sg.stype == "fa"].pid, sg[sg.stype == "fa"].year, sg[sg.stype == "fa"].tm))
# arrivals split: signed as free agent vs came by trade/other
A["is_fa"] = [(p, y, t) in fa_keys for p, y, t in zip(A.pid, A.year, A.tm)]
arr = A[A.arrival & (A.mp >= 300)]
fa_team = arr[arr.has_sal & arr.is_fa].groupby(["tm", "year"], as_index=False).agg(fa_surplus=("surplus_w", "sum"), fa_n=("pid", "count"))
# fringe pipeline: two-way / Exhibit 10 / 10-day signees who became rotation players; credit the team that first signed them
pipe = sg[sg.stype.isin(["two_way", "exh10", "ten_day"])].sort_values(["year"]).drop_duplicates("pid")
pipe["pipe_wa"] = [max(horizon_value(p, y), 0.0) for p, y in zip(pipe.pid, pipe.year)]
pipe_team = pipe.groupby(["tm", "year"], as_index=False).agg(pipeline_wa=("pipe_wa", "sum"), pipe_n=("pid", "count"))
pipe_find = pipe[pipe.pipe_wa >= 4].sort_values("pipe_wa", ascending=False)[["year", "tm", "pid", "stype", "pipe_wa"]]
pipe_find["player"] = pipe_find.pid.map(lambda p: pid_name.get(p, p))
pipe_find.round(2).to_csv(OUT + "fringe_finds_two_way_exhibit10.csv", index=False)
two_way_n = sg[sg.stype == "two_way"].groupby(["tm", "year"]).size().rename("two_way_signed").reset_index()

# ---- trades: who received / gave which players (draft picks resolve to the player later selected)
rows = []
for r in TX[TX.txt.str.contains(" traded ") & ~TX.txt.str.contains("sold")].itertuples():
    toks = re.findall(r"/(teams)/([A-Z]{3})/|/(players)/\w/(\w+)\.html", r.links)
    cur = None; clauses = []
    for t in toks:
        if t[0] == "teams":
            if cur is None or cur["recv"] is not None:
                cur = {"give": fr(t[1]), "recv": None, "p": [], "q": []}; clauses.append(cur)
            else: cur["recv"] = fr(t[1])
        elif cur is not None:
            (cur["p"] if cur["recv"] is None else cur["q"]).append(t[3])
    for c in clauses:
        if c["recv"] is None: continue
        for p in c["p"]: rows.append((r.y, p, c["give"], c["recv"]))      # player moves give -> recv
        for p in c["q"]: rows.append((r.y, p, c["recv"], c["give"]))      # returned player moves recv -> give
TR = pd.DataFrame(rows, columns=["year", "pid", "frm", "to"])
TR["val"] = [horizon_value(p, y) for p, y in zip(TR.pid, TR.year)]
gain = TR.groupby(["to", "year"]).val.sum(); lose = TR.groupby(["frm", "year"]).val.sum()
trade_team = pd.concat([gain.rename("tr_in"), lose.rename("tr_out")], axis=1).fillna(0).reset_index().rename(columns={"level_0": "tm"})
trade_team.columns = ["tm", "year", "tr_in", "tr_out"]; trade_team["trade_net"] = trade_team.tr_in - trade_team.tr_out
tr_big = TR.assign(sign=1)[TR.val.abs() >= 6].copy()
bigtrades = TR.groupby(["year", "pid"]).agg(frm=("frm", "first"), to=("to", "last"), val=("val", "first")).reset_index()
bigtrades["player"] = bigtrades.pid.map(lambda p: pid_name.get(p, p))
bigtrades.sort_values("val", ascending=False).head(60).round(2).to_csv(OUT + "biggest_trade_acquisitions.csv", index=False)

# ---- head-coach hires/fires per team-season
cf = TX[TX.txt.str.contains("fired") & TX.txt.str.contains("Head Coach") & (TX.teams.map(len) >= 1)].copy()
cf["tm"] = cf.teams.map(lambda x: x[0]); coach_fired = cf.groupby(["tm", "y"]).size().rename("coach_fired").reset_index().rename(columns={"y": "year"})

# ---- executives (general manager / president) tenures
EX = pd.read_csv(DATA + "executives.csv")
EX["start"] = pd.to_datetime(EX.date_start.astype(str).str.slice(0, 10).where(EX.date_start.astype(str).str.len() > 4, EX.date_start.astype(str) + "-07-01"), errors="coerce")
EX["end"] = pd.to_datetime(EX.date_end.replace("present", "2030-12-31").astype(str).str.slice(0, 10).where(EX.date_end.astype(str).str.len() > 4, EX.date_end.astype(str) + "-06-30"), errors="coerce")
EX["team"] = EX.team.map(fr)
def exec_on(team, date):
    s = EX[(EX.team == team) & (EX.start <= date) & (EX.end > date)]
    return s.sort_values("start").exec.iloc[-1] if len(s) else None
# ---- penalties (hand-compiled from reporting; see penalties.csv)
PEN = pd.read_csv(DATA + "penalties.csv")

# ---------------- play-in (2021+): teams finishing 7th-10th in their conference; losers get half a playoff point ----------------
ST2 = pd.DataFrame(D["st"]); ST2 = ST2[ST2.tm != ""]; ST2["tm"] = ST2.tm.map(fr)
ST2["conf"] = None
for y, g in ST2.groupby("year"):
    if y >= 2005 and len(g) == 30: ST2.loc[g.index[:15], "conf"] = "E"; ST2.loc[g.index[15:], "conf"] = "W"
ST2 = ST2.drop_duplicates(["tm", "year"])
TMc = TM.merge(ST2[["tm", "year", "conf"]], on=["tm", "year"], how="left")
playin = []
for (y, c), g in TMc[TMc.year >= 2021].groupby(["year", "conf"]):
    g = g.sort_values(["win_pct", "srs"], ascending=False).reset_index(drop=True)
    for i in range(6, min(10, len(g))): playin.append((g.tm[i], y))
playin = pd.DataFrame(playin, columns=["tm", "year"]).assign(playin=1)
TM = TM.merge(playin, on=["tm", "year"], how="left"); TM["playin"] = TM.playin.fillna(0)
TM["playoff_pts"] = np.where((TM.playin == 1) & (TM.playoff_pts == 0), 0.5, TM.playoff_pts)

# ---------------- assemble franchise-season panel ----------------
Panel = TM[["tm", "year", "wins", "losses", "win_pct", "srs", "pace", "off_rtg", "def_rtg", "fg3a_per_fga_pct", "playoff_pts", "champ", "playin"]].copy()
for part in [draft_team, con_team, fa_team, trade_team, pipe_team, two_way_n, coach_fired]:
    Panel = Panel.merge(part, on=["tm", "year"], how="left")
for c in ["draft_z", "draft_over", "fa_surplus", "fa_n", "trade_net", "pipeline_wa", "pipe_n", "two_way_signed", "coach_fired", "tr_in", "tr_out"]: Panel[c] = Panel[c].fillna(0)
Panel["contract_other"] = Panel.contract_surplus - Panel.fa_surplus.where(Panel.contract_surplus.notna())   # surplus from everyone who was not a fresh free-agent signing
# penalties -> expected-wins cost of forfeited picks (pick slot: firsts ~#20, seconds ~#45; value = 5-yr expected wins at that slot)
def slot_cost(rd): return EXP[5][20 if rd == "1st" else 45]
PEN["cost_wins"] = [r.picks_lost * slot_cost(r.round) for r in PEN.itertuples()]
pen_team = PEN.groupby(["team", "draft_year"]).cost_wins.sum().reset_index().rename(columns={"team": "tm", "draft_year": "year", "cost_wins": "penalty_wins"})
Panel = Panel.merge(pen_team, on=["tm", "year"], how="left"); Panel["penalty_wins"] = Panel.penalty_wins.fillna(0)
# executive on the job when the roster was built (Oct 1 before the season) and at the draft (June 25)
Panel["exec_build"] = [exec_on(t, pd.Timestamp(y - 1, 10, 1)) for t, y in zip(Panel.tm, Panel.year)]
Panel["exec_draft"] = [exec_on(t, pd.Timestamp(y, 6, 25)) for t, y in zip(Panel.tm, Panel.year)]
Panel = Panel.sort_values(["tm", "year"]).reset_index(drop=True)
Panel["new_exec"] = (Panel.exec_build != Panel.groupby("tm").exec_build.shift(1)).astype(int) * Panel.groupby("tm").exec_build.shift(1).notna()
# style / era: team 3-point rate and pace relative to the league that year, and 3-year change in 3-point rate
for c, n in [("fg3a_per_fga_pct", "three_z"), ("pace", "pace_z")]:
    g = Panel.groupby("year")[c]; Panel[n] = (Panel[c] - g.transform("mean")) / g.transform("std")
Panel["three_delta"] = Panel.three_z - Panel.groupby("tm").three_z.shift(3)
Panel.to_csv(OUT + "franchise_season_panel.csv", index=False)

# ---------------- backtest (leak-free): every feature for season t uses only information that was fully known by the end of season t-1 ----------------
G = Panel.groupby("tm", group_keys=False)
def trail(col, n=3, lag=0): return G[col].transform(lambda s: s.shift(1 + lag).rolling(n, min_periods=1).mean())
Panel["t_srs"] = trail("srs"); Panel["t_playoff_pts"] = trail("playoff_pts")
Panel["t_contract_other"] = trail("contract_other"); Panel["t_fa"] = trail("fa_surplus")
Panel["t_draft"] = trail("draft_z", 5, lag=5)      # a draft class is judged on its first 5 seasons -> only classes >= 6 years old are fully known
Panel["t_trade"] = trail("trade_net", 3, lag=2)     # trade value uses the next 3 seasons -> only trades >= 3 years old are known
Panel["t_pipe"] = trail("pipeline_wa", 3, lag=2)
Panel["t_three_z"] = trail("three_z", 1); Panel["t_three_delta"] = trail("three_delta", 1); Panel["t_pace_z"] = trail("pace_z", 1)
Panel["t_coach_fired"] = trail("coach_fired", 1); Panel["t_new_exec"] = trail("new_exec", 1)
def ols(X, y, lam=0.0):
    X1 = np.column_stack([np.ones(len(X)), X]); A_ = X1.T @ X1 + lam * np.eye(X1.shape[1]); A_[0, 0] -= lam
    return np.linalg.solve(A_, X1.T @ y)
def run_bt(feats, first=2004, lam=5.0):
    errs = []
    for t in range(first, LAST + 1):
        tr = Panel[(Panel.year < t)].dropna(subset=feats + ["srs"]); te = Panel[Panel.year == t].dropna(subset=feats + ["srs"])
        if len(te) == 0 or len(tr) < 100: continue
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        b = ols(((tr[feats] - mu) / sd).values, tr.srs.values, lam)
        errs += list(te.srs.values - np.column_stack([np.ones(len(te)), ((te[feats] - mu) / sd).values]) @ b)
    return np.sqrt(np.mean(np.square(errs))), len(errs)
base_feats = ["t_srs"]
models = {"League average (everyone 0 SRS)": None, "Last 3 yrs SRS only": base_feats,
          "+ playoff history": base_feats + ["t_playoff_pts"], "+ draft": base_feats + ["t_draft"], "+ contract efficiency (non-FA)": base_feats + ["t_contract_other"],
          "+ free-agent signings": base_feats + ["t_fa"], "+ trades": base_feats + ["t_trade"], "+ two-way/Exhibit 10 pipeline": base_feats + ["t_pipe"],
          "+ playing style (3pt rate, pace)": base_feats + ["t_three_z", "t_three_delta", "t_pace_z"], "+ coach/GM changes": base_feats + ["t_coach_fired", "t_new_exec"],
          "All components": base_feats + ["t_playoff_pts", "t_draft", "t_contract_other", "t_fa", "t_trade", "t_pipe", "t_three_z", "t_three_delta", "t_pace_z", "t_coach_fired", "t_new_exec"]}
BT = []; base = np.sqrt(np.mean(Panel[Panel.year >= 2004].srs.dropna() ** 2))
for k, f in models.items():
    if f is None: BT.append((k, base, 0.0)); continue
    r, n = run_bt(f); BT.append((k, r, 1 - (r / base) ** 2))
BT = pd.DataFrame(BT, columns=["model", "rmse_srs", "r2_vs_avg"]).round(3); print("BACKTEST (predict each season's point differential, 2004-2026, leak-free)\n", BT.to_string(index=False)); BT.to_csv(OUT + "backtest_components.csv", index=False)
# learned weights = pooled ridge on standardized leak-free features
allf = models["All components"]; tr = Panel.dropna(subset=allf + ["srs"]); mu, sd = tr[allf].mean(), tr[allf].std().replace(0, 1)
W = pd.Series(ols(((tr[allf] - mu) / sd).values, tr.srs.values, 5.0)[1:], index=allf); W.round(3).to_csv(OUT + "learned_weights.csv"); print("\nlearned weights (standardized, per 1 SD):\n", W.round(3).to_string())

# ---------------- retrospective franchise scores (what each franchise built and won, season by season, z-scored within year) ----------------
Z = Panel.copy()
comps = ["srs", "playoff_pts", "draft_z", "trade_net", "fa_surplus", "contract_other", "pipeline_wa"]
for c in comps:
    g = Z.groupby("year")[c]; Z["z_" + c] = ((Z[c] - g.transform("mean")) / g.transform("std")).fillna(0)
sd_pick = Panel.draft_over.std()
Z["z_penalty"] = -Z.penalty_wins / sd_pick
Z["results_z"] = 0.5 * Z.z_srs + 0.5 * Z.z_playoff_pts
Z["built_z"] = (Z.z_draft_z + Z.z_trade_net + Z.z_fa_surplus + Z.z_contract_other + Z.z_pipeline_wa) / 5 + Z.z_penalty
Z.to_csv(OUT + "franchise_season_z.csv", index=False)
def window(lo, hi): return Z[(Z.year >= lo) & (Z.year <= hi)]
def table(lo, hi):
    w = window(lo, hi)
    t = w.groupby("tm").agg(seasons=("year", "count"), wins=("wins", "sum"), losses=("losses", "sum"), avg_srs=("srs", "mean"), titles=("champ", "sum"),
                           playoff_pts=("playoff_pts", "sum"), results_score=("results_z", "mean"), built_score=("built_z", "mean"),
                           draft=("z_draft_z", "mean"), trades=("z_trade_net", "mean"), free_agency=("z_fa_surplus", "mean"), contracts=("z_contract_other", "mean"),
                           pipeline=("z_pipeline_wa", "mean"), penalty_wins=("penalty_wins", "sum"))
    t["win_pct"] = t.wins / (t.wins + t.losses); t["blend"] = (t.results_score + t.built_score) / 2
    return t
FS = table(1991, 2026).sort_values("blend", ascending=False).round(2); FS["rank"] = range(1, len(FS) + 1)
for nm, lo, hi in [("blend_1991_99", 1991, 1999), ("blend_2000_09", 2000, 2009), ("blend_2010_19", 2010, 2019), ("blend_2020_26", 2020, 2026)]:
    FS[nm] = table(lo, hi).blend.round(2)
FS.to_csv(OUT + "franchise_scores.csv"); print("\nFRANCHISE SCORES 1991-2026 (results = point differential + playoffs; built = draft, trades, free agency, contracts, fringe pipeline, minus penalties)\n", FS.to_string())

# ---------------- GM / executive scorecard ----------------
Zb = Z.dropna(subset=["exec_build"]).groupby(["tm", "exec_build"]).agg(seasons=("year", "count"), first=("year", "min"), last=("year", "max"), results=("results_z", "mean"),
        trades=("z_trade_net", "mean"), fa=("z_fa_surplus", "mean"), contracts=("z_contract_other", "mean"), pipeline=("z_pipeline_wa", "mean")).reset_index().rename(columns={"exec_build": "exec"})
Zd = Z.dropna(subset=["exec_draft"]).groupby(["tm", "exec_draft"]).agg(draft=("z_draft_z", "mean")).reset_index().rename(columns={"exec_draft": "exec"})
GM = Zb.merge(Zd, on=["tm", "exec"], how="left").fillna({"draft": 0})
GM["built"] = (GM.draft + GM.trades + GM.fa + GM.contracts + GM.pipeline) / 5; GM["blend"] = (GM.built + GM.results) / 2
GMt = GM[GM.seasons >= 3].sort_values("blend", ascending=False).round(2); GMt.to_csv(OUT + "gm_scorecard.csv", index=False)
print("\nTOP / BOTTOM EXECUTIVES (>=3 seasons in charge)\n", pd.concat([GMt.head(12), GMt.tail(12)]).to_string(index=False))
# does an executive's built-value travel? compare each exec's first stint vs second stint with a different team
m = GM[GM.seasons >= 3]; pers = []
for e, g in m.groupby("exec"):
    g = g.sort_values("first")
    if g.tm.nunique() >= 2: pers.append((e, g.built.iloc[0], g.built.iloc[-1], g.results.iloc[0], g.results.iloc[-1]))
pers = pd.DataFrame(pers, columns=["exec", "built_1", "built_2", "res_1", "res_2"])
if len(pers) >= 5:
    from math import isnan
    cb, cr = pers[["built_1", "built_2"]].corr(method="spearman").iloc[0, 1], pers[["res_1", "res_2"]].corr(method="spearman").iloc[0, 1]
    print(f"\nExecutive persistence across teams (n={len(pers)} executives with 3+ seasons at two teams): built-value rank corr {cb:.2f}, results rank corr {cr:.2f}")
# do new GMs / coach firings help? teams with bad recent history, change vs no change
P3 = Panel.copy(); g = P3.groupby("tm")
P3["prev3"] = g.srs.transform(lambda s: s.shift(1).rolling(3).mean()); P3["next3"] = g.srs.transform(lambda s: s[::-1].shift(0).rolling(3).mean()[::-1])
P3["chg"] = P3.next3 - P3.prev3; bad = P3[(P3.prev3 < -2) & P3.next3.notna()]
for lab, col in [("new GM/president this season", "new_exec"), ("head coach fired this season", "coach_fired")]:
    a, b2 = bad[bad[col] > 0].chg, bad[bad[col] == 0].chg
    print(f"Bad teams (prev-3yr SRS < -2): change in SRS over next 3 seasons after {lab}: {a.mean():+.2f} (n={len(a)}) vs no change {b2.mean():+.2f} (n={len(b2)})")

# ---------------- other tables ----------------
DR[DR.k >= 3].sort_values("z", ascending=False).head(60)[["year", "pick", "tm", "player", "k", "act", "exp", "z"]].round(2).to_csv(OUT + "best_draft_picks.csv", index=False)
DR[DR.k >= 3].sort_values("z").head(40)[["year", "pick", "tm", "player", "k", "act", "exp", "z"]].round(2).to_csv(OUT + "worst_draft_picks.csv", index=False)
era_style = TM.groupby("year").agg(pace=("pace", "mean"), three_rate=("fg3a_per_fga_pct", "mean"), ortg=("off_rtg", "mean")).round(3); era_style["price_of_win_pct_cap"] = (price * 100).round(2)
era_style.to_csv(OUT + "era_style.csv")
PEN.to_csv(OUT + "penalties_used.csv", index=False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(10, 10)); ax.scatter(FS.built_score, FS.results_score, s=40 + 60 * FS.titles, color="#1f6f4a", alpha=.7)
for t, r in FS.iterrows(): ax.annotate(t, (r.built_score, r.results_score), fontsize=9, xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
ax.set_xlabel("Built value (draft, trades, free agency, contracts, fringe pipeline, penalties), z-score"); ax.set_ylabel("Results (point differential + playoff success), z-score")
ax.set_title("NBA franchises 1991-2026: what they built vs what they won\n(dot size = championships)"); plt.tight_layout(); plt.savefig(OUT + "franchise_map.png", dpi=130)
print("\nDone. Files are in:", OUT)
