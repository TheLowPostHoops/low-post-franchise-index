"""windowfo.py - current rosters (last season plus the offseason's trades, signings and draft), window clock projections,
and front-office report cards. Called by extras.py. Returns a dict that goes into output/extras.json."""
import os, re, json
import numpy as np, pandas as pd

def compute(D, DATA, OUT, fr, curve, nn):
    P = pd.DataFrame(D["pl"]); P = P[P.phase == "R"].copy()
    for c in ["age", "mp", "ws", "vorp"]: P[c] = pd.to_numeric(P[c], errors="coerce")
    P["wa"] = (0.5 * 2.7 * P.vorp + 0.5 * (P.ws - 0.040 * P.mp / 48)).fillna(0)
    LASTY = int(P.year.max())
    name_of = P.drop_duplicates("pid", keep="last").set_index("pid").name_display.to_dict()
    pid_of = {v: k for k, v in name_of.items()}
    # ---- talent: last three seasons, weights .5/.3/.2, minutes-weighted with a prior of 800 replacement minutes ----
    tot = P[~P.team_name_abbr.str.match(r"^\d+TM$|^TOT$")]
    A = tot.groupby(["pid", "year"]).agg(wa=("wa", "sum"), mp=("mp", "sum")).reset_index()
    num, den, wsum = {}, {}, {}
    for r in A.itertuples():
        k = LASTY - r.year
        if 0 <= k <= 2:
            w = (.5, .3, .2)[k]; num[r.pid] = num.get(r.pid, 0) + w * r.wa; den[r.pid] = den.get(r.pid, 0) + w * r.mp; wsum[r.pid] = wsum.get(r.pid, 0) + w
    T = {p: num[p] * 2900 / (den[p] + 500 * wsum[p]) / 1.0 for p in num}      # a full 2,400-minute season keeps its value; thin samples shrink toward zero
    cur = lambda a: curve.get(int(max(19, min(38, round(a)))), -0.8 if a >= 30 else 0.3)
    lastrow = tot[tot.year >= LASTY - 1].drop_duplicates("pid", keep="last")
    age26 = {p: a + (LASTY - y) for p, a, y in zip(lastrow.pid, lastrow.age, lastrow.year)}
    # ---- rookies and pick values from history ----
    dr = pd.read_csv(os.path.join(DATA, "drafts_1990_2025.csv")); dr["pick"] = pd.to_numeric(dr.pick, errors="coerce")
    ys = A.assign(name=A.pid.map(name_of))
    wa_by = {(n, y): w for n, y, w in zip(ys.name, ys.year, ys.wa)}
    def hist(pk_lo, pk_hi):
        rows = dr[(dr.year.between(2008, LASTY - 5)) & dr.pick.between(pk_lo, pk_hi)]
        y1 = [wa_by.get((n, y + 1), 0) for n, y in zip(rows.player, rows.year)]
        c4 = [sum(wa_by.get((n, y + k), 0) for k in (1, 2, 3, 4)) for n, y in zip(rows.player, rows.year)]
        return (np.mean(y1) if y1 else 0, np.mean(c4) if c4 else 0)
    B = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 45), (46, 60)]
    HB = {b: hist(*b) for b in B}
    def rookie_val(pk):
        for b in B:
            if b[0] <= pk <= b[1]: return max(0, HB[b][0])
        return 0
    v1 = np.mean([HB[b][1] for b in B[:7]]) / 4 * .6; v2 = np.mean([HB[b][1] for b in B[7:]]) / 4 * .6
    # ---- baseline rosters: where each player finished last season ----
    ST = pd.DataFrame(D["st"]); ST["team_name"] = ST.team_name.str.replace("*", "", regex=False).str.strip(); ST = ST[ST.year == LASTY]
    tmap = {n: fr(t) for n, t in zip(ST.team_name, ST.tm) if t}
    last = tot[tot.year >= LASTY - 1].drop_duplicates("pid", keep="last")        # includes players who missed all of last season
    last = last[(last.age + (LASTY - last.year) <= 36)]
    pm = A.set_index(["pid", "year"]).mp.to_dict()
    last = last[[(pm.get((p, LASTY), 0) > 0) or (pm.get((p, LASTY - 1), 0) >= 1000) for p in last.pid]]
    team = {p: fr(t) for p, t in zip(last.pid, last.team_name_abbr)}
    # rookies
    rookies = {}
    for r in dr[dr.year == LASTY].itertuples():
        pid = "D:" + str(r.player); rookies[pid] = (str(r.player), int(r.pick)); team[pid] = fr(r.tm); name_of[pid] = str(r.player); pid_of[str(r.player)] = pid
        T[pid] = rookie_val(int(r.pick)); age26[pid] = 20
    def traj(p, n=3):
        w = max(0.0, T[p]) if p in rookies else max(0.0, T.get(p, 0) + cur(age26.get(p, 27)))
        out = [w]
        for k in range(1, n):
            w = max(0.0, w + cur(age26.get(p, 27) + k - (1 if p in rookies else 0))); out.append(w)
        return out
    v = lambda p: float(np.mean(traj(p)))          # three-year value, in wins added per season
    FRS = set(fr(t) for t in ST.tm if t)
    net = {t: 0.0 for t in FRS}; adds = {t: [] for t in net}; losses = {t: [] for t in net}; pk = {t: [0, 0, 0, 0] for t in net}   # firsts in, firsts out, seconds in, seconds out
    for pid, tm_ in list(team.items()):
        if pid in rookies and tm_ in net: net[tm_] += v(pid); adds[tm_].append((v(pid), name_of[pid] + " (#" + str(rookies[pid][1]) + " pick)", "draft"))
    def move(p, new, why=""):
        old = team.get(p)
        if old == new: return
        if old is not None and old in net: net[old] -= v(p); losses[old].append((v(p), name_of.get(p, p), why))
        if new is not None and new in net: net[new] += v(p); adds[new].append((v(p), name_of.get(p, p), why))
        team[p] = new
    def assets(txt):
        pl, f1, f2 = [], 0, 0
        for tok in re.split(r",\s*(?:and\s+)?|\s+and\s+", txt):
            tok = tok.strip().rstrip(".")
            m = re.match(r"an? (\d{4}) (1st|2nd) round draft pick", tok)
            if m: f1 += m.group(2) == "1st"; f2 += m.group(2) == "2nd"; continue
            if tok in pid_of: pl.append(pid_of[tok])
        return pl, f1, f2
    O = pd.read_csv(os.path.join(DATA, "offseason.csv")).fillna(""); O = O.sort_values("d", kind="stable")
    tname = lambda s: tmap.get(re.sub(r"^[Tt]he ", "", s.strip()))
    for r in O.itertuples():
        txt = re.split(r"(?<!Jr)(?<!Sr)(?<!St)\.\s+(?=[A-Z])", r.txt)[0]
        if " traded " in txt:
            for cl in txt.split("; "):
                m = re.search(r"(?:the )?(?P<g>[A-Z][A-Za-z. ]+?) traded (?P<a>.+?) to the (?P<r>[A-Z][A-Za-z. ]+?)(?: for (?P<b>.+?))?\.?$", cl)
                if not m: continue
                g, rc = tname(m.group("g")), tname(m.group("r"))
                if not g or not rc: continue
                pl, f1, f2 = assets(m.group("a"))
                for p in pl: move(p, rc, "trade")
                if g in pk: pk[g][1] += f1; pk[g][3] += f2
                if rc in pk: pk[rc][0] += f1; pk[rc][2] += f2
                if m.group("b"):
                    pl, f1, f2 = assets(m.group("b"))
                    for p in pl: move(p, g, "trade")
                    if g in pk: pk[g][0] += f1; pk[g][2] += f2
                    if rc in pk: pk[rc][1] += f1; pk[rc][3] += f2
        elif " signed " in txt and not re.search(r"Exhibit 10|two-way|10-day|extension", txt):
            tm = re.search(r"/teams/([A-Z]{3})/", r.links); pm = re.search(r"/players/\w/(\w+)\.html", r.links)
            m = re.search(r" signed (.+?)(?: to .*)?\.?$", txt)
            if not tm: continue
            t = fr(tm.group(1)); nm = m.group(1).strip().rstrip(".") if m else None
            p = pid_of.get(nm) or (pm.group(1) if pm else None)
            if p: move(p, t, "signed")
        elif " waived " in txt or " released " in txt:
            m = re.search(r" (?:waived|released) (.+?)\.?$", txt)
            if m and m.group(1).strip() in pid_of: move(pid_of[m.group(1).strip()], None, "waived")
    # ---- current rosters, core, projections ----
    win = {}
    for t in net:
        ps = [p for p, tt in team.items() if tt == t and (p in T)]
        ps.sort(key=lambda p: -v(p)); core = ps[:8]
        w = {p: v(p) for p in core}; age = {p: age26.get(p, 27) + 1 for p in core}
        p_ = [sum(w.values())]
        w2 = dict(w)
        for k in range(1, 7):
            for p in core:
                w2[p] = max(0.0, w2[p] + cur(age[p] + k - 1))
            p_.append(sum(w2.values()))
        new = {re.sub(r" \(#\d+ pick\)", "", x[1]) for x in adds[t]}
        win[t] = {"p": [round(float(x), 1) for x in p_], "core": [[name_of.get(p, p), int(age[p]), round(w[p], 1), round(cur(age[p]), 1), name_of.get(p, p) in new] for p in core],
                  "adds": [[a[1], round(a[0], 1)] for a in sorted(adds[t], reverse=True)[:4] if a[0] >= .5], "lost": [[a[1], round(a[0], 1)] for a in sorted(losses[t], reverse=True)[:3] if a[0] >= .5],
                  "net": round(net[t], 1), "picks": pk[t], "pickval": round((pk[t][0] - pk[t][1]) * v1 + (pk[t][2] - pk[t][3]) * v2, 1)}
    # ---- front office report cards (current regimes) ----
    z = pd.read_csv(os.path.join(OUT, "franchise_season_z.csv")); fo = {}
    new_re = set()
    fp = os.path.join(DATA, "frontoffice.json")
    if os.path.exists(fp):
        for m in json.load(open(fp)):
            if re.search(r"president|general manager|front office|executive vice", m["x"], re.I): new_re.update(m["t"])
    rows = []
    for t in win:
        zz = z[z.tm == t].sort_values("year"); ex = zz.exec_build.iloc[-1] if len(zz) else ""
        mine = zz[zz.exec_build == ex]
        ww = 0.5 ** ((LASTY - mine.year) / 2.0)
        ten = float((ww * (0.5 * mine.results_z + 0.5 * mine.built_z)).sum() / ww.sum()) if len(mine) and t not in new_re else None
        assets_ = float(np.mean(win[t]["p"][:4])); off_ = win[t]["net"] + win[t]["pickval"]
        rows.append([t, ex, ten, len(mine), assets_, off_])
    df = pd.DataFrame(rows, columns=["tm", "exec", "ten", "yrs", "assets", "off"])
    zs = lambda s: (s - s.mean()) / s.std(ddof=0)
    df["za"], df["zo"] = zs(df.assets), zs(df.off)
    df["zt"] = zs(df.ten.astype(float)).fillna(0)
    has = df.ten.notna()
    df["score"] = np.where(has, .30 * df.zt + .30 * df.za + .40 * df.zo, .50 * df.za + .50 * df.zo)
    df["pct"] = df.score.rank(pct=True)
    for r in df.itertuples():
        g = "A" if r.pct >= .85 else "B" if r.pct >= .6 else "C" if r.pct >= .35 else "D" if r.pct >= .15 else "F"
        fo[r.tm] = {"exec": r.exec, "new": bool(r.tm in new_re), "yrs": int(r.yrs), "ten": nn(r.ten, 2), "assets": nn(r.za, 2), "off": nn(r.zo, 2), "net": win[r.tm]["net"], "pickval": win[r.tm]["pickval"], "score": nn(r.score, 2), "grade": g}
    return {"win": win, "fo": fo, "pickv": [round(v1, 2), round(v2, 2)]}
