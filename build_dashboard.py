"""Turns the model's output CSVs into index.html (a single self-contained page)."""
import os, json
import pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(BASE, "output")

def _nn(v, nd=None):
    return None if pd.isna(v) else (round(float(v), nd) if nd is not None else v)

def season_detail():
    """Per franchise-season roster and moves, keyed 'TEAM|year' (empty if the model has not produced them)."""
    roster, moves = {}, {}
    rp, mp = os.path.join(OUT, "roster_by_season.csv"), os.path.join(OUT, "team_moves.csv")
    if os.path.exists(rp):
        for r in pd.read_csv(rp).itertuples():
            roster.setdefault(f"{r.tm}|{int(r.year)}", []).append([r.player, _nn(r.age, 0), _nn(r.games, 0), _nn(r.mp, 0), round(float(r.wins_added), 1), _nn(r.salary, 1), None if pd.isna(r.how) else r.how])
    if os.path.exists(mp):
        for r in pd.read_csv(mp).itertuples():
            moves.setdefault(f"{r.tm}|{int(r.year)}", []).append([r.kind, r.text])
    return roster, moves

def build():
    fs = pd.read_csv(os.path.join(OUT, "franchise_scores.csv")).round(2)
    z = pd.read_csv(os.path.join(OUT, "franchise_season_z.csv"))
    keep = ["tm", "year", "wins", "losses", "srs", "playoff_pts", "champ", "results_z", "built_z", "z_draft_z", "z_trade_net", "z_fa_surplus", "z_contract_other", "z_pipeline_wa", "exec_build"]
    z = z[keep].round(2); z["exec_build"] = z.exec_build.fillna("")
    gm = pd.read_csv(os.path.join(OUT, "gm_scorecard.csv")).round(2)
    bt = pd.read_csv(os.path.join(OUT, "backtest_components.csv"))
    pen = pd.read_csv(os.path.join(OUT, "penalties_used.csv"))[["team", "draft_year", "round", "picks_lost", "note"]]
    pen = pen[pen.note.fillna("").str.len() > 0]
    roster, moves = season_detail()
    data = dict(fs=fs.to_dict("records"), z=z.to_dict("records"), gm=gm.to_dict("records"), bt=bt.to_dict("records"), pen=pen.to_dict("records"), roster=roster, moves=moves)
    tpl = open(os.path.join(BASE, "dashboard_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    for name in ("franchise_dashboard.html", "index.html"):      # index.html is the file GitHub Pages serves
        path = os.path.join(BASE, name); open(path, "w", encoding="utf-8").write(html)
    print("Wrote franchise_dashboard.html and index.html")

if __name__ == "__main__":
    build()
