"""Turns the model's output CSVs into franchise_dashboard.html (a single self-contained page)."""
import os, json
import pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(BASE, "output")

def build():
    fs = pd.read_csv(os.path.join(OUT, "franchise_scores.csv")).round(2)
    z = pd.read_csv(os.path.join(OUT, "franchise_season_z.csv"))
    keep = ["tm", "year", "wins", "losses", "srs", "playoff_pts", "champ", "results_z", "built_z", "z_draft_z", "z_trade_net", "z_fa_surplus", "z_contract_other", "z_pipeline_wa", "exec_build"]
    z = z[keep].round(2); z["exec_build"] = z.exec_build.fillna("")
    gm = pd.read_csv(os.path.join(OUT, "gm_scorecard.csv")).round(2)
    bt = pd.read_csv(os.path.join(OUT, "backtest_components.csv"))
    pen = pd.read_csv(os.path.join(OUT, "penalties_used.csv"))[["team", "draft_year", "round", "picks_lost", "note"]]
    pen = pen[pen.note.fillna("").str.len() > 0]
    data = dict(fs=fs.to_dict("records"), z=z.to_dict("records"), gm=gm.to_dict("records"), bt=bt.to_dict("records"), pen=pen.to_dict("records"))
    tpl = open(os.path.join(BASE, "dashboard_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    for name in ("franchise_dashboard.html", "index.html"):      # index.html is the file GitHub Pages serves
        path = os.path.join(BASE, name); open(path, "w", encoding="utf-8").write(html)
    print("Wrote franchise_dashboard.html and index.html")

if __name__ == "__main__":
    build()
