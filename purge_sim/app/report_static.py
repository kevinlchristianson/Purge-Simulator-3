"""
report_static.py  —  the parts of the client web page written straight into the html.

Phone mail apps (iPhone Mail / Files preview, Outlook and Gmail viewers) show an emailed
.html attachment with scripts turned off. So everything a reader needs is written into the
file here: header, summary, job basis, Purge Report table, notes, and a still SVG of each
chart (the profile at the same moment the page opens on). In a real browser the page script
then swaps the still charts for the interactive ones; the text and tables stay as written.
"""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional

_esc = lambda s: html.escape("" if s is None else str(s), quote=True)   # noqa: E731


def fmt(v, nd: int = 0) -> str:
    """Same as the page script's fmt(): thousands commas, fixed decimals, a dash for no value."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:,.{nd}f}"


# ---------------------------------------------------------------- header + summary

def header(d: dict) -> Dict[str, str]:
    b = d.get("brand") or {}
    route = next((r[1] for r in d.get("basis", []) if r[0] == "Route"), "")
    out = {
        "EYEBROW": _esc((b["name"] + " · " if b.get("name") else "") + "Nitrogen purge simulation"),
        "H1": _esc(d.get("title") or ""),
        "SUB": _esc(" · ".join(x for x in (route, "Prepared " + str(d.get("date", ""))) if x)),
        "ROOT_STYLE": _esc("".join(f"--brand-{k}:{b[k]};" for k in ("primary", "mid", "secondary", "highlight") if b.get(k))),
        "LOGO": "",
    }
    if b.get("logo"):
        cls = "logo chip" if b.get("logo_on") == "light" else "logo"
        out["LOGO"] = f'<img class="{cls}" id="logo" src="{_esc(b["logo"])}" alt="{_esc(b.get("name") or "Logo")}">'
    return out


def summary(d: dict) -> str:
    S = d.get("summary") or {}
    sp = S.get("pig_speed_mph") or {}
    face = S.get("pig_face_psig") or {}

    def kpi(v, label):
        return f'<div class="kpi"><div class="v">{v}</div><div class="l">{_esc(label)}</div></div>'

    exit_txt = "—" if S.get("exit_psig_max") is None else f'{fmt(S.get("exit_min_psig"))} – {fmt(S.get("exit_psig_max"))}'
    kpis = "".join([
        kpi(fmt(S.get("n2_total_fresh_scf")), "Total N2 (SCF)"),
        kpi(fmt(S.get("duration_hr"), 1), "Duration (hours)"),
        kpi(fmt(sp.get("avg"), 2), "Average pig speed (mph)"),
        kpi(f'{fmt(sp.get("min"), 2)} – {fmt(sp.get("max"), 2)}', "Pig speed range (mph)"),
        kpi(fmt(face.get("max")), "Peak N2 drive (psig)"),
        kpi(fmt(S.get("injection_psig_max")), "Peak injection (psig)"),
        kpi(fmt(S.get("worst_mop_margin_psig")), "Smallest MOP margin (psi)"),
        kpi(fmt(S.get("n2_vented_scf")), "N2 vented (SCF)"),
        kpi(exit_txt, "Endpoint pressure held (psig)"),
        kpi(fmt(S.get("overspeed_steps")), "Steps over max speed"),
    ])
    flags = S.get("flags") or []
    if flags:
        flag_html = "".join(f'<div class="flag bad">{_esc(f)}</div>' for f in flags)
    else:
        flag_html = (f'<div class="flag good">The run completed to MP {fmt(S.get("final_mp"), 2)} with no N2 venting, '
                     "no slack-line risk, no MOP exceedance and the pig never over its maximum speed.</div>")
    bp = S.get("booster_plan") or {}
    boosters = ""
    if bp.get("sites_mp"):
        lines = []
        for m in bp["sites_mp"]:
            why = (bp.get("reasons") or {}).get(str(m))
            lines.append("MP " + fmt(m, 2) + (": " + _esc(why) if why else ""))
        boosters = f'<div class="card"><h3>N2 booster plan</h3><div class="small">{"<br>".join(lines)}</div></div>'
    return f'<div id="kpis" class="kpis">{kpis}</div><div id="flags">{flag_html}</div><div id="boosters">{boosters}</div>'


# ---------------------------------------------------------------- tables

def basis_rows(d: dict) -> str:
    return "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in d.get("basis", []))


def facilities(d: dict) -> str:
    fac = d.get("facilities") or []
    if not fac:
        return '<p class="muted small">No pump stations, valves or boosters on this route.</p>'
    rows = "".join(f'<tr><td>{_esc(f["type"])}</td><td>{_esc(f["name"])}</td><td class="n">{fmt(f["mp"], 2)}</td></tr>'
                   for f in fac)
    return f'<table><thead><tr><th>Type</th><th>Name</th><th class="n">MP</th></tr></thead><tbody>{rows}</tbody></table>'


def purge_table(d: dict) -> str:
    """Every row, feature rows included: the table's "showfeat" class shows or hides them
    (and the Feature column), so the checkbox works without rebuilding the table."""
    PR = d.get("purge") or {"cols": [], "rows": []}
    cols = PR["cols"]
    speed = next((i for i, c in enumerate(cols) if c["h"].startswith("Pig Speed")), -1)
    head = [f'<th class="n">{_esc(c["h"])}</th>' for c in cols]
    head.insert(1, '<th class="fc">Feature</th>')
    body = []
    for r in PR["rows"]:
        cells = [f'<td class="n{" speed" if i == speed else ""}">{fmt(v, cols[i]["nd"])}</td>' for i, v in enumerate(r["v"])]
        cells.insert(1, f'<td class="fc" style="white-space:nowrap"><b>{_esc(r.get("f"))}</b></td>')
        cls = " ".join(c for c in (r.get("flag"), "feat" if r.get("f") else "") if c)
        body.append(f'<tr class="{cls}">{"".join(cells)}</tr>' if cls else f'<tr>{"".join(cells)}</tr>')
    return f'<thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(body)}</tbody>'


def endpoint_note(d: dict) -> str:
    S = d.get("summary") or {}
    return (f'Exit Pressure is what the receiving end (pump suction, BPCV or tank inlet) holds. It rises above the '
            f'{fmt(S.get("exit_min_psig"))} psig minimum to keep the pig at or under {fmt(S.get("max_speed_mph"), 2)} mph, '
            "and to hold the column back where a high point ahead would otherwise drop below the slack floor as the "
            "liquid past it runs down to the exit.")


# ---------------------------------------------------------------- still charts (SVG)

_W, _H = 600, 330           # viewBox; text is sized for a phone-width screen
_FONT = 14
_MAXPTS = 500               # points per line, enough at this size


def _nice(raw: float) -> float:
    if not raw > 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    return next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9)), raw)


def _axis(vals: List[float], n: int = 5, zero: bool = False):
    lo, hi = min(vals), max(vals)
    if zero:
        lo = min(lo, 0.0)
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    step = _nice((hi - lo) / n)
    lo = math.floor(lo / step + 1e-9) * step
    hi = math.ceil(hi / step - 1e-9) * step
    ticks = [round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)]
    nd = next((k for k in range(5) if abs(step * 10 ** k - round(step * 10 ** k)) < 1e-6), 4)
    return lo, hi, ticks, nd


def _pairs(x, y):
    out = []
    for a, b in zip(x or [], y or []):
        ok = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        out.append((float(a), float(b)) if ok else None)
    if len(out) > _MAXPTS:
        k = math.ceil(len(out) / _MAXPTS)
        out = [p for i, p in enumerate(out) if i % k == 0 or i == len(out) - 1]
    return out


def still_chart(series: List[dict], x_label: str, y_label: str, y2_label: str = "", zero: bool = False,
                markers: Optional[List[dict]] = None, title: str = "") -> str:
    """A plain line chart. series: {name, x, y, color, axis ('left'|'right'), dash, width, fill}.
    markers: {x, label, color}. Colors are the page's CSS variables, so dark mode follows."""
    pts = [(s, _pairs(s["x"], s["y"])) for s in series]
    allx = [p[0] for _, ps in pts for p in ps if p]
    if not allx:
        return '<p class="muted small">No data</p>'
    left = [p[1] for s, ps in pts if s.get("axis", "left") == "left" for p in ps if p]
    right = [p[1] for s, ps in pts if s.get("axis") == "right" for p in ps if p]
    axes = {}
    if left:
        axes["left"] = _axis(left, zero=zero)
    if right:
        axes["right"] = _axis(right, zero=zero)
    # margins wide enough for the longest tick label (about 0.6 em a character)
    width = lambda a: max((len(fmt(t, axes[a][3])) for t in axes[a][2]), default=0) * _FONT * 0.6 if a in axes else 0  # noqa: E731
    m = {"l": 30 + width("left"), "r": (30 + width("right")) if right else 14, "t": 18, "b": 44}
    x0, x1 = min(allx), max(allx)
    if x1 <= x0:
        x1 = x0 + 1
    pw, ph = _W - m["l"] - m["r"], _H - m["t"] - m["b"]
    sx = lambda v: m["l"] + (v - x0) / (x1 - x0) * pw          # noqa: E731
    sy = lambda v, a: m["t"] + ph - (v - axes[a][0]) / (axes[a][1] - axes[a][0]) * ph   # noqa: E731
    g = []
    _, _, xt, xnd = _axis([x0, x1], n=6)
    for t in xt:
        if x0 - 1e-9 <= t <= x1 + 1e-9:
            g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{m["t"]}" y2="{_H - m["b"]}" style="stroke:var(--grid)"/>'
                     f'<text x="{sx(t):.1f}" y="{_H - m["b"] + 18}" text-anchor="middle">{fmt(t, xnd)}</text>')
    if "left" in axes:
        for t in axes["left"][2]:
            y = sy(t, "left")
            g.append(f'<line x1="{m["l"]}" x2="{_W - m["r"]}" y1="{y:.1f}" y2="{y:.1f}" style="stroke:var(--grid)"/>'
                     f'<text x="{m["l"] - 6}" y="{y + 5:.1f}" text-anchor="end">{fmt(t, axes["left"][3])}</text>')
    if "right" in axes:
        for t in axes["right"][2]:
            g.append(f'<text x="{_W - m["r"] + 6}" y="{sy(t, "right") + 5:.1f}">{fmt(t, axes["right"][3])}</text>')
    for s, ps in pts:
        a = s.get("axis", "left")
        if a not in axes:
            continue
        runs, cur = [], []
        for p in ps:
            if p is None:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(f"{sx(p[0]):.1f},{sy(p[1], a):.1f}")
        if cur:
            runs.append(cur)
        col = s.get("color", "var(--text)")
        for r in runs:
            if s.get("fill") and len(r) > 1:
                base = f'{_H - m["b"]}'
                first, last = r[0].split(",")[0], r[-1].split(",")[0]
                g.append(f'<path d="M{first},{base}L{"L".join(r)}L{last},{base}Z" style="fill:{col};fill-opacity:.12;stroke:none"/>')
            dash = f';stroke-dasharray:{s["dash"]}' if s.get("dash") else ""
            g.append(f'<path d="M{"L".join(r)}" style="fill:none;stroke:{col};stroke-width:{s.get("width", 2)}{dash};'
                     'stroke-linejoin:round"/>')
    for mk in markers or []:
        if mk.get("x") is None or not (x0 <= mk["x"] <= x1):
            continue
        x = sx(mk["x"])
        col = mk.get("color", "var(--text)")
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{m["t"]}" y2="{_H - m["b"]}" style="stroke:{col};stroke-width:1.5"/>')
        if mk.get("label"):
            anchor = "end" if x > _W - 80 else "start"
            dx = -4 if anchor == "end" else 4
            g.append(f'<text x="{x + dx:.1f}" y="{m["t"] + 12}" text-anchor="{anchor}" style="fill:var(--text)">{_esc(mk["label"])}</text>')
    g.append(f'<line x1="{m["l"]}" x2="{_W - m["r"]}" y1="{_H - m["b"]}" y2="{_H - m["b"]}" style="stroke:var(--border)"/>')
    g.append(f'<text x="{(m["l"] + _W - m["r"]) / 2:.1f}" y="{_H - 6}" text-anchor="middle">{_esc(x_label)}</text>')
    g.append(f'<text x="14" y="{m["t"] + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {m["t"] + ph / 2:.1f})">{_esc(y_label)}</text>')
    if right and y2_label:
        xr = _W - 8
        g.append(f'<text x="{xr}" y="{m["t"] + ph / 2:.1f}" text-anchor="middle" transform="rotate(90 {xr} {m["t"] + ph / 2:.1f})">{_esc(y2_label)}</text>')
    legend = "".join(f'<span><i style="background:{s.get("color")}"></i>{_esc(s["name"])}</span>' for s, ps in pts if any(ps))
    return (f'<div class="legend">{legend}</div>'
            f'<svg class="still" viewBox="0 0 {_W} {_H}" role="img" aria-label="{_esc(title)}" style="font-size:{_FONT}px">{"".join(g)}</svg>')


def _profile_series(G: dict, f: dict, pig: float) -> List[dict]:
    gx, gy, lx, ly = [], [], [], []
    for x, p in zip(G["mp"], f["p"]):
        (gx if x <= pig else lx).append(x)
        (gy if x <= pig else ly).append(p)
    if f.get("gas_end") is not None:
        gx.append(pig); gy.append(f["gas_end"])
    if f.get("liq_start") is not None:
        lx.insert(0, pig); ly.insert(0, f["liq_start"])
    out = [{"name": "N2 pressure (behind pig)", "x": gx, "y": gy, "color": "var(--gas)", "width": 2.4},
           {"name": "Liquid pressure (ahead of pig)", "x": lx, "y": ly, "color": "var(--liquid)", "width": 2.4}]
    if G.get("mop"):
        out.append({"name": "MOP", "x": G["mp"], "y": G["mop"], "color": "var(--mop)", "dash": "6 4", "width": 1.4})
    out.append({"name": "Elevation (ft, right axis)", "x": G["mp"], "y": G["elev"], "color": "var(--ground)",
                "axis": "right", "width": 1.4, "fill": True})
    return out


def frame_index(d: dict) -> int:
    """The moment the page opens on (the page script starts at the same one)."""
    n = len((d.get("profile") or {}).get("frames") or [])
    return max(0, (n - 1) // 2)


def profile_readout(f: dict) -> str:
    return (f'<b>{fmt(f.get("t"), 1)} hr</b> into the run · pig at <b>MP {fmt(f.get("pig"), 2)}</b> moving '
            f'<b>{fmt(f.get("speed"), 2)} mph</b> · N2 drive <b>{fmt(f.get("face"))} psig</b> · exit '
            f'<b>{fmt(f.get("exit_psig"))} psig</b> at MP {fmt(f.get("exit_mp"), 2)}')


def profile(d: dict) -> Dict[str, str]:
    P = d.get("profile") or {}
    frames = P.get("frames") or []
    out = {"PF_READOUT": "", "PF_CHART": '<p class="muted small">No profile data</p>', "SYS_CHART": "", "SYS_HIDDEN": "hidden"}
    if not frames:
        return out
    i = frame_index(d)
    f = frames[i]
    out["PF_READOUT"] = profile_readout(f)
    out["PF_CHART"] = still_chart(_profile_series(P, f, f["pig"]), "Milepost", "psig", "Elevation (ft)",
                                  markers=[{"x": f["pig"], "label": "Pig"}], title="Pipeline profile")
    Y = P.get("system")
    if Y and i < len(Y.get("frames") or []):
        sf = Y["frames"][i]
        series = _profile_series(Y, sf, f["pig"])
        fl = Y.get("floor", 25)
        series.insert(-1, {"name": f"Slack floor ({fmt(fl)} psig)", "x": [Y["mp"][0], Y["mp"][-1]], "y": [fl, fl],
                           "color": "var(--bad)", "dash": "2 3", "width": 1})
        out["SYS_CHART"] = still_chart(series, "Milepost", "psig", "Elevation (ft)", title="Whole hydraulic system",
                                       markers=[{"x": f["pig"], "label": "Pig"},
                                                {"x": Y.get("stop_mp"), "label": "Pig stop", "color": "var(--muted)"}])
        out["SYS_HIDDEN"] = ""
        out["SYS_TITLE"] = _esc(f'Whole hydraulic system, launch to the hydraulic endpoint at MP {fmt(Y.get("end_mp"), 2)}')
    return out


def charts(d: dict) -> str:
    SE = d.get("series") or {}
    mp, t = SE.get("pig_mp") or [], SE.get("t_hr") or []
    specs = [
        ("Pressures along the route", mp, "Pig milepost", "psig", "", False, [
            {"name": "Injection", "y": SE.get("injection_psig"), "color": "var(--s2)", "width": 3},
            {"name": "Pig face (N2 drive)", "y": SE.get("pig_face_psig"), "color": "var(--gas)"},
            {"name": "Exit", "y": SE.get("exit_psig"), "color": "var(--liquid)", "dash": "5 4"}]),
        ("Pig speed along the route", mp, "Pig milepost", "mph", "", False, [
            {"name": "Pig speed", "y": SE.get("pig_speed_mph"), "color": "var(--gas)"}]),
        ("Nitrogen over time", t, "Hours", "SCF in line", "Injection SCFM", True, [
            {"name": "N2 in line", "y": SE.get("total_scf"), "color": "var(--s4)", "fill": True},
            {"name": "Injection rate", "y": SE.get("injection_scfm"), "color": "var(--s2)", "axis": "right"}]),
        ("Pig position over time", t, "Hours", "Milepost", "", False, [
            {"name": "Pig milepost", "y": mp, "color": "var(--gas)"}]),
    ]
    cards = []
    for title, x, xl, yl, y2l, zero, ser in specs:
        ser = [dict(s, x=x) for s in ser if s.get("y")]
        cards.append(f'<div class="card"><h3>{_esc(title)}</h3><div class="chart">'
                     f'{still_chart(ser, xl, yl, y2l, zero=zero, title=title)}</div></div>')
    return "".join(cards)


def fill(d: dict) -> Dict[str, str]:
    """Every __REPORT_<KEY>__ placeholder of the page template except TITLE and DATA."""
    PR = d.get("purge") or {}
    out = header(d)
    out.update({
        "SUMMARY": summary(d),
        "BASIS": basis_rows(d),
        "FACILITIES": facilities(d),
        "PURGE": purge_table(d),
        "PURGE_CLASS": "purge showfeat" if PR.get("show_features") else "purge",
        "FEAT_HIDDEN": "" if PR.get("has_features") else "hidden",
        "FEAT_CHECKED": "checked" if PR.get("show_features") else "",
        "ENDPOINT_NOTE": endpoint_note(d) if PR.get("has_endpoint") else "",
        "ENDPOINT_HIDDEN": "" if PR.get("has_endpoint") else "hidden",
        "NOTES": _esc(d.get("notes") or ""),
        "NOTES_HIDDEN": "" if d.get("notes") else "hidden",
        "FOOT": f'Prepared {_esc(d.get("date"))} with Purge Simulator. Results are simulation estimates based on the job basis listed above.',
        "CHARTS": charts(d),
        "SYS_TITLE": "",
    })
    out.update(profile(d))
    return out
