"""The client web page reads with scripts off (phone mail previews): text, tables and still
charts are written into the html, not built by the page script."""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from purge_sim.app.client_report import build_data, render_html   # noqa: E402
from purge_sim.app.workspace import Workspace                     # noqa: E402

SHORT = "bundled:CHS_TipvilleSantaRita_East10/tipville_east10_3mph.json"


def test_page_content_is_in_the_html_without_the_script():
    ws = Workspace()
    ws.load(SHORT)
    res = ws.run()
    data = build_data(res, "Job <A> & B", inputs=ws.scenario.inputs, notes="tender <note>")
    html = render_html(data)
    body = html.split('<script type="application/json"')[0]      # what a no-script preview shows
    assert not re.search(r"__REPORT_[A-Z0-9_]+__", body)
    assert "Job &lt;A&gt; &amp; B" in body and "tender &lt;note&gt;" in body
    assert "Total N2 (SCF)" in body and f'{data["summary"]["n2_total_fresh_scf"]:,.0f}' in body
    assert body.count("<tr") >= len(data["purge"]["rows"]) + len(data["basis"])
    assert body.count('<svg class="still"') == 5                 # profile + the four charts
    assert "still copy of this report" in body and 'classList.add("js")' in html
