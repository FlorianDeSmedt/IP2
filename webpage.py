"""
Write the one-screen web page for a script (toll_bo.py, toll_bes.py).

The page embeds the script's own source code. Its "Run" button re-runs that
source in the browser with Pyodide (Python compiled to WebAssembly, no
server), passing the form values in WEB_PARAMS, and redraws the plotly
figure the script leaves in the variable `fig`.

Only imported when a script runs locally, never in the browser.
"""
import json
from pathlib import Path

import plotly
import plotly.offline

HERE = Path(__file__).resolve().parent


def write_page(script, out_name, title, intro, params, param_help, fig):
    """Fill page_template.html and save it next to the script.

    Parameters
    ----------
    script     : path of the script (pass __file__); its source is embedded
    out_name   : file name of the page, e.g. "toll_bo.html"
    title      : page heading
    intro      : one line of HTML under the heading
    params     : dict of default parameters (the form starts with these)
    param_help : dict  name -> (group, short description) for the form
    fig        : the plotly figure to show before the first browser run
    """
    def as_js(obj_json):
        # "</" would end the <script> block early; "<\/" is the same JSON string.
        return obj_json.replace("</", "<\\/")

    page = (HERE / "page_template.html").read_text(encoding="utf-8")
    page = (page.replace("{{TITLE}}", title)
                .replace("{{INTRO}}", intro)
                .replace("{{PLOTLYJS_VERSION}}", plotly.offline.get_plotlyjs_version())
                .replace("{{PLOTLY_VERSION}}", plotly.__version__)
                .replace("{{PARAMS}}", as_js(json.dumps(params)))
                .replace("{{HELP}}", as_js(json.dumps(param_help)))
                .replace("{{FIGURE}}", as_js(fig.to_json()))
                # last, so placeholders inside the source are left alone
                .replace("{{SOURCE}}", as_js(json.dumps(Path(script).read_text(encoding="utf-8")))))
    (HERE / out_name).write_text(page, encoding="utf-8")
    print(f"Interactive version written to {out_name}")
