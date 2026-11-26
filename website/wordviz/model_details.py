"""Collection records for the datasets used by the explorer and analyses."""

import json
from pathlib import Path

from dash import html

from .model_options import ORIGINS, pinned_label, pinned_models
from .model_provenance import MODEL_PROVENANCE, REPO
from .models import GRAPH_MODELS, MATRIX_MODELS, MODEL_LABELS

RELEASE = json.loads((Path(__file__).with_name("resources") / "network_release.json").read_text())


def _record(key, record):
    human = key == "humans"
    identifier_label = record.get("identifier_label") or ("Data source" if human else "Model identifier")
    identifier = record.get("identifier") or ("Human participants" if human else "Unrecorded")
    if key in {name.lower() for name in MATRIX_MODELS}:
        usage = "Explore, Lexical decision and Implicit bias"
    elif key in GRAPH_MODELS:
        usage = "Explore and Association paths"
    else:
        usage = record.get("availability", "Collected responses")

    fields = [("Availability", usage)]
    if record.get("snapshot"):
        fields.append(("Snapshot mapped from alias", html.Code(record["snapshot"])))
    rho = pinned_models().get(key)
    if rho is not None:
        fields.append(("Pinned model", f"{ORIGINS[key]} · Table 6 Spearman ρ = {rho:.3f}"))
    if key in RELEASE["networks"]:
        url = f"{REPO}/releases/download/{RELEASE['tag']}/{key}.zip"
        size_mb = RELEASE["networks"][key]["bytes"] / 1_000_000
        fields.append(("Research data", html.A(
            f"Cleaned trials, graph and activation matrices (ZIP, {size_mb:.1f} MB)", href=url,
        )))
    if not human or record.get("revision"):
        fields.append(("Version or revision", record.get("revision") or "Unrecorded"))
    if record.get("collection"):
        fields.append(("Collection", record["collection"]))
    fields.extend(record.get("details", ()))
    if record.get("notes"):
        fields.append(("Notes", record["notes"]))
    sources = record["sources"]
    if sources:
        fields.append(("Sources", html.Ul(
            [html.Li(html.A(label, href=url)) for label, url in sources],
            className="model-sources",
        )))

    name = pinned_label(record["label"]) if rho is not None else html.Span(record["label"], className="model-name")
    if key not in GRAPH_MODELS:
        name = html.Span([name, html.Span(usage, className="model-field-label")])
    return html.Details([
        html.Summary([
            name,
            html.Span([html.Span(identifier_label, className="model-field-label"),
                       html.Code(identifier)], className="model-identifier"),
        ]),
        html.Dl([html.Div([html.Dt(label), html.Dd(value)]) for label, value in fields],
                className="model-fields"),
    ], id=f"model-{key}", className="model-record")


def layout():
    pin_note = html.P([
        "Stars mark one release per original developer. Among documented developer APIs and original "
        "checkpoints with complete lexical-decision data, these models have the strongest negative "
        "correlations between activation and human reaction time across the 50 triplets. ",
        html.A("View Table 6", href="/lexical-decision"),
    ], className="source-note") if pinned_models() else None
    return html.Div([
        pin_note,
        html.Div([_record(key, MODEL_PROVENANCE[key]) for key in MODEL_LABELS], className="model-list"),
    ], className="model-catalog")
