"""Collection records for the datasets used by the explorer and analyses."""

import json
from pathlib import Path

from dash import html

from .model_options import ORIGINS, pinned_models
from .model_provenance import MODEL_PROVENANCE, REPO
from .models import GRAPH_MODELS, MATRIX_MODELS

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

    return html.Details([
        html.Summary([
            html.Span(record["label"], className="model-name"),
            html.Span([html.Span(identifier_label, className="model-field-label"),
                       html.Code(identifier)], className="model-identifier"),
        ]),
        html.Dl([html.Div([html.Dt(label), html.Dd(value)]) for label, value in fields],
                className="model-fields"),
    ], id=f"model-{key}", className="model-record")


def layout():
    pins = pinned_models()
    available, additional = [], []
    for key, record in MODEL_PROVENANCE.items():
        if key in pins:
            continue
        group = available if key in GRAPH_MODELS else additional
        group.append(_record(key, record))
    sections = []
    if pins:
        sections.append(html.Section([
            html.H2("Pinned models"),
            html.P([
                "Stars mark one release per original developer. Among documented developer APIs and original "
                "checkpoints with complete lexical-decision data, these models have the strongest negative "
                "correlations between activation and human reaction time across the 50 triplets. ",
                html.A("View Table 6", href="/lexical-decision"),
            ], className="source-note"),
            html.Div([_record(key, MODEL_PROVENANCE[key]) for key in pins], className="model-list"),
        ]))
    sections.extend(
        html.Section([html.H2(heading), html.Div(records, className="model-list")])
        for heading, records in (("Analysis datasets", available), ("Additional collections", additional))
        if records
    )
    return html.Div(sections, className="model-catalog")
