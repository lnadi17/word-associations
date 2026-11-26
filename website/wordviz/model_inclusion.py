"""Shared network selection for the cross-model implicit-association figures."""

from dash import ALL, Input, Output, State, dcc, html

from .custom_domains import NEW_DOMAIN
from .models import MATRIX_MODELS, ordered_model_keys


def included_models(saved):
    """Keep canonical matrix keys and preserve an intentionally empty selection."""
    if not isinstance(saved, list):
        return list(MATRIX_MODELS)
    return [key for key in MATRIX_MODELS if key in saved]


def network_groups():
    families = {}
    for key in MATRIX_MODELS:
        family = next((name for name in ("Gemma", "Llama", "Nemotron", "Qwen")
                       if key.lower().startswith(name.lower())), "Individual networks")
        families.setdefault(family, []).append(key)
    return dict(sorted(families.items()))


def _toggle(label, control_id, accessible_name):
    return html.Button(label, id=control_id, type="button", n_clicks=0, role="checkbox",
                       className="inclusion-toggle",
                       **{"aria-label": accessible_name, "aria-checked": "true"})


def layout():
    groups = []
    for family, members in network_groups().items():
        name = family if family == "Individual networks" else f"{family} family"
        group_id = {"type": "network-family", "family": family}
        groups.append(html.Fieldset([
            html.Legend(_toggle(family, group_id, name)),
            dcc.Checklist(id={"type": "network-members", "family": family},
                          options=[{"label": MATRIX_MODELS[key], "value": key} for key in members],
                          value=members, className="inclusion-members", labelStyle={"display": "flex"}),
        ]))
    count = len(MATRIX_MODELS)
    return html.Div([
        dcc.Store(id="included-models", storage_type="local"),
        html.Details([
            html.Summary(["Networks shown", html.Span(f"{count} of {count}", id="inclusion-count")]),
            _toggle("All networks", "inclusion-all", "All networks"),
            html.Div(groups, className="inclusion-families"),
        ], id="model-inclusion-picker", hidden=True),
        html.P(id="inclusion-highlight-status", className="source-note", role="status", hidden=True),
    ])


def register(app):
    @app.callback(Output("model-inclusion-picker", "hidden"),
                  Output("inclusion-highlight-status", "children"), Output("inclusion-highlight-status", "hidden"),
                  Input("bias-view", "value"), Input("social-domain", "value"), Input("included-models", "data"),
                  Input("social-a", "value"), Input("social-b", "value"))
    def picker_state(method, domain, saved, first, second):
        hidden = method not in ("priming", "valence", "emotion") or domain == NEW_DOMAIN
        selected = included_models(saved)
        excluded = ordered_model_keys({key for key in (first, second) if key in MATRIX_MODELS and key not in selected})
        if hidden or not excluded:
            return hidden, None, True
        names = " and ".join(MATRIX_MODELS[key] for key in excluded)
        action = "Include it to show the highlight." if len(excluded) == 1 else "Include them to show both highlights."
        return False, f"Your network selection excludes {names}. {action}", False

    family = {"type": "network-family", "family": ALL}
    members = {"type": "network-members", "family": ALL}
    app.clientside_callback(
        """function(saved, allClicks, familyClicks, checked, memberIds, options, familyIds) {
            const groups = Object.fromEntries(memberIds.map((id, i) =>
                [id.family, options[i].map(option => option.value)]));
            const all = Object.values(groups).flat();
            const active = new Set(Array.isArray(saved) ? saved : all);
            const trigger = dash_clientside.callback_context.triggered_id;
            function toggle(keys) {
                const remove = keys.every(key => active.has(key));
                for (const key of keys) {
                    if (remove) active.delete(key);
                    else active.add(key);
                }
            }
            if (trigger === 'inclusion-all') {
                toggle(all);
            } else if (trigger?.type === 'network-family') {
                toggle(groups[trigger.family]);
            } else if (trigger?.type === 'network-members') {
                const index = memberIds.findIndex(id => id.family === trigger.family);
                for (const key of groups[trigger.family]) active.delete(key);
                for (const key of checked[index]) active.add(key);
            }
            function state(keys) {
                const count = keys.filter(key => active.has(key)).length;
                return count === 0 ? 'false' : count === keys.length ? 'true' : 'mixed';
            }
            const selection = all.filter(key => active.has(key));
            // An unrestricted cohort automatically includes newly published networks.
            const unrestricted = selection.length === all.length;
            const same = unrestricted ? saved == null : Array.isArray(saved)
                && saved.length === selection.length && saved.every((key, i) => key === selection[i]);
            const next = unrestricted ? null : selection;
            return [same ? dash_clientside.no_update : next,
                    selection.length + ' of ' + all.length, state(all),
                    familyIds.map(id => state(groups[id.family])),
                    memberIds.map(id => groups[id.family].filter(key => active.has(key)))];
        }""",
        Output("included-models", "data"), Output("inclusion-count", "children"),
        Output("inclusion-all", "aria-checked"), Output(family, "aria-checked"), Output(members, "value"),
        Input("included-models", "data"), Input("inclusion-all", "n_clicks"),
        Input(family, "n_clicks"), Input(members, "value"),
        State(members, "id"), State(members, "options"), State(family, "id"),
    )
