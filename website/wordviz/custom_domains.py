"""Browser-local social domains and their editable target-word lists."""

import json
import re
from uuid import uuid4

from dash import ALL, Input, Output, Patch, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .social_domains import DOMAIN_LABELS, DOMAIN_PRIMES

NEW_DOMAIN = "new-custom-domain"
MAX_PRIMES = 12
MAX_TARGETS = 100
MAX_LISTS = 12
MAX_DOMAINS = 20
NAME_LIMIT = 80
WORD_LIMIT = 64
_ID = re.compile(r"custom-[0-9a-f]{32}\Z")


def _label(raw, field):
    if not isinstance(raw, str) or len(raw) > NAME_LIMIT or not raw.strip():
        raise ValueError(f"Enter a {field} of 1–{NAME_LIMIT} characters.")
    return " ".join(raw.split())


def _words(raw, field, minimum, maximum):
    if not isinstance(raw, list) or len(raw) > 2 * maximum + 1:
        raise ValueError(f"Use {minimum}–{maximum} distinct words for {field}.")
    words = {}
    for word in raw:
        if not isinstance(word, str) or len(word) > WORD_LIMIT:
            raise ValueError(f"Each word in {field} can contain up to {WORD_LIMIT} characters.")
        word = " ".join(word.casefold().split())
        if word:
            words[word] = None
    if not minimum <= len(words) <= maximum:
        raise ValueError(f"Use {minimum}–{maximum} distinct words for {field}.")
    return list(words)


def _domain(key, name, primes, pairs):
    if not isinstance(key, str) or not _ID.fullmatch(key):
        raise ValueError("Select a saved domain or create a new one.")
    name = _label(name, "domain name")
    if name.casefold() in {label.casefold() for label in DOMAIN_LABELS.values()}:
        raise ValueError("Choose a name distinct from the published domains.")
    if len(pairs) > MAX_LISTS:
        raise ValueError(f"Use up to {MAX_LISTS} named word lists.")
    primes = _words(primes, "primes", 2, MAX_PRIMES)
    lists = {}
    for title, words in pairs:
        title = _label(title, "word-list name")
        if title.casefold() in {label.casefold() for label in lists}:
            raise ValueError("Give each word list a distinct name.")
        lists[title] = _words(words, title, 1, MAX_TARGETS)
    return {"id": key, "name": name, "primes": primes, "lists": lists}


def validate_domain(value):
    """Return a canonical record or None for malformed browser state."""
    if not isinstance(value, dict):
        return None
    lists = value.get("lists", {})
    if not isinstance(lists, dict):
        return None
    try:
        return _domain(value.get("id"), value.get("name"), value.get("primes"), lists.items())
    except ValueError:
        return None


def get_domain(data, key):
    if not isinstance(key, str) or not _ID.fullmatch(key) or not isinstance(data, dict):
        return None
    record = validate_domain(data.get(key))
    return record if record and record["id"] == key else None


def _library(data):
    if not isinstance(data, dict) or len(data) > MAX_DOMAINS:
        return {}
    return {key: record for key in data if (record := get_domain(data, key))}


def _split(text, maximum):
    if not isinstance(text, str) or len(text) > (WORD_LIMIT + 2) * maximum:
        raise ValueError("Enter words separated by commas or new lines, within the displayed limits.")
    return re.split(r"[,;\n]", text)


def save_domain(data, original, name, prime_text, titles, target_texts):
    library = _library(data)
    key = original["id"] if original else "custom-" + uuid4().hex
    if original and library.get(key) != original:
        raise ValueError("This domain changed in another tab. Reopen it before saving.")
    if not original and len(library) >= MAX_DOMAINS:
        raise ValueError(f"This browser already has {MAX_DOMAINS} custom domains.")
    if not isinstance(titles, list) or not isinstance(target_texts, list) or len(titles) != len(target_texts):
        raise ValueError("Enter a name and words for each list.")
    if len(titles) > MAX_LISTS:
        raise ValueError(f"Use up to {MAX_LISTS} named word lists.")
    pairs = []
    for title, words in zip(titles, target_texts):
        if isinstance(title, str) and isinstance(words, str) and not (title.strip() or words.strip()):
            continue
        pairs.append((title, _split(words, MAX_TARGETS)))
    record = _domain(key, name, _split(prime_text, MAX_PRIMES), pairs)
    if any(d["name"].casefold() == record["name"].casefold() for k, d in library.items() if k != key):
        raise ValueError("A saved domain already has that name.")
    library[key] = record
    return library, key


def _field(label, control):
    component_id = control.id
    if isinstance(component_id, dict):
        component_id = json.dumps(component_id, sort_keys=True, separators=(",", ":"))
    return html.Div([html.Label(label, htmlFor=component_id), control], className="field")


def _list_row(title="", words=()):
    key = uuid4().hex
    return html.Div([
        _field("Word-list name", dcc.Input(id={"type": "domain-list-name", "key": key},
               type="text", value=title, maxLength=NAME_LIMIT, style={"width": "100%"})),
        _field("Target words", dcc.Textarea(id={"type": "domain-list-words", "key": key},
               value=", ".join(words), placeholder="calm, anxious, curious", rows=2,
               maxLength=(WORD_LIMIT + 2) * MAX_TARGETS, style={"width": "100%"})),
        html.Button("Remove list", id={"type": "domain-list-remove", "key": key}, n_clicks=0),
    ], className="controls", key=key)


def layout():
    return html.Div([
        dcc.Store(id="custom-domains", storage_type="local"),
        dcc.Store(id="custom-last-domain", storage_type="local"),
        dcc.Store(id="custom-action"),
        dcc.Store(id="custom-ready"),
        html.Details([html.Summary("Add a custom domain", id="custom-summary"),
                      html.Div(id="custom-form")], className="panel", id="custom-manager"),
    ])


def _form(record):
    creating = not record
    lists = {} if creating else record["lists"]
    primes = [] if creating else record["primes"]
    template = [] if record else [_field("Start with primes from", dcc.Dropdown(
        id="custom-template", clearable=False, searchable=False, placeholder="Blank list",
        options=[{"value": "", "label": "Blank list"}] + [
            {"value": key, "label": name} for key, name in DOMAIN_LABELS.items()]))]
    delete = [] if creating else [dcc.ConfirmDialogProvider(
        html.Button("Delete domain"), id="custom-delete",
        message=f'Delete “{record["name"]}” and its word lists from this browser?')]
    return [
        dcc.Store(id="custom-original", data=record),
        html.P("Save domains in this browser. Enter words separated by commas or new lines."),
        html.Div([
            _field("Domain name", dcc.Input(id="custom-name", type="text", maxLength=NAME_LIMIT,
                   value="" if creating else record["name"], placeholder="e.g. Age",
                   style={"width": "100%"})),
            *template,
            _field(f"Prime words (2–{MAX_PRIMES})", dcc.Textarea(id="custom-primes",
                   value=", ".join(primes), rows=2, placeholder="young, old",
                   maxLength=(WORD_LIMIT + 2) * MAX_PRIMES, style={"width": "100%"})),
        ], className="controls"),
        html.H3("Target word lists (optional)"),
        html.P("Word valence and Emotion word sets use your primes with their own reference words. "
               f"For the other views, you can add up to {MAX_LISTS} lists of {MAX_TARGETS} words each, now or later.",
               className="hint"),
        html.Div([_list_row(title, words) for title, words in lists.items()], id="custom-list-rows"),
        html.Div([html.Button("Add word list", id="custom-add-list", n_clicks=0,
                              disabled=len(lists) >= MAX_LISTS),
                  html.Button("Save and use domain", id="custom-save", n_clicks=0), *delete],
                 style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),
        html.Div(id="custom-error", role="status"),
    ]


def register(app):
    @app.callback(Output("social-domain", "options"), Output("social-domain", "value"),
                  Output("custom-ready", "data"), Input("custom-domains", "modified_timestamp"), Input("custom-action", "data"),
                  Input("custom-summary", "n_clicks"),
                  State("custom-domains", "data"), State("custom-last-domain", "data"),
                  State("social-domain", "value"), State("custom-ready", "data"))
    def domain_options(_timestamp, action, open_editor, data, last, current, ready):
        if set(ctx.triggered_prop_ids) == {"custom-summary.n_clicks"}:
            if open_editor and current in DOMAIN_LABELS:
                return no_update, NEW_DOMAIN, no_update
            raise PreventUpdate
        library = _library(data)
        options = [{"label": name, "value": key} for key, name in DOMAIN_LABELS.items()]
        options += [{"label": record["name"], "value": key} for key, record in library.items()]
        options.append({"label": "Add a custom domain…", "value": NEW_DOMAIN})
        selected = current if ready else (last or current)
        if "custom-action.data" in ctx.triggered_prop_ids and isinstance(action, dict):
            selected = action.get("select")
        valid = isinstance(selected, str) and selected in set(DOMAIN_LABELS) | set(library) | {NEW_DOMAIN}
        selected = selected if valid else "gender"
        return options, no_update if ready and selected == current else selected, True

    app.clientside_callback(
        """function(key) {
            if (!key || key === 'new-custom-domain') return window.dash_clientside.no_update;
            return key;
        }""",
        Output("custom-last-domain", "data"), Input("social-domain", "value"), prevent_initial_call=True,
    )

    @app.callback(Output("custom-form", "children"), Output("custom-summary", "children"),
                  Output("custom-manager", "open"),
                  Input("social-domain", "value"), Input("custom-action", "data"),
                  State("custom-domains", "data"))
    def show_editor(key, _action, data):
        record = get_domain(data, key)
        label = f'Edit {record["name"]}' if record else "Add a custom domain"
        if not record and key != NEW_DOMAIN:
            return None, label, False
        return html.Div(_form(record), key=key), label, key == NEW_DOMAIN

    @app.callback(Output("custom-primes", "value"), Input("custom-template", "value", allow_optional=True),
                  prevent_initial_call=True)
    def fill_primes(template):
        if template is None:
            raise PreventUpdate
        return ", ".join(DOMAIN_PRIMES.get(template, ()))

    @app.callback(Output("custom-list-rows", "children"), Output("custom-add-list", "disabled"),
                  Input("custom-add-list", "n_clicks"),
                  Input({"type": "domain-list-remove", "key": ALL}, "n_clicks"),
                  State({"type": "domain-list-remove", "key": ALL}, "id"), prevent_initial_call=True)
    def change_lists(add, clicks, ids):
        trigger = ctx.triggered_id
        patch = Patch()
        if trigger == "custom-add-list" and add:
            if len(ids) >= MAX_LISTS:
                return no_update, True
            patch.append(_list_row())
            return patch, len(ids) + 1 >= MAX_LISTS
        elif isinstance(trigger, dict) and trigger in ids and clicks[ids.index(trigger)]:
            del patch[ids.index(trigger)]
            return patch, False
        raise PreventUpdate

    @app.callback(Output("custom-domains", "data"), Output("custom-action", "data"),
                  Output("custom-error", "children"),
                  Input("custom-save", "n_clicks"), Input("custom-delete", "submit_n_clicks", allow_optional=True),
                  State("custom-domains", "data"), State("custom-original", "data"),
                  State("custom-name", "value"), State("custom-primes", "value"),
                  State({"type": "domain-list-name", "key": ALL}, "value"),
                  State({"type": "domain-list-words", "key": ALL}, "value"),
                  prevent_initial_call=True)
    def change_domain(save, delete, data, original, name, primes, titles, words):
        try:
            if original is not None:
                original = validate_domain(original)
                if not original:
                    raise ValueError("Reopen the domain to edit its saved words.")
            if ctx.triggered_id == "custom-save" and save:
                library, key = save_domain(data, original, name, primes, titles, words)
            elif ctx.triggered_id == "custom-delete" and delete and original:
                library = _library(data)
                if library.get(original["id"]) != original:
                    raise ValueError("This domain changed in another tab. Reopen it before deleting.")
                del library[original["id"]]
                key = "gender"
            else:
                raise PreventUpdate
        except ValueError as exc:
            return no_update, no_update, str(exc)
        return library, {"select": key, "event": uuid4().hex}, ""
