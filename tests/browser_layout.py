"""Run rendered chart and dropdown regressions against a running website."""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
from wordviz.models import GRAPH_MODELS, MATRIX_MODELS

from playwright.sync_api import expect, sync_playwright


def check_chart(host, mark, minimum_height):
    """A collapsed Plotly figure can still contain every expected SVG element."""
    expect(host.locator(mark).first).to_be_visible()
    size = host.bounding_box()
    plot = host.locator(".nsewdrag").first.bounding_box()
    assert size and size["height"] >= minimum_height, (host, size)
    assert plot and plot["height"] > 100 and plot["width"] > 100, (host, plot)


def check_width(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def check_touch_scroll(page, results):
    if page.viewport_size["width"] > 400 or page.context.browser.browser_type.name != "chromium":
        return
    region = results.get_by_role("region", name=re.compile("chart$"))
    if not region.evaluate("el => el.scrollWidth > el.clientWidth"):
        return
    region.scroll_into_view_if_needed()
    rect = region.bounding_box()
    start = rect["x"] + rect["width"] * .85
    y = rect["y"] + rect["height"] * .45
    touch = page.context.new_cdp_session(page)
    try:
        touch.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": start, "y": y}]})
        for distance in (20, 40, 70, 100, 140):
            touch.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": start - distance, "y": y}]})
        touch.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        page.wait_for_function("el => el.scrollLeft > 50", arg=region.element_handle())
    finally:
        touch.detach()


def check_network_inclusion(page, methods):
    count = len(MATRIX_MODELS)
    family_count = sum(key.lower().startswith("gemma") for key in MATRIX_MODELS)
    picker = page.locator("#model-inclusion-picker")
    valence = page.locator("#valence-results")
    emotion = page.locator("#emotion-results")
    traces = ".scatterlayer .trace"
    all_models = picker.get_by_role("checkbox", name="All networks", exact=True)
    gemma = picker.get_by_role("checkbox", name="Gemma family", exact=True)
    picker.locator("summary").click()
    expect(all_models).to_be_checked()
    all_models.click()
    expect(valence.get_by_role("status")).to_be_visible()
    expect(valence.locator(".dash-graph")).to_have_count(0)
    gemma.click()
    expect(gemma).to_be_checked()
    expect(valence.locator(traces)).to_have_count(family_count)
    expect(valence.locator("table").first.locator(".selected-row")).to_have_count(0)

    # Dropdowns highlight only included models. An excluded primary stays excluded.
    marker_size = valence.locator(".js-plotly-plot").evaluate(
        "el => el.data.find(trace => trace.name === 'Gemma').marker.size"
    )
    search_model(page, "social-a", "Gemma")
    expect(valence.locator("table").first.locator(".selected-row")).to_have_text([re.compile("Gemma")])
    expect(valence.locator(".series-legend")).to_contain_text("Gemma")
    page.wait_for_function("""oldSize => {
        const traces = document.querySelector('#valence-results .js-plotly-plot')?.data;
        return traces?.find(trace => trace.name === 'Gemma')?.marker.size > oldSize;
    }""", arg=marker_size)
    if family_count > 1:
        picker.get_by_role("checkbox", name="Gemma", exact=True).click()
        expect(gemma).to_have_attribute("aria-checked", "mixed")
        expect(valence.locator(traces)).to_have_count(family_count - 1)
        expect(valence.locator("table").first.locator(".selected-row")).to_have_count(0)
        family_count -= 1
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(emotion.locator(traces)).to_have_count(family_count)
    expect(emotion.locator("table tbody tr")).to_have_count(8 * family_count)

    page.reload()
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(emotion.locator(traces)).to_have_count(family_count)
    methods.get_by_text("Priming effects", exact=True).click()
    expect(page.locator("#effect-results .dash-graph").first.locator(".ytick")).to_have_count(family_count)
    if not picker.evaluate("el => el.open"):
        picker.locator("summary").click()
    all_models.click()
    expect(all_models).to_be_checked()
    picker.locator("summary").click()
    search_model(page, "social-a", "Humans")
    search_model(page, "social-b", "GPT-5 mini")
    methods.get_by_text("Word valence", exact=True).click()
    expect(valence.locator(traces)).to_have_count(count)
    check_width(page)


def choose(page, control, option):
    dropdown = page.locator("#" + control)
    dropdown.locator(".Select-control").click()
    dropdown.locator(".Select-menu-outer").get_by_text(option, exact=True).click()


def check_religion_targets(page, methods):
    targets = page.locator("#bias-targets-0")
    chips = targets.locator(".Select-value-label")
    effect = page.locator("#effect-results")
    source = page.locator("#target-editor a")
    default_words = ["religious", "faithful", "holy", "pure", "moral", "modest"]
    aggression_words = ["aggressive", "angry", "violent", "dangerous", "mean"]
    expect(chips).to_have_text(default_words)
    expect(effect).to_contain_text("6/6 targets")
    expect(source).to_be_visible()
    expect(source).to_have_attribute("href", re.compile("13684302221138036"))

    choose(page, "target-preset", "Perceived warmth")
    expect(chips).to_have_text(["benevolent", "altruistic", "honest", "sincere", "sociable", "warm"])
    expect(source).to_have_attribute("href", re.compile("S1755048326100601"))
    choose(page, "target-preset", re.compile("aggress", re.I))
    expect(chips).to_have_text(aggression_words)
    expect(effect).to_contain_text("5/5 targets")
    methods.get_by_text("Heatmaps", exact=True).click()
    heatmaps = page.locator("#heat-results .dash-graph")
    expect(heatmaps).to_have_count(2)
    expect(chips).to_have_text(aggression_words)
    expect(source).to_have_attribute("href", re.compile("13684302221138036"))
    for heatmap in heatmaps.all():
        check_chart(heatmap, ".heatmaplayer image", 320)
        expect(heatmap.locator(".ytick text")).to_have_text(aggression_words)

    targets.locator(".Select-value").filter(has_text="violent").locator(".Select-value-icon").click()
    edited_words = [word for word in aggression_words if word != "violent"]
    for heatmap in heatmaps.all():
        expect(heatmap.locator(".ytick text")).to_have_text(edited_words)
    methods.get_by_text("Priming effects", exact=True).click()
    expect(chips).to_have_text(edited_words)
    expect(effect).to_contain_text("4/4 targets")

    page.locator("#target-reset").click()
    expect(chips).to_have_text(aggression_words)
    expect(effect).to_contain_text("5/5 targets")
    methods.get_by_text("Heatmaps", exact=True).click()
    expect(heatmaps).to_have_count(2)
    for heatmap in heatmaps.all():
        expect(heatmap.locator(".ytick text")).to_have_text(aggression_words)
    check_width(page)


def check_page(page, url):
    page.goto(url)
    check_chart(page.locator(".bar-scroll .dash-graph"), ".barlayer .point", 300)
    compare = page.get_by_role("combobox", name="Compare model", exact=True)
    expect(compare).to_be_visible()
    page.get_by_text("Compare model", exact=True).click()
    expect(compare).to_be_focused()
    expect(compare).to_have_css("outline-style", "none")
    expect(page.locator("#ex-model-b .Select-control")).not_to_have_css("box-shadow", "none")
    compare.fill("pinned")
    menu = page.locator("#ex-model-b .Select-menu-outer")
    expect(menu.locator(".VirtualizedSelectOption").first).to_contain_text("GPT-5 mini")
    expect(menu.locator(".model-pin")).to_have_count(5)
    compare.fill("OpenAI")
    expect(menu.locator(".VirtualizedSelectOption")).to_have_count(1)
    compare.press("Enter")
    expect(page.locator("#ex-model-b .Select-value-label")).to_contain_text("GPT-5 mini")
    expect(page.locator(".series-legend li")).to_have_count(2)
    check_chart(page.locator(".bar-scroll .dash-graph"), ".barlayer .point", 300)

    # Non-searchable selects also respond when their visible label is clicked.
    page.get_by_text("Display limit", exact=True).click()
    expect(page.get_by_role("combobox", name="Display limit")).to_be_focused()
    page.keyboard.press("Escape")
    page.get_by_text("2D graph", exact=True).click()
    networks = page.locator(".explorer-network .dash-graph")
    expect(networks).to_have_count(2)
    for network in networks.all():
        check_chart(network, ".scatterlayer .point", 295)
    check_width(page)

    page.get_by_role("link", name="Lexical decision").click()
    check_chart(page.locator("#ldt-results .dash-graph"), ".boxlayer .point", 380)
    expect(page.get_by_text("Table 6 · Priming and reaction time")).to_be_visible()
    check_width(page)

    page.get_by_role("link", name="Implicit bias").click()
    effects = page.locator("#effect-results .dash-graph")
    expect(effects).to_have_count(2)
    for effect in effects.all():
        check_chart(effect, ".barlayer .point path", 320)
    choose(page, "social-b", "GPT-5 mini")
    choose(page, "social-domain", "Religion")
    paired_title = page.locator("#target-preset .Select-value-label")
    expect(page.locator("#bias-targets-1 .Select-value-label")).to_have_count(25)
    expect(page.locator("#bias-targets-0 .Select-value-label")).to_have_count(25)
    research_title = paired_title.inner_text()
    expect(page.locator("#effect-results table tbody tr")).to_have_count(2 * len(MATRIX_MODELS))
    for chart in page.locator("#effect-results .dash-graph").all():
        expect(chart.locator(".ytick")).to_have_count(len(MATRIX_MODELS))
    choose(page, "target-preset", "Perceived religiousness")
    choose(page, "effect-left", "christian")
    choose(page, "effect-right", "muslim")
    expect(page.locator("#effect-results")).to_contain_text("christian versus muslim")
    expect(page.locator("#effect-results table tbody tr")).to_have_count(len(MATRIX_MODELS))

    methods = page.locator("#bias-view")
    check_religion_targets(page, methods)

    methods.get_by_text("Word valence", exact=True).click()
    valence = page.locator("#valence-results .dash-graph")
    check_chart(valence, ".scatterlayer .point", 320)
    expect(valence.locator(".xtitle")).to_have_text("Prime")
    expect(valence.locator(".xaxislayer-above .xtick")).to_contain_text(["christian", "muslim"])
    expect(valence.locator(".xtick text").last).to_be_visible()
    expect(valence.locator(".ytitle")).to_have_text("Valence slope")
    expect(valence.locator(".errorbars .yerror").first).to_be_visible()
    expect(page.locator("#valence-results table").first.locator("tbody tr")).to_have_count(len(MATRIX_MODELS))
    expect(valence.locator(".scatterlayer .trace")).to_have_count(len(MATRIX_MODELS))
    check_network_inclusion(page, methods)
    choose(page, "social-domain", "Political party (US)")
    methods.get_by_text("Emotion word sets", exact=True).click()
    emotion = page.locator("#emotion-results .dash-graph")
    check_chart(emotion, ".scatterlayer .point", 320)
    expect(emotion.locator(".xtitle")).to_have_text("Emotion word set")
    expect(emotion.locator(".xaxislayer-above .xtick")).to_contain_text(["anger", "trust"])
    expect(emotion.locator(".xtick text").last).to_be_visible()
    expect(emotion.locator(".ytitle")).to_have_text("Signed-rank effect (z/√N)")
    expect(page.locator("#emotion-results table tbody tr")).to_have_count(8 * len(MATRIX_MODELS))
    expect(emotion.locator(".scatterlayer .trace")).to_have_count(len(MATRIX_MODELS))
    check_touch_scroll(page, page.locator("#emotion-results"))
    check_width(page)

    methods.get_by_text("Association paths", exact=True).click()
    expect(page.locator("#model-inclusion-picker")).to_be_hidden()
    expect(page.locator("#path-start")).to_contain_text("democrat")
    expect(page.locator("#path-b")).to_contain_text("GPT-5 mini")
    check_chart(page.locator("#path-results .dash-graph").first, ".scatterlayer .point", 185)
    check_width(page)

    check_custom_domain(page, methods)
    check_primes_only_domain(page, methods)
    check_research_template(page, methods, research_title)
    check_model_details(page)
    if "gemma4_26b_uncensored_fp8" in MATRIX_MODELS:
        check_variant_network(page)


def check_variant_network(page):
    key = "gemma4_26b_uncensored_fp8"
    variant = GRAPH_MODELS[key]
    page.get_by_role("link", name="Explore", exact=True).click()
    search_model(page, "ex-model-a", variant)
    expect(page.locator(".series-legend")).to_contain_text(variant)
    page.get_by_text("2D graph", exact=True).click()
    check_chart(page.locator(".explorer-network .dash-graph").first, ".scatterlayer .point", 295)
    page.get_by_role("link", name="Lexical decision").click()
    search_model(page, "ldt-a", variant)
    expect(page.locator("#ldt-results .selected-row").filter(has_text=variant)).to_have_count(1)
    long_keys = ["nemotron3_nano30b_transformers", "nemotron3_nano30b_vllm"]
    if all(m in MATRIX_MODELS for m in long_keys):
        labels = [MATRIX_MODELS[m] for m in long_keys]
        for control, label in zip(("ldt-a", "ldt-b"), labels):
            search_model(page, control, label)
        result = page.locator("#ldt-results")
        expect(result.locator(".selected-row")).to_have_count(2)
        for label in labels:
            expect(result.locator(".selected-row").filter(has_text=label)).to_have_count(1)
        host = result.locator(".dash-graph")
        check_chart(host, ".boxlayer .point", 380)
        expect(host.locator(".xtick")).to_have_count(2)
        assert host.evaluate("""el => {
            const rect = el.getBoundingClientRect();
            const ticks = [...el.querySelectorAll('.xtick text')].map(t => t.getBoundingClientRect());
            return ticks.every(t => t.left >= rect.left - 1 && t.right <= rect.right + 1 &&
                                     t.bottom <= rect.bottom + 1) && ticks[0].right <= ticks[1].left;
        }""")
    page.get_by_role("link", name="Implicit bias").click()
    search_model(page, "social-a", variant)
    page.locator("#bias-view").get_by_text("Heatmaps", exact=True).click()
    expect(page.locator("#heat-results h3").first).to_have_text(variant)
    check_chart(page.locator("#heat-results .dash-graph").first, ".heatmaplayer image", 320)
    check_width(page)


def search_model(page, control_id, label):
    dropdown = page.locator("#" + control_id)
    dropdown.locator("input").fill(label)
    dropdown.locator(".Select-menu-outer").get_by_text(label, exact=True).click()


def check_model_details(page):
    page.get_by_role("link", name="Model details", exact=True).click()
    expect(page.get_by_role("heading", name="Model and collection details")).to_be_visible()
    served = page.locator("#model-gpt5_mini")
    expect(served.locator("summary")).to_contain_text("gpt-5-mini-2025-08-07")
    served.locator("summary").press("Enter")
    expect(served.locator("dt", has_text="Version or revision")).to_be_visible()
    expect(served.locator("dd a").first).to_have_attribute("href", re.compile("^https://"))
    latest = "gemma4_26b_uncensored_fp8"
    group = "Analysis datasets" if latest in GRAPH_MODELS else "Additional collections"
    pending = page.locator(".model-catalog section").filter(
        has=page.get_by_role("heading", name=group)
    ).locator("#model-gemma4_26b_uncensored_fp8")
    identifier = pending.locator("summary code")
    expect(identifier).to_have_text("orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8")
    pending.locator("summary").click()
    expect(pending.locator("dd a").first).to_be_visible()
    assert identifier.evaluate("el => el.getBoundingClientRect().right <= innerWidth")
    check_width(page)


def check_custom_domain(page, methods):
    choose(page, "social-domain", "Add a custom domain…")
    page.locator("#custom-name").fill("Age associations")
    page.locator("#custom-primes").fill("young, old")
    rows = page.locator("#custom-list-rows")
    page.locator("#custom-add-list").click()
    rows.get_by_label("Word-list name").fill("Experience")
    rows.get_by_label("Target words").fill("wise, capable")
    rows.get_by_label("Expected prime (optional)").fill("young")
    page.locator("#custom-add-list").click()
    expect(rows.get_by_label("Word-list name")).to_have_count(2)
    rows.get_by_label("Word-list name").last.fill("Energy")
    rows.get_by_label("Target words").last.fill("active, quick")
    rows.get_by_label("Expected prime (optional)").last.fill("old")
    page.locator("#custom-save").click()
    expect(page.locator("#social-domain .Select-value-label")).to_have_text("Age associations")
    methods.get_by_text("Priming effects", exact=True).click()
    expect(page.locator("#effect-results .dash-graph")).to_have_count(2)
    expect(page.locator("#effect-results table tbody tr")).to_have_count(2 * len(MATRIX_MODELS))
    expect(page.locator("#effect-results")).to_contain_text("2/2 targets")
    choose(page, "target-preset", "Energy")
    expect(page.locator("#effect-results")).to_contain_text("2/2 targets")
    methods.get_by_text("Heatmaps", exact=True).click()
    expect(page.locator("#heat-results .dash-graph")).to_have_count(2)
    expect(page.locator("#heat-results .dash-graph").first.locator(".ytick text")).to_have_text(["active", "quick"])
    methods.get_by_text("Word valence", exact=True).click()
    expect(page.locator("#valence-results .dash-graph .xtick text")).to_have_text(["young", "old"])
    expect(page.locator("#valence-results table").first.locator("tbody tr")).to_have_count(len(MATRIX_MODELS))
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(page.locator("#emotion-results table tbody tr")).to_have_count(8 * len(MATRIX_MODELS))
    expect(page.locator("#emotion-results .scatterlayer .point")).to_have_count(8 * len(MATRIX_MODELS))
    methods.get_by_text("Association paths", exact=True).click()
    choose(page, "path-list", "Energy")
    expect(page.locator("#path-results h3").first).to_have_text("young → active")
    page.reload()
    expect(page.locator("#social-domain .Select-value-label")).to_have_text("Age associations")
    methods.get_by_text("Priming effects", exact=True).click()
    page.locator("#custom-summary").click()
    page.locator("#custom-name").fill("Age and experience")
    page.locator("#custom-save").click()
    expect(page.locator("#effect-results h2")).to_have_text("Age and experience priming effects")
    expect(page.locator("#custom-summary")).to_have_text("Edit Age and experience")
    if not page.locator("#custom-manager").evaluate("element => element.open"):
        page.locator("#custom-summary").click()
    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Delete domain").click()
    expect(page.locator("#social-domain .Select-value-label")).to_have_text("Gender")
    check_width(page)


def search_word(page, control, word):
    dropdown = page.locator("#" + control)
    dropdown.locator("input").fill(word)
    dropdown.locator(".Select-menu-outer").get_by_text(word, exact=True).click()


def check_primes_only_domain(page, methods):
    methods.get_by_text("Word valence", exact=True).click()
    valence_results = page.locator("#valence-results")
    expect(valence_results).to_be_attached()
    choose(page, "social-domain", "Add a custom domain…")
    # A pending chart response still needs its Dash output while the editor is open.
    expect(valence_results).to_be_hidden()
    expect(valence_results).to_be_attached()
    page.locator("#custom-name").fill("Occupations")
    page.locator("#custom-primes").fill("teacher, nurse")
    rows = page.locator("#custom-list-rows")
    expect(rows.get_by_label("Target words")).to_have_count(0)
    page.locator("#custom-save").click()
    domain_label = page.locator("#social-domain .Select-value-label")
    expect(domain_label).to_have_text("Occupations")
    valence = valence_results.locator(".dash-graph")
    expect(valence.locator(".xtick text")).to_have_text(["teacher", "nurse"])
    expect(valence.locator(".yerror").first).to_be_visible()

    page.reload()
    expect(domain_label).to_have_text("Occupations")
    methods.get_by_text("Emotion word sets", exact=True).click()
    emotion = page.locator("#emotion-results")
    expect(emotion).to_contain_text("Positive effects mean words in a set received stronger activation from teacher")
    expect(emotion.locator(".scatterlayer .trace").first.locator(".point")).to_have_count(8)

    for method, results in (("Priming effects", "effect"), ("Heatmaps", "heat")):
        methods.get_by_text(method, exact=True).click()
        expect(page.locator(f"#{results}-results [role=status]")).to_contain_text("target")
        expect(page.locator("#target-preset")).to_contain_text("Enter two target sets")
    search_word(page, "bias-targets-0", "wise")
    search_word(page, "bias-targets-1", "care")
    expect(page.locator("#heat-results .dash-graph").first.locator(".ytick text")).to_have_text(["wise"])
    methods.get_by_text("Priming effects", exact=True).click()
    expect(page.locator("#effect-results")).to_contain_text("1/1 targets")

    methods.get_by_text("Association paths", exact=True).click()
    expect(page.locator("#path-results")).to_contain_text("target")
    expect(page.locator("#path-list")).to_have_count(0)
    expect(page.locator("#path-start .Select-value-label")).to_have_text(["teacher", "nurse"])
    expect(page.locator("#path-end .Select-value-label")).to_have_count(0)
    search_word(page, "path-end", "care")
    expect(page.locator("#path-results h3").first).to_have_text("teacher → care")
    expect(page.locator("#path-results .path-pair").first.locator(".scatterlayer .point").first).to_be_visible()

    page.locator("#custom-summary").click()
    expect(rows.get_by_label("Target words")).to_have_count(0)
    page.locator("#custom-add-list").click()
    rows.get_by_label("Word-list name").fill("Training")
    rows.get_by_label("Target words").fill("skill, knowledge")
    page.locator("#custom-save").click()
    expect(page.locator("#path-list .Select-value-label")).to_have_text("Training")
    expect(page.locator("#path-results h3").first).to_have_text("teacher → skill")
    check_width(page)


def check_research_template(page, methods, title):
    methods.get_by_text("Priming effects", exact=True).click()
    disclosure = page.locator("#custom-summary")
    name = page.locator("#custom-name")
    disclosure.click()
    name.fill("Unsaved occupation draft")
    disclosure.click()
    expect(name).to_be_hidden()
    disclosure.click()
    expect(name).to_be_visible()
    expect(name).to_have_value("Unsaved occupation draft")
    # Switching directly from an open edit must direct the next input to the new draft.
    choose(page, "social-domain", "Add a custom domain…")
    name.fill("Replication")
    choose(page, "custom-template", re.compile(re.escape(title) + r" · 25 \+ 25 targets"))
    lists = page.locator("#custom-list-rows")
    expect(lists.get_by_label("Target words")).to_have_count(2)
    for words in lists.get_by_label("Target words").all():
        assert len(words.input_value().split(",")) == 25
    page.locator("#custom-save").click()
    expect(page.locator("#social-domain")).to_contain_text("Replication")
    expect(page.locator("#effect-results table tbody tr")).to_have_count(2 * len(MATRIX_MODELS))
    expect(page.locator("#target-editor a").first).to_be_visible()
    evidence = page.locator("#target-editor").get_by_role("link", name="Target rankings and nomination counts (CSV)")
    response = page.request.get(evidence.evaluate("link => link.href"))
    assert response.ok and response.text().startswith("domain,prime_1,prime_2,")
    assert "religion,christian,muslim," in response.text()
    page.reload()
    expect(page.locator("#social-domain")).to_contain_text("Replication")
    expect(page.locator("#bias-targets-1 .Select-value-label")).to_have_count(25)
    expect(page.locator("#effect-results .dash-graph")).to_have_count(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Local or deployed website URL")
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--channel", help="Installed Chromium channel, e.g. chrome")
    args = parser.parse_args()
    expect.set_options(timeout=30_000)
    with sync_playwright() as playwright:
        engine = getattr(playwright, args.browser)
        browser = engine.launch(channel=args.channel) if args.channel else engine.launch()
        for width in (1360, 375):
            page = browser.new_page(viewport={"width": width, "height": 850},
                                    is_mobile=width == 375, has_touch=width == 375)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            check_page(page, args.url)
            assert not errors, errors
            print(f"Passed {args.browser} at {width}px", flush=True)
            page.close()
        browser.close()
