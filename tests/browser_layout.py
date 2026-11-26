"""Run rendered chart and dropdown regressions against a running website."""

import argparse
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))
from wordviz.bias_norms import EMOTIONS
from wordviz.models import GRAPH_MODELS, MATRIX_MODELS
from wordviz.social_domains import DOMAIN_PRIMES

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


def check_matrix(results, categories):
    graph = results.locator(".dash-graph")
    check_chart(graph, ".heatmaplayer image", 320)
    expect(results.locator(".norm-matrix-columns > span")).to_have_text(list(categories))
    expect(results.locator(".norm-matrix-row")).to_have_count(len(MATRIX_MODELS))
    data = graph.locator(".js-plotly-plot").evaluate("""el => {
        const heatmap = el.data[0];
        return {type: heatmap.type, x: heatmap.x, y: heatmap.y,
                shape: heatmap.z.map(row => row.length), hasNegative: heatmap.z.flat().some(v => v < 0),
                hasPositive: heatmap.z.flat().some(v => v > 0)};
    }""")
    assert data["type"] == "heatmap" and data["x"] == list(categories), data
    assert data["y"] == list(MATRIX_MODELS.values()), data["y"]
    assert data["shape"] == [len(categories)] * len(MATRIX_MODELS), data["shape"]
    assert data["hasNegative"] and data["hasPositive"], data
    expect(graph.locator(".heatmaplayer text").first).to_be_visible()
    # A sticky label has to line up with the plotted row for every model.
    assert results.evaluate("""el => {
        const plot = el.querySelector('.js-plotly-plot'), {yaxis} = plot._fullLayout;
        const origin = plot.getBoundingClientRect().top + yaxis._offset;
        return [...el.querySelectorAll('.norm-matrix-row')].every((row, i) => {
            const rect = row.getBoundingClientRect();
            return Math.abs(rect.top + rect.height / 2 - origin - yaxis.l2p(i)) < 2;
        });
    }""")


def check_norm_tooltip(page, results, method, scatter=False):
    plot = results.locator(".js-plotly-plot")
    # Exercise a real point or cell after scrolling the chart. Prefer a long
    # build name in the matrix and an unobstructed highlighted marker in scatter.
    target = plot.evaluate("""(el, scatter) => {
        const {xaxis, yaxis} = el._fullLayout;
        const trace = scatter ? el.data.at(-1) : el.data[0];
        const col = trace.x.length - 1;
        const row = scatter ? null : trace.y.reduce((best, name, i) =>
            trace.z[i][col] != null && name.length > (trace.y[best]?.length ?? 0) ? i : best, -1);
        const x = xaxis._offset + xaxis.l2p(scatter ? trace.x[col] : col);
        const y = yaxis._offset + yaxis.l2p(scatter ? trace.y[col] : row);
        const scroller = el.closest('[role=region]');
        scroller.scrollIntoView({block:'center'});
        scroller.scrollLeft = Math.max(0, x - scroller.clientWidth + 22);
        scroller.scrollTop = Math.max(0, y - scroller.clientHeight + 35);
        const rect = el.getBoundingClientRect();
        return {x:x + rect.left, y:y + rect.top, model:scatter ? trace.name : trace.y[row],
                category:scatter ? trace.customdata[col][0] : trace.x[col]};
    }""", scatter)
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    touch = page.evaluate("navigator.maxTouchPoints > 0")
    if touch:
        page.touchscreen.tap(target["x"], target["y"])
    else:
        page.mouse.move(target["x"], target["y"])
    card = page.get_by_role("tooltip")
    expect(card.locator(".norm-tooltip-network")).to_have_text(target["model"])
    expect(card.locator(".norm-tooltip-category")).to_contain_text(re.compile(re.escape(target["category"]), re.I))
    expect(card.locator(".norm-tooltip-summary")).not_to_be_empty()
    if method == "emotion":
        for field in ("emotion-left", "emotion-right"):
            prime = page.locator(f"#{field} .Select-value-label").inner_text()
            expect(card).to_contain_text(prime)
    metric = card.locator(".norm-tooltip-metric").inner_text()
    estimate = metric.split(" · ")[0]
    assert re.search(r"\d", estimate) and not re.search(r"\d\.\d{3,}", estimate), metric
    expect(plot.locator(".hoverlayer .hovertext")).to_have_count(0)
    assert card.evaluate("""el => {
        const r = el.getBoundingClientRect(), color = getComputedStyle(el).backgroundColor;
        return r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight &&
            color.startsWith('rgba(') && Number(color.slice(5, -1).split(',')[3]) < 1;
    }""")
    # Panning the table should clear the details of the cell that just moved.
    scrolled = not scatter and plot.evaluate("""el => {
        const region = el.closest('[role=region]'), before = region.scrollTop;
        region.scrollTop += before > 5 ? -5 : 5;
        return before !== region.scrollTop;
    }""")
    if not scrolled:
        if touch:
            page.touchscreen.tap(2, 2)
        else:
            page.keyboard.press("Escape")
    expect(card).to_be_hidden()


def check_norm_scatter(page, method):
    results = page.locator(f"#{method}-results")
    switch = page.locator(f"#{method}-chart-view")
    expect(switch.get_by_role("radio", name="Table", exact=True)).to_be_checked()
    graph = results.locator(".dash-graph")
    plot = graph.locator(".js-plotly-plot")
    expect(graph.locator(".heatmaplayer image")).to_be_visible()
    matrix = plot.evaluate("el => { const {x, y, z} = el.data[0]; return {x, y, z}; }")
    check_norm_tooltip(page, results, method)
    expected = {(model, category): value for model, row in zip(matrix["y"], matrix["z"])
                for category, value in zip(matrix["x"], row) if value is not None}
    stats = results.locator("table").all_text_contents()
    switch.get_by_text("Scatter plot", exact=True).click()
    check_chart(graph, ".scatterlayer .point", 320)
    expect(graph.locator(".xtick text")).to_have_text(matrix["x"])
    traces = plot.evaluate("""el => el.data.map(trace => ({
        model: trace.name, categories: trace.customdata.map(row => row[0]), values: trace.y,
        hasIntervals: Boolean(trace.error_y?.array?.length)
    }))""")
    points = {(trace["model"], category): value for trace in traces
              for category, value in zip(trace["categories"], trace["values"])}
    assert points == expected
    assert all(trace["hasIntervals"] == (method == "valence") for trace in traces)
    check_norm_tooltip(page, results, method, scatter=True)
    assert results.locator("table").all_text_contents() == stats
    check_width(page)


def check_touch_scroll(page, results):
    if page.viewport_size["width"] > 400 or page.context.browser.browser_type.name != "chromium":
        return
    region = results.get_by_role("region", name=re.compile("chart$"))
    if not region.evaluate("el => el.scrollWidth > el.clientWidth"):
        return
    region.scroll_into_view_if_needed()
    first_label = results.locator(".norm-matrix-row").first
    label_x = first_label.bounding_box()["x"]
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
        assert abs(first_label.bounding_box()["x"] - label_x) <= 1
        columns = results.locator(".norm-matrix-columns")
        header_y = columns.bounding_box()["y"]
        region.evaluate("el => { el.scrollTop = el.scrollHeight; el.scrollLeft = el.scrollWidth; }")
        assert abs(columns.bounding_box()["y"] - header_y) <= 1
        assert abs(first_label.bounding_box()["x"] - label_x) <= 1
        corner = results.locator(".norm-matrix-corner").bounding_box()
        assert abs(corner["y"] - header_y) <= 1 and abs(corner["x"] - label_x) <= 1
    finally:
        touch.detach()


def check_network_inclusion(page, methods):
    count = len(MATRIX_MODELS)
    family_count = sum(key.lower().startswith("gemma") for key in MATRIX_MODELS)
    picker = page.locator("#model-inclusion-picker")
    valence = page.locator("#valence-results")
    emotion = page.locator("#emotion-results")
    matrix_rows = ".norm-matrix-row"
    legacy_gemma = MATRIX_MODELS["Gemma"]
    all_models = picker.get_by_role("checkbox", name="All networks", exact=True)
    gemma = picker.get_by_role("checkbox", name="Gemma family", exact=True)
    picker.locator("summary").click()
    expect(all_models).to_be_checked()
    all_models.click()
    expect(valence.get_by_role("status")).to_be_visible()
    expect(valence.locator(".dash-graph")).to_have_count(0)
    gemma.click()
    expect(gemma).to_be_checked()
    expect(valence.locator(matrix_rows)).to_have_count(family_count)
    expect(valence.get_by_label("Joint valence tests").locator(".selected-row")).to_have_count(0)

    # Dropdowns highlight only included models. An excluded primary stays excluded.
    expect(valence.locator(".norm-matrix-row[data-highlight=primary]")).to_have_count(0)
    search_model(page, "social-a", legacy_gemma)
    expect(valence.get_by_label("Joint valence tests").locator(".selected-row th")).to_have_text([legacy_gemma])
    expect(valence.locator(".norm-matrix-row[data-highlight=primary]")).to_contain_text(legacy_gemma)
    check_norm_scatter(page, "valence")
    expect(valence.locator(".series-legend")).to_contain_text(legacy_gemma)
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(page.locator("#emotion-chart-view").get_by_role("radio", name="Table", exact=True)).to_be_checked()
    expect(emotion.locator(matrix_rows)).to_have_count(family_count)
    methods.get_by_text("Word valence", exact=True).click()
    expect(valence.locator(".scatterlayer .point").first).to_be_visible()
    page.locator("#valence-chart-view").get_by_text("Table", exact=True).click()
    expect(valence.locator(matrix_rows)).to_have_count(family_count)
    if family_count > 1:
        picker.get_by_role("checkbox", name=legacy_gemma, exact=True).click()
        expect(gemma).to_have_attribute("aria-checked", "mixed")
        expect(valence.locator(matrix_rows)).to_have_count(family_count - 1)
        expect(valence.get_by_label("Joint valence tests").locator(".selected-row")).to_have_count(0)
        family_count -= 1
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(emotion.locator(matrix_rows)).to_have_count(family_count)
    expect(emotion.get_by_label("Emotion word set tests").locator("tbody tr")).to_have_count(8 * family_count)

    page.reload()
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(emotion.locator(matrix_rows)).to_have_count(family_count)
    methods.get_by_text("Priming effects", exact=True).click()
    expect(page.locator("#effect-results .dash-graph").first.locator(".ytick")).to_have_count(family_count)
    if not picker.evaluate("el => el.open"):
        picker.locator("summary").click()
    all_models.click()
    expect(all_models).to_be_checked()
    picker.locator("summary").click()
    search_model(page, "social-a", "Humans")
    search_model(page, "social-b", MATRIX_MODELS["Gpt5_mini"])
    methods.get_by_text("Word valence", exact=True).click()
    expect(valence.locator(matrix_rows)).to_have_count(count)
    check_width(page)


def choose(page, control, option):
    dropdown = page.locator("#" + control)
    dropdown.locator(".Select-control").click()
    dropdown.locator(".Select-menu-outer").get_by_text(option, exact=True).click()


def check_pair_directions(page, titles, primes):
    headings = page.locator("#effect-results h3")
    expect(headings).to_have_text(titles)
    for i, heading in enumerate(headings.all()):
        panel = heading.locator("..")
        expect(panel).to_contain_text(f"Positive bars mean stronger activation from “{primes[i]}”.")
        expect(panel).to_contain_text(f"Negative bars mean stronger activation from “{primes[1 - i]}”.")
        assert heading.evaluate("el => el.scrollWidth <= el.clientWidth + 1")


def check_paired_religion(page):
    titles = ["Target words associated with Christians", "Target words associated with Muslims"]
    primes = ["christian", "muslim"]
    check_pair_directions(page, titles, primes)
    expect(page.locator("#bias-targets-0-label")).to_have_text("Target words for Christians (expected prime “christian”)")
    expect(page.locator("#bias-targets-1-label")).to_have_text("Target words for Muslims (expected prime “muslim”)")
    panel = page.locator("#effect-results h3").first.locator("..")
    matched = panel.locator("details").first
    matched.locator("summary").click()
    expect(matched).to_contain_text(re.compile(r"\bfaithful\b"))
    expect(matched).not_to_contain_text(re.compile(r"\bjudgmental\b"))
    page.locator("#bias-targets-0 .Select-value").filter(has_text="faithful").locator(".Select-value-icon").click()
    check_pair_directions(page, ["Selected target words for Christians", titles[1]], primes)
    page.locator("#target-reset").click()
    check_pair_directions(page, titles, primes)
    choose(page, "target-preset", re.compile("Christians vs. Atheists"))
    check_pair_directions(page, [titles[0], "Target words associated with Atheists"], ["christian", "atheist"])
    check_width(page)


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
    menu = page.locator("#ex-model-b .Select-menu-outer")
    for query, key in [("mistral-7b-instruct-v0.1.Q4_K_M.gguf", "mistral"),
                       ("claude-3-5-haiku-latest", "haiku"), ("claude-3-5-haiku-20241022", "haiku"),
                       ("gpt-5-mini-2025-08-07", "gpt5_mini")]:
        compare.fill(query)
        option = menu.locator(".VirtualizedSelectOption")
        expect(option).to_have_count(1)
        expect(option).to_contain_text(GRAPH_MODELS[key])
    compare.fill("pinned")
    expect(menu.locator(".VirtualizedSelectOption").first).to_contain_text(GRAPH_MODELS["gpt5_mini"])
    expect(menu.locator(".model-pin")).to_have_count(5)
    compare.fill("OpenAI")
    expect(menu.locator(".VirtualizedSelectOption")).to_have_count(1)
    compare.press("Enter")
    expect(page.locator("#ex-model-b .Select-value-label")).to_contain_text(GRAPH_MODELS["gpt5_mini"])
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
    choose(page, "social-b", MATRIX_MODELS["Gpt5_mini"])
    choose(page, "social-domain", "Religion")
    paired_title = page.locator("#target-preset .Select-value-label")
    expect(page.locator("#bias-targets-1 .Select-value-label")).to_have_count(25)
    expect(page.locator("#bias-targets-0 .Select-value-label")).to_have_count(25)
    research_title = paired_title.inner_text()
    expect(page.locator("#effect-results table tbody tr")).to_have_count(2 * len(MATRIX_MODELS))
    for chart in page.locator("#effect-results .dash-graph").all():
        expect(chart.locator(".ytick")).to_have_count(len(MATRIX_MODELS))
    check_paired_religion(page)
    choose(page, "target-preset", "Perceived religiousness")
    choose(page, "effect-left", "christian")
    choose(page, "effect-right", "muslim")
    expect(page.locator("#effect-results")).to_contain_text("christian versus muslim")
    expect(page.locator("#effect-results table tbody tr")).to_have_count(len(MATRIX_MODELS))

    methods = page.locator("#bias-view")
    check_religion_targets(page, methods)

    methods.get_by_text("Word valence", exact=True).click()
    valence = page.locator("#valence-results")
    check_matrix(valence, DOMAIN_PRIMES["religion"])
    expect(valence.get_by_label("Joint valence tests").locator("tbody tr")).to_have_count(len(MATRIX_MODELS))
    expect(valence.get_by_label("Valence coefficients and intervals").locator("thead")).to_contain_text("95% lower")
    check_network_inclusion(page, methods)
    choose(page, "social-domain", "Political party (US)")
    methods.get_by_text("Emotion word sets", exact=True).click()
    emotion = page.locator("#emotion-results")
    check_matrix(emotion, EMOTIONS)
    expect(emotion.locator(".norm-matrix-row[data-highlight=primary]")).to_contain_text("Humans")
    expect(emotion.locator(".norm-matrix-row[data-highlight=secondary]")).to_contain_text(MATRIX_MODELS["Gpt5_mini"])
    expect(emotion.get_by_label("Emotion word set tests").locator("tbody tr")).to_have_count(8 * len(MATRIX_MODELS))
    check_norm_scatter(page, "emotion")
    for prime_id, prime in (("emotion-left", "democrat"), ("emotion-right", "republican")):
        expect(page.locator(f"#{prime_id} .Select-value-label")).to_have_text(prime)
    expect(emotion.locator(".series-legend")).to_contain_text(MATRIX_MODELS["Gpt5_mini"])
    page.locator("#emotion-chart-view").get_by_text("Table", exact=True).click()
    expect(emotion.locator(".norm-matrix-row")).to_have_count(len(MATRIX_MODELS))
    with page.expect_download() as downloaded:
        emotion.get_by_role("button", name="Download PNG").click()
    png = downloaded.value
    assert png.suggested_filename == "emotion-word-sets.png"
    with png.path().open("rb") as image:
        header = image.read(24)
    assert header.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", header[16:24])
    assert width >= 860 and height > 46 * len(MATRIX_MODELS)
    check_touch_scroll(page, emotion)
    check_width(page)

    methods.get_by_text("Association paths", exact=True).click()
    expect(page.locator("#model-inclusion-picker")).to_be_hidden()
    expect(page.locator("#path-start")).to_contain_text("democrat")
    expect(page.locator("#path-b")).to_contain_text(GRAPH_MODELS["gpt5_mini"])
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
    for key in ("gpt5_mini", "haiku", "mistral", "gemma"):
        expect(page.locator(f"#model-{key} summary .model-name")).to_have_text(GRAPH_MODELS[key])
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
    experience = "IntergenerationalLearningAndProfessionalExperience"
    rows.get_by_label("Word-list name").fill(experience)
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
    check_pair_directions(page, [f"Target words in “{experience}”", "Target words in “Energy”"], ["young", "old"])
    choose(page, "target-preset", "Energy")
    expect(page.locator("#effect-results")).to_contain_text("2/2 targets")
    methods.get_by_text("Heatmaps", exact=True).click()
    expect(page.locator("#heat-results .dash-graph")).to_have_count(2)
    expect(page.locator("#heat-results .dash-graph").first.locator(".ytick text")).to_have_text(["active", "quick"])
    methods.get_by_text("Word valence", exact=True).click()
    expect(page.locator("#valence-results .dash-graph .xtick text")).to_have_text(["young", "old"])
    expect(page.get_by_label("Joint valence tests").locator("tbody tr")).to_have_count(len(MATRIX_MODELS))
    methods.get_by_text("Emotion word sets", exact=True).click()
    expect(page.get_by_label("Emotion word set tests").locator("tbody tr")).to_have_count(8 * len(MATRIX_MODELS))
    check_matrix(page.locator("#emotion-results"), EMOTIONS)
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
    expect(valence.locator(".heatmaplayer image")).to_be_visible()
    expect(valence_results.get_by_label("Valence coefficients and intervals").locator("tbody tr")).to_have_count(2 * len(MATRIX_MODELS))

    page.reload()
    expect(domain_label).to_have_text("Occupations")
    methods.get_by_text("Emotion word sets", exact=True).click()
    emotion = page.locator("#emotion-results")
    expect(emotion).to_contain_text("Positive effects mean words in a set received stronger activation from teacher")
    check_matrix(emotion, EMOTIONS)

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
