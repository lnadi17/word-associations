"""Run rendered chart and dropdown regressions against a running website."""

import argparse
import re

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
    compare.fill("GPT-5 mini")
    page.locator("#ex-model-b .Select-menu-outer").get_by_text("GPT-5 mini", exact=True).click()
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
    choose(page, "effect-left", "christian")
    choose(page, "effect-right", "muslim")
    expect(page.locator("#effect-results")).to_contain_text("christian versus muslim")
    expect(page.locator("#effect-results table tbody tr")).to_have_count(2)

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
    expect(page.locator("#valence-results table").first.locator("tbody tr")).to_have_count(2)
    choose(page, "social-domain", "Political party (US)")
    methods.get_by_text("Emotion word sets", exact=True).click()
    emotion = page.locator("#emotion-results .dash-graph")
    check_chart(emotion, ".scatterlayer .point", 320)
    expect(emotion.locator(".xtitle")).to_have_text("Emotion word set")
    expect(emotion.locator(".xaxislayer-above .xtick")).to_contain_text(["anger", "trust"])
    expect(emotion.locator(".xtick text").last).to_be_visible()
    expect(emotion.locator(".ytitle")).to_have_text("Signed-rank effect (z/√N)")
    expect(page.locator("#emotion-results table tbody tr")).to_have_count(16)
    check_width(page)

    methods.get_by_text("Association paths", exact=True).click()
    expect(page.locator("#path-start")).to_contain_text("democrat")
    expect(page.locator("#path-b")).to_contain_text("GPT-5 mini")
    check_chart(page.locator("#path-results .dash-graph").first, ".scatterlayer .point", 185)
    check_width(page)


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
