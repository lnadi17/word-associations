"""Run rendered chart and dropdown regressions against a running website."""

import argparse

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


def check_page(page, url):
    page.goto(url)
    compare = page.get_by_role("combobox", name="Compare model", exact=True)
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
    compare = page.get_by_role("combobox", name="Compare with", exact=True)
    page.get_by_text("Compare with", exact=True).click()
    compare.fill("GPT-5 mini")
    page.locator("#bias-b .Select-menu-outer").get_by_text("GPT-5 mini", exact=True).click()
    heatmaps = page.locator("#bias-results .panel").nth(1).locator(".dash-graph")
    expect(heatmaps).to_have_count(2)
    effects = page.locator("#bias-results .panel").first.locator(".dash-graph")
    expect(effects).to_have_count(2)
    for effect in effects.all():
        check_chart(effect, ".barlayer .point path", 320)
    for heatmap in heatmaps.all():
        check_chart(heatmap, ".heatmaplayer image", 320)
    check_width(page)

    page.get_by_text("Association paths", exact=True).click()
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
