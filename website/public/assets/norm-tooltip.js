// Keep norm chart details in the viewport, including when a matrix is scrolled.
(() => {
  const graphs = new WeakSet();
  let card;
  let activePlot;
  let activeTrace;
  let plotBounds;
  let pinned = false;
  let pointer;

  function hide() {
    if (activePlot) {
      activePlot.removeAttribute('aria-describedby');
      card.hidden = true;
      activePlot = null;
      activeTrace = null;
      pinned = false;
      unmountObserver.disconnect();
    }
  }

  const unmountObserver = new MutationObserver(() => {
    if (activePlot && !activePlot.isConnected) hide();
  });

  function show(plot, data, pin) {
    const point = data.points[0];
    const details = point?.customdata;
    if (!Array.isArray(details) || !details[1] || pointer?.plot !== plot || !plot.isConnected) {
      hide();
      return;
    }
    if (!card) {
      card = document.createElement('div');
      card.id = 'norm-tooltip';
      card.setAttribute('role', 'tooltip');
      for (const part of ['network', 'category', 'summary', 'metric', 'detail']) {
        const line = document.createElement(part === 'network' ? 'strong' : 'div');
        line.className = 'norm-tooltip-' + part;
        card.append(line);
      }
      document.body.append(card);
    }
    if (activePlot !== plot) {
      hide();
      plot.setAttribute('aria-describedby', card.id);
      activePlot = plot;
      unmountObserver.observe(document.body, {subtree: true, childList: true});
    }
    activeTrace = point.data;
    plotBounds = plot.getBoundingClientRect();
    pinned = pin && pointer.type === 'touch';
    Array.from(card.children).forEach((line, i) => {
      line.textContent = details[i + 1] ?? '';
    });
    card.hidden = false;
    // Prefer a corner away from the pointer. Flip and clamp at viewport edges.
    const gutter = 8, gap = 12;
    const {width, height} = card.getBoundingClientRect();
    const right = document.documentElement.clientWidth - width - gutter;
    const bottom = window.innerHeight - height - gutter;
    const x = pointer.x + gap <= right ? pointer.x + gap : pointer.x - width - gap;
    const y = pointer.y + gap <= bottom ? pointer.y + gap : pointer.y - height - gap;
    card.style.left = Math.max(gutter, Math.min(x, right)) + 'px';
    card.style.top = Math.max(gutter, Math.min(y, bottom)) + 'px';
  }

  function track(event) {
    const plot = event.target.closest?.('.norm-hover-chart .js-plotly-plot');
    if (!plot || typeof plot.on !== 'function') {
      pointer = null;
      hide();
      return;
    }
    if (event.type === 'pointerdown') hide();
    pointer = {plot, x: event.clientX, y: event.clientY, type: event.pointerType};
    if (graphs.has(plot)) return;
    graphs.add(plot);
    plot.on('plotly_hover', data => { if (!pinned) show(plot, data, false); });
    plot.on('plotly_click', data => show(plot, data, true));
    plot.on('plotly_unhover', () => { if (activePlot === plot && !pinned) hide(); });
    // A touch tap can emit unhover and afterplot as its synthetic mouse
    // events finish. Keep the tapped value until its trace actually changes.
    plot.on('plotly_afterplot', () => {
      if (activePlot === plot && !plot.data.includes(activeTrace)) hide();
    });
    plot.addEventListener('pointerleave', event => {
      if (event.pointerType === 'mouse' && activePlot === plot) hide();
    });
  }

  for (const type of ['pointerover', 'pointermove', 'pointerdown']) {
    document.addEventListener(type, track, {capture: true, passive: true});
  }
  window.addEventListener('scroll', () => {
    if (!activePlot) return;
    // A scroll notification can arrive after the pointer has already landed
    // on a cell at its new position. Dismiss only when its anchor has moved.
    const bounds = activePlot.getBoundingClientRect();
    if (bounds.left !== plotBounds.left || bounds.top !== plotBounds.top) hide();
  }, {capture: true, passive: true});
  window.addEventListener('resize', hide);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') hide(); });
})();
