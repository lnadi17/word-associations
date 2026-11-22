// Dash puts each dropdown's ID on a wrapper. Connect its visible label to
// the focusable combobox when that control mounts or is replaced.
(() => {
  function labelCombobox(control) {
    const dropdown = control.closest('.dash-dropdown');
    const label = dropdown?.previousElementSibling;
    if (label?.tagName !== 'LABEL') return;

    label.id = dropdown.id + '-label';
    control.id = dropdown.id + '-control';
    control.setAttribute('aria-labelledby', label.id);
    label.htmlFor = control.id;
  }

  function labelControlsIn(node) {
    if (!(node instanceof HTMLElement)) return;
    if (node.matches('[role="combobox"]')) labelCombobox(node);
    else if (node.childElementCount) node.querySelectorAll('[role="combobox"]').forEach(labelCombobox);
  }

  labelControlsIn(document.documentElement);
  new MutationObserver(records => {
    for (const record of records) record.addedNodes.forEach(labelControlsIn);
  }).observe(document.documentElement, { childList: true, subtree: true });

  // React Select renders non-searchable controls as divs. Native labels
  // already focus searchable inputs and textareas.
  document.addEventListener('click', event => {
    const label = event.target.closest?.('label[for]');
    if (!label) return;
    const control = document.getElementById(label.htmlFor);
    if (control?.matches('div[role="combobox"]')) control.focus();
  });
})();
