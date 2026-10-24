/* Presentation and navigation only. No analysis state is modified here. */
(function () {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const modes = $$('.mode');
  const activeMode = () => modes.find(mode => !mode.hidden);
  const labels = { 'mode-3d': '3D structures', 'mode-2d': '2D frames', 'mode-fe': 'Finite element plates' };
  const icons = {
    setup: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M8 3v18M12 8h5m-5 4h5m-5 4h3"/>',
    results: '<path d="M5 20V10m7 10V4m7 16v-7M2 20h20"/>',
    view: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9"/>'
  };
  const icon = key => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[key]}</svg>`;
  for (const mode of modes) {
    for (const [selector, title, glyph, meta] of [
      ['.editor', 'Model setup', 'setup', 'INPUTS'],
      ['.draw', mode.id === 'mode-fe' ? 'Field viewer' : 'Structural model', 'view', mode.id === 'mode-3d' ? '3D VIEW' : '2D VIEW'],
      ['.results', 'Analysis overview', 'results', 'RESULTS']
    ]) {
      const heading = document.createElement('div');
      heading.className = 'panel-heading';
      heading.innerHTML = `${icon(glyph)}<h2${selector === '.draw' ? ' class="viewport-label"' : ''}>${title}</h2><span class="panel-meta">${meta}</span>`;
      $(selector, mode).prepend(heading);
    }
    const foot = document.createElement('div');
    foot.className = 'viewport-footer';
    foot.innerHTML = mode.id === 'mode-3d'
      ? '<span><kbd>Drag</kbd> Orbit &nbsp; <kbd>Scroll</kbd> Zoom</span><span>Click to inspect</span>'
      : mode.id === 'mode-fe' ? '<span><kbd>Drag</kbd> Pan &nbsp; <kbd>Scroll</kbd> Zoom</span><span>Click an element to inspect</span>'
      : '<span>Interactive frame drawing</span><span>Click a member to inspect</span>';
    $('.draw', mode).append(foot);
    // Keep the compact mobile navigation outside the changing editor content.
    mode.dataset.panel = 'draw';
  }

  function updateWorkspace() {
    const mode = activeMode();
    if (!mode) return;
    $('#workspace-label').textContent = labels[mode.id];
    $$('.mobile-panels button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.panel === mode.dataset.panel)));
    updateStatus();
  }
  modes.forEach(mode => new MutationObserver(updateWorkspace).observe(mode, { attributes:true, attributeFilter:['hidden'] }));
  $$('.mobile-panels button').forEach(button => button.addEventListener('click', () => {
    const mode = activeMode();
    mode.dataset.panel = button.dataset.panel;
    updateWorkspace();
    window.dispatchEvent(new Event('resize'));
  }));

  const focus = $('#studio-focus');
  focus.addEventListener('click', () => {
    const focused = document.body.classList.toggle('focus-view');
    focus.setAttribute('aria-pressed', String(focused));
    $('span', focus).textContent = focused ? 'Exit focus' : 'Focus view';
    focus.title = focused ? 'Restore setup and results panels' : 'Expand the model viewport';
    window.dispatchEvent(new Event('resize'));
  });
  const compact = window.matchMedia('(max-width:900px)');
  compact.addEventListener('change', () => {
    if (compact.matches && document.body.classList.contains('focus-view')) focus.click();
  });

  const theme = $('#studio-theme');
  function syncTheme() {
    const dark = document.documentElement.dataset.theme === 'dark';
    theme.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
    theme.title = `Switch to ${dark ? 'light' : 'dark'} theme`;
    theme.innerHTML = dark
      ? '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg>'
      : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10z"/></svg>';
  }
  theme.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('tragwerk.theme', next); } catch (_) { /* Private browsing is supported. */ }
    syncTheme();
  });
  syncTheme();

  const guide = $('#studio-guide');
  $$('[data-help]').forEach(button => button.addEventListener('click', () => guide.showModal()));
  $('#guide-close').addEventListener('click', () => guide.close());
  guide.addEventListener('click', event => {
    const rect = guide.getBoundingClientRect();
    if (event.target === guide && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) guide.close();
  });

  // Reuse each workspace's original import/export controls and validation.
  $('#studio-export').addEventListener('click', () => {
    const mode = activeMode();
    const id = { 'mode-3d':'#b-json', 'mode-2d':'#btn-json', 'mode-fe':'#f-json' }[mode.id];
    $(id).click();
  });
  for (const drawer of $$('.drawer')) {
    const text = $('textarea', drawer);
    const error = document.createElement('p');
    error.className = 'hint';
    error.setAttribute('role', 'alert');
    error.hidden = true;
    text.after(error);
    text.addEventListener('input', () => { error.hidden = true; });
    const button = document.createElement('button');
    button.className = 'btn primary';
    button.textContent = 'Download JSON';
    button.addEventListener('click', () => {
      let data;
      try { data = JSON.parse(text.value); } catch (_) {
        error.textContent = 'Enter valid JSON before downloading.';
        error.hidden = false;
        text.focus();
        return;
      }
      error.hidden = true;
      const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${String(data?.name || 'tragwerk-model').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-|-$/g, '').slice(0, 100) || 'tragwerk-model'}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    $('.row:last-child', drawer).append(button);
    // The original drawers use role=dialog. Add a focus loop and restore focus.
    let returnFocus;
    new MutationObserver(() => {
      if (!drawer.hidden) {
        returnFocus = $('#studio-export');
        $('.studio-topbar').inert = true;
        $('.studio-rail').inert = true;
        error.hidden = true;
        text.focus();
      } else {
        $('.studio-topbar').inert = false;
        $('.studio-rail').inert = false;
        if (returnFocus) returnFocus.focus();
      }
    }).observe(drawer, {attributes:true, attributeFilter:['hidden']});
    drawer.addEventListener('keydown', event => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); drawer.hidden = true; }
      if (event.key !== 'Tab') return;
      const items = $$('button:not([disabled]),textarea,input,select,a[href]', drawer).filter(el => !el.hidden);
      if (event.shiftKey && document.activeElement === items[0]) { event.preventDefault(); items.at(-1).focus(); }
      else if (!event.shiftKey && document.activeElement === items.at(-1)) { event.preventDefault(); items[0].focus(); }
    });
  }

  // Real tab controls support keyboard navigation as well as pointer input.
  for (const list of $$('[role="tablist"]')) {
    list.addEventListener('keydown', event => {
      if (!['ArrowRight','ArrowLeft','ArrowDown','ArrowUp','Home','End'].includes(event.key)) return;
      const buttons = $$('[role="tab"]', list).filter(button => !button.disabled);
      const current = buttons.indexOf(document.activeElement);
      if (current < 0) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
        : (current + (['ArrowRight','ArrowDown'].includes(event.key) ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next].click(); buttons[next].focus();
    });
  }

  function updateStatus() {
    const mode = activeMode();
    let text = '';
    if (mode && mode.id === 'mode-3d') {
      const state = window.TragwerkStudio3D && window.TragwerkStudio3D.S;
      if (state && state.data) text = `${state.data.nodes.length} nodes · ${state.data.members.length} members`;
    } else if (mode && mode.id === 'mode-fe') {
      const result = window.TragwerkFE && window.TragwerkFE.S.res;
      if (result) text = `${result.model.nodes.length} nodes · ${result.model.elements.length} elements`;
    } else if (mode) {
      const title = $('#titleblock');
      const entries = title ? Array.from(title.children) : [];
      const model = entries.find(el => el.textContent === 'Model');
      text = model && model.nextElementSibling ? model.nextElementSibling.textContent : '2D frame analysis';
    }
    $('#studio-stats').textContent = text || 'Workspace ready';
  }
  ['#b-title','#titleblock','#f-title'].forEach(selector => {
    const el = $(selector);
    if (el) new MutationObserver(updateStatus).observe(el, {childList:true,subtree:true});
  });
  updateWorkspace();
})();
