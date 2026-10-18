(() => {
  const toc = document.querySelector('.page-toc');
  if (!toc) return;

  const summary = toc.querySelector('summary');
  const currentLabel = toc.querySelector('.toc-current');
  const entries = [...toc.querySelectorAll('nav a[href^="#"]')]
    .map(link => ({ link, target: document.getElementById(link.hash.slice(1)) }))
    .filter(entry => entry.target);
  if (!entries.length) return;

  const wide = window.matchMedia('(min-width: 1200px)');
  let activeLink;
  let scheduled = false;

  function setActive(entry) {
    if (activeLink === entry.link) return;
    activeLink?.removeAttribute('aria-current');
    activeLink = entry.link;
    activeLink.setAttribute('aria-current', 'location');
    currentLabel.textContent = activeLink.textContent.trim();
  }

  function update() {
    scheduled = false;
    const offset = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 76;
    let current = entries[0];
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top <= offset + 8) current = entry;
      else break;
    }
    // The final section may be too short to reach the top of the viewport.
    const root = document.documentElement;
    if (window.scrollY > 0 && window.scrollY + root.clientHeight >= root.scrollHeight - 4) {
      current = entries[entries.length - 1];
    }
    setActive(current);
  }

  function scheduleUpdate() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(update);
  }

  function syncLayout() {
    toc.open = wide.matches;
    scheduleUpdate();
  }

  for (const entry of entries) {
    entry.link.addEventListener('click', event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (!wide.matches) {
        toc.open = false;
        entry.target.setAttribute('tabindex', '-1');
        entry.target.focus({ preventScroll: true });
      }
      // Native anchors preserve deep links and browser back/forward behavior.
      setActive(entry);
    });
  }

  toc.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !wide.matches && toc.open) {
      toc.open = false;
      summary.focus();
    }
  });
  document.addEventListener('click', event => {
    if (!wide.matches && toc.open && !toc.contains(event.target)) toc.open = false;
  });
  window.addEventListener('scroll', scheduleUpdate, { passive: true });
  window.addEventListener('resize', scheduleUpdate);
  window.addEventListener('hashchange', scheduleUpdate);
  window.addEventListener('pageshow', scheduleUpdate);
  document.addEventListener('load', scheduleUpdate, true);
  wide.addEventListener('change', syncLayout);
  new ResizeObserver(scheduleUpdate).observe(document.querySelector('main'));
  document.fonts.ready.then(scheduleUpdate);
  syncLayout();
})();
