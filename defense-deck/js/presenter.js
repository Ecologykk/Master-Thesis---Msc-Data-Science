/* ==========================================================================
   presenter.js - the PRESENTER window (laptop): previews, notes, timer,
   overview, go-to, keyboard control, and sending the state to the audience.

   Also contains the "solo" mode (one window, audience view, keyboard control)
   used when the room only allows mirrored/duplicated screens.
   ========================================================================== */
(function (root) {
  'use strict';
  var doc = root.document;
  var DECK = root.DECK;
  var TIMER_KEY = 'ljp-deck-timer';

  function el(tag, cls, html) { var e = doc.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function fmt(ms) { var s = Math.floor(ms / 1000); return pad(Math.floor(s / 60)) + ':' + pad(s % 60); }
  function fmtT(s) { s = Math.floor(s || 0); return Math.floor(s / 60) + ':' + (s % 60 < 10 ? '0' : '') + (s % 60); }
  function plain(html) { var d = doc.createElement('div'); d.innerHTML = html; return (d.textContent || '').replace(/\s+/g, ' ').trim(); }

  /* ================= PRESENTER ================= */
  function startPresenter(app) {
    var ctl = new root.Controller(DECK, render);
    ctl.restore();
    var audWin = null, lastHeard = 0, popupBlocked = false, notesScale = 1, seq = 0;

    /* ---- timer ---- */
    var timer = { running: false, elapsed: 0, startedAt: 0 };
    try { var t = JSON.parse(root.localStorage.getItem(TIMER_KEY) || 'null'); if (t) { timer.elapsed = t.elapsed || 0; } } catch (e) {}
    function timerMs() { return timer.elapsed + (timer.running ? Date.now() - timer.startedAt : 0); }
    function saveTimer() { try { root.localStorage.setItem(TIMER_KEY, JSON.stringify({ elapsed: timerMs() })); } catch (e) {} }
    function toggleTimer() {
      if (timer.running) { timer.elapsed = timerMs(); timer.running = false; }
      else { timer.startedAt = Date.now(); timer.running = true; }
      saveTimer(); renderTimer();
    }
    function resetTimer() { timer.running = false; timer.elapsed = 0; saveTimer(); renderTimer(); }

    /* ---- DOM ---- */
    app.innerHTML = '';
    app.className = 'pres';
    var top = el('header', 'p-top');
    var count = el('div', 'p-count'); var title = el('div', 'p-title');
    var aud = el('div', 'chip aud off', 'Audience: not connected'); var vidChip = el('div', 'chip vid hidden', ''); var modeChip = el('div', 'chip live', 'LIVE');
    var timerBox = el('div', 'p-timer');
    var tTime = el('span', 'p-time', '00:00'); var tTarget = el('span', 'p-target', ' / ' + pad(DECK.meta.targetMinutes) + ':00');
    var tBtn = el('button', 'btn', 'Start (T)'); var tReset = el('button', 'btn ghost', 'Reset (Shift+T)');
    var openBtn = el('button', 'btn accent', 'Open audience window (A)');
    timerBox.append(tTime, tTarget, tBtn, tReset);
    top.append(count, title, vidChip, aud, modeChip, timerBox, openBtn);

    var main = el('main', 'p-main');
    var colL = el('section', 'p-col');
    colL.append(el('div', 'p-label', 'NOW &nbsp;·&nbsp; what the audience sees'));
    var curWrap = el('div', 'p-prev'); var curHost = el('div', 'p-host'); curWrap.appendChild(curHost); colL.appendChild(curWrap);
    var stepsBox = el('div', 'p-steps'); colL.appendChild(stepsBox);
    var srcBox = el('div', 'p-src'); colL.appendChild(srcBox);
    var colR = el('section', 'p-col');
    var nextLabel = el('div', 'p-label', 'NEXT');
    var nextWrap = el('div', 'p-prev small'); var nextHost = el('div', 'p-host'); nextWrap.appendChild(nextHost);
    var notes = el('div', 'p-notes');
    colR.append(nextLabel, nextWrap, notes);
    main.append(colL, colR);

    var help = el('footer', 'p-help',
      '<b>→ Space PgDn</b> next · <b>← PgUp</b> back · <b>Shift+←/→</b> slide · <b>B</b> blank · <b>F</b> freeze · <b>Esc Esc</b> panic · ' +
      '<b>G</b> go to · <b>O</b> overview · <b>V</b> video · <b>Z</b> zoom · <b>N</b> notes size · <b>T</b> timer · <b>A</b> audience window · <b>R</b> back from appendix · <b>?</b> help');
    app.append(top, main, help);

    var curView = new root.StageView(curHost, { inertLinks: true });
    var nextView = new root.StageView(nextHost, { inertLinks: true });

    /* ---- overlays ---- */
    var overlay = el('div', 'overlay hidden'); app.appendChild(overlay);
    var overviewBuilt = false;
    function closeOverlay() { overlay.classList.add('hidden'); overlay.innerHTML = ''; overviewBuilt = false; }
    function openOverview() {
      overlay.innerHTML = ''; overlay.classList.remove('hidden');
      var box = el('div', 'ov-box');
      box.appendChild(el('div', 'ov-head', '<b>Overview</b> &nbsp; click a slide to jump (it opens at step 1) · <b>Esc</b> or <b>O</b> to close'));
      var grid = el('div', 'ov-grid');
      var appxDone = false, hosts = [];
      DECK.slides.forEach(function (s, i) {
        if (s.appendix && !appxDone) {
          appxDone = true;
          grid.appendChild(el('div', 'ov-divider', 'APPENDIX · backup slides for questions · NOT part of the normal flow'));
        }
        var cell = el('div', 'ov-cell' + (i === ctl.slide ? ' cur' : '') + (s.appendix ? ' appx' : ''));
        var host = el('div', 'ov-thumb'); cell.appendChild(host);
        cell.appendChild(el('div', 'ov-cap', '<b>' + (s.number != null ? s.number : (s.badge || '·')) + '</b> ' + plain(s.title)));
        cell.addEventListener('click', function () { closeOverlay(); ctl.goto(i, 0); });
        grid.appendChild(cell);
        hosts[i] = host;
      });
      box.appendChild(grid);
      overlay.appendChild(box);
      // draw thumbnails AFTER they are in the page (they need a size)
      DECK.slides.forEach(function (s, i) {
        var v = new root.StageView(hosts[i], { inertLinks: true });
        v.show(i, s.steps.length - 1);
      });
      var cur = overlay.querySelector('.ov-cell.cur'); if (cur) cur.scrollIntoView({ block: 'center' });
      overviewBuilt = true;
    }
    function openGoto() {
      overlay.innerHTML = ''; overlay.classList.remove('hidden');
      var box = el('div', 'goto-box', '<div>Go to slide number (0–19), <b>A1</b>–<b>A4</b> or <b>B1</b>–<b>B15</b> for backup, or a slide id. <b>Enter</b> to go, <b>Esc</b> to cancel.</div>');
      var inp = el('input', 'goto-in'); inp.type = 'text'; inp.autofocus = true; inp.setAttribute('autocomplete', 'off');
      var msg = el('div', 'goto-msg', '');
      inp.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
          var i = ctl.find(inp.value);
          if (i < 0) { msg.textContent = 'No slide "' + inp.value + '"'; return; }
          closeOverlay(); ctl.goto(i, 0);
        } else if (e.key === 'Escape') closeOverlay();
        e.stopPropagation();
      });
      box.append(inp, msg); overlay.appendChild(box); setTimeout(function () { inp.focus(); }, 0);
    }
    function openHelp() {
      overlay.innerHTML = ''; overlay.classList.remove('hidden');
      var rows = [
        ['→ / Space / PageDown', 'next step (or next slide when all steps are shown)'], ['← / PageUp', 'previous step (hides it again)'],
        ['Shift + → / ←', 'next / previous slide (skips the remaining steps)'], ['B  or  .', 'blank the audience screen (black)'],
        ['F', 'freeze the audience screen (you can browse privately); press F again to catch up'], ['Esc, Esc', 'panic: unblank, unfreeze, back to the last shown position'],
        ['G', 'go to a slide by number'], ['O', 'overview of all slides + appendix'], ['V', 'play / pause the demo video (audience window)'],
        ['Z', 'zoom the audience into the point under your mouse (in the NOW preview); Z again to zoom out'],
        ['N', 'bigger / smaller notes'], ['T  /  Shift+T', 'start-pause / reset the timer'], ['A', 'open (or re-attach) the audience window'],
        ['R', 'return from an appendix slide to where you were'], ['Home', 'back to the cover (resets blank/freeze)']
      ];
      overlay.appendChild(el('div', 'goto-box help-box', '<h2>Keyboard</h2><table>' + rows.map(function (r) { return '<tr><td><b>' + r[0] + '</b></td><td>' + r[1] + '</td></tr>'; }).join('') + '</table><div class="goto-msg">Esc to close</div>'));
    }

    /* ---- audience window ---- */
    function openAudience() {
      var url = root.location.href.split('#')[0].split('?')[0] + '#audience';
      var w = root.open(url, 'ljp-audience', 'popup=yes,width=1280,height=720,left=80,top=80');
      if (!w) { popupBlocked = true; renderAud(); return; }
      popupBlocked = false; audWin = w; root.Sync.setPeer(w);
      setTimeout(sendState, 600); setTimeout(sendState, 1500);
    }
    function sendState() {
      var a = ctl.audienceState();
      root.Sync.send({ t: 'state', seq: ++seq, slide: a.slide, step: a.step, blank: a.blank, zoom: a.zoom });
    }
    function renderAud() {
      var ok = Date.now() - lastHeard < 6000;
      aud.className = 'chip aud ' + (ok ? 'on' : 'off');
      aud.textContent = ok ? 'Audience: connected' : (popupBlocked ? 'Pop-up blocked: allow pop-ups, press A' : 'Audience: not connected (press A)');
    }

    /* ---- rendering ---- */
    function renderTimer() {
      var ms = timerMs(), target = DECK.meta.targetMinutes * 60000, hard = DECK.meta.hardStopMinutes * 60000;
      tTime.textContent = fmt(ms);
      timerBox.className = 'p-timer' + (ms > hard ? ' over2' : ms > target ? ' over' : ms > target * 0.9 ? ' warn' : '');
      tBtn.textContent = timer.running ? 'Pause (T)' : 'Start (T)';
    }
    function render() {
      var s = DECK.slides[ctl.slide], n = s.steps.length;
      count.innerHTML = (s.appendix ? 'APPENDIX ' + (s.badge || '') : 'Slide <b>' + (s.number != null ? s.number : (ctl.slide === 0 ? 'cover' : 'end')) + '</b> / ' + (ctl.lastMain)) +
        ' &nbsp;·&nbsp; step <b>' + (ctl.step + 1) + '</b>/' + n;
      title.textContent = plain(s.title);
      modeChip.className = 'chip ' + (ctl.blank ? 'blank' : ctl.frozen ? 'frozen' : 'live');
      modeChip.textContent = ctl.blank ? 'BLANK' : ctl.frozen ? 'FROZEN' : 'LIVE';
      if (ctl.blank && ctl.frozen) modeChip.textContent = 'BLANK + FROZEN';
      curView.show(ctl.slide, ctl.step, { animate: false });
      curView.setZoom(ctl.zoom);
      // next preview
      var ns, nstep, hl = null, label;
      if (ctl.step < n - 1) { ns = ctl.slide; nstep = ctl.step + 1; hl = nstep; label = 'NEXT KEY &nbsp;·&nbsp; step ' + (nstep + 1) + '/' + n + ' <span class="dash">(dashed outline = what appears)</span>'; }
      else if (!s.appendix && ctl.slide < ctl.lastMain) { ns = ctl.slide + 1; nstep = 0; label = 'NEXT SLIDE &nbsp;·&nbsp; ' + plain(DECK.slides[ns].title); }
      else { ns = null; label = s.appendix ? 'END OF APPENDIX SLIDE · press R to return' : 'END OF DECK'; }
      nextLabel.innerHTML = label;
      if (ns != null) { nextWrap.classList.remove('empty'); nextView.show(ns, nstep, { highlight: hl }); } else nextWrap.classList.add('empty');
      // step chips
      stepsBox.innerHTML = '';
      s.steps.forEach(function (st, i) {
        var c = el('button', 'step-chip' + (i === ctl.step ? ' cur' : '') + (i < ctl.step ? ' done' : ''), (i + 1) + ' · ' + st.cue);
        c.addEventListener('click', function () { ctl.goto(ctl.slide, i); });
        stepsBox.appendChild(c);
      });
      // notes
      var nh = '<h3>' + (s.steps[ctl.step].cue ? 'Step ' + (ctl.step + 1) + ': ' + s.steps[ctl.step].cue : '') + '</h3>';
      if (s.steps[ctl.step].say) nh += '<p class="say">' + s.steps[ctl.step].say + '</p>';
      if (Array.isArray(s.notes) && s.notes.length) nh += '<ul class="bullets">' + s.notes.map(function (b) { return '<li>' + b + '</li>'; }).join('') + '</ul>';
      else if (typeof s.notes === 'string' && s.notes) nh += s.notes.split('\n\n').map(function (p) { return '<p>' + p + '</p>'; }).join('');
      else nh += '<p class="muted">(No speaker notes for this slide.)</p>';
      notes.innerHTML = nh; notes.style.fontSize = (20 * notesScale) + 'px'; notes.scrollTop = 0;
      // sources
      srcBox.innerHTML = '';
      if (s.sources && s.sources.length) {
        srcBox.appendChild(el('div', 'p-label', 'SOURCES · click to open (opens here, never on the projector)'));
        s.sources.forEach(function (x) { srcBox.appendChild(el('div', 'src-i', '<a href="' + x.url + '" target="_blank" rel="noopener">' + x.label + '</a>')); });
      }
      if (s.appendix) srcBox.appendChild(el('div', 'p-label', 'Appendix slide: use R to return to the main flow'));
      renderAud();
      sendState();
    }

    /* ---- messages from the audience ---- */
    root.Sync.init('presenter', function (m) {
      if (m.t === 'vstat') {
        var vs = m.s, txt = vs === 'playing' ? 'Video: playing ' + fmtT(m.time) : vs === 'paused' ? 'Video: paused ' + fmtT(m.time) : vs === 'blocked' ? 'Video: BLOCKED, click once in the audience window' : vs === 'error' ? 'Video: file error' : vs === 'missing' ? 'Video: not on this slide' : vs === 'ended' ? 'Video: ended' : 'Video: ready';
        vidChip.textContent = txt + (vs === 'playing' && m.muted ? ' (muted: click the audience window for sound)' : ''); vidChip.className = 'chip vid ' + (vs === 'blocked' || vs === 'error' ? 'bad' : vs === 'playing' ? 'ok' : '');
        lastHeard = Date.now(); renderAud(); return;
      }
      if (m.t === 'hello' || m.t === 'ping') { lastHeard = Date.now(); renderAud(); if (m.t === 'hello' || m.seq !== seq) sendState(); }
    });
    setInterval(function () { renderAud(); renderTimer(); }, 500);

    /* ---- keyboard ---- */
    var lastEsc = 0;
    doc.addEventListener('keydown', function (e) {
      if (e.target && e.target.tagName === 'INPUT') return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var k = e.key, handled = true;
      if (!overlay.classList.contains('hidden')) {
        if (k === 'Escape' || k.toLowerCase() === 'o' || k === '?') closeOverlay();
        return;
      }
      switch (k) {
        case 'ArrowRight': case 'PageDown': case ' ': case 'Enter': e.shiftKey && k === 'ArrowRight' ? ctl.nextSlide() : ctl.next(); break;
        case 'ArrowLeft': case 'PageUp': case 'Backspace': e.shiftKey && k === 'ArrowLeft' ? ctl.prevSlide() : ctl.prev(); break;
        case 'Home': ctl.reset(); break;
        case 'b': case 'B': case '.': ctl.toggleBlank(); break;
        case 'f': case 'F': ctl.toggleFreeze(); break;
        case 'Escape': { var now = Date.now(); if (now - lastEsc < 900) { ctl.panic(); lastEsc = 0; } else lastEsc = now; break; }
        case 'g': case 'G': openGoto(); e.preventDefault(); break;
        case 'o': case 'O': openOverview(); break;
        case '?': openHelp(); break;
        case 'v': case 'V': root.Sync.send({ t: 'video', a: 'toggle' }); break;
        case 'z': case 'Z': ctl.toggleZoom(zoomPt.x, zoomPt.y); break;
        case 'n': case 'N': notesScale = notesScale >= 1.8 ? 0.8 : notesScale + 0.2; render(); break;
        case 't': case 'T': e.shiftKey ? resetTimer() : toggleTimer(); break;
        case 'a': case 'A': openAudience(); break;
        case 'r': case 'R': ctl.returnFromAppendix(); break;
        default: handled = false;
      }
      if (handled) e.preventDefault();
    });
    var zoomPt = { x: 0.5, y: 0.5 };
    curHost.addEventListener('mousemove', function (e) {
      var r = curHost.getBoundingClientRect();
      zoomPt = { x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)) };
    });
    openBtn.addEventListener('click', openAudience);
    tBtn.addEventListener('click', toggleTimer); tReset.addEventListener('click', resetTimer);
    count.addEventListener('click', openOverview);

    renderTimer(); render();
    root.__ctl = ctl;   // handy for testing in the browser console
  }

  /* ================= SOLO (single window, keyboard) ================= */
  function startSolo(app) {
    app.className = 'solo'; app.innerHTML = '';
    var host = el('div', 'solo-host'); app.appendChild(host);
    var view = new root.StageView(host, { live: true });
    var ctl = new root.Controller(DECK, function () {
      var a = ctl.audienceState();
      view.show(a.slide, a.step, { animate: true }); view.setBlank(a.blank); view.setZoom(a.zoom);
    });
    ctl.restore();
    var a = ctl.audienceState(); view.show(a.slide, a.step);
    var lastEsc = 0;
    doc.addEventListener('keydown', function (e) {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var k = e.key, h = true;
      switch (k) {
        case 'ArrowRight': case 'PageDown': case ' ': case 'Enter': e.shiftKey && k === 'ArrowRight' ? ctl.nextSlide() : ctl.next(); break;
        case 'ArrowLeft': case 'PageUp': case 'Backspace': e.shiftKey && k === 'ArrowLeft' ? ctl.prevSlide() : ctl.prev(); break;
        case 'Home': ctl.reset(); break;
        case 'b': case 'B': case '.': ctl.toggleBlank(); break;
        case 'v': case 'V': root.Video.command('toggle'); break;
        case 'z': case 'Z': ctl.toggleZoom(0.5, 0.5); break;
        case 'Escape': { var n = Date.now(); if (n - lastEsc < 900) { ctl.panic(); lastEsc = 0; } else lastEsc = n; break; }
        case 'F11': h = false; break;
        default: h = false;
      }
      if (h) e.preventDefault();
    });
    host.addEventListener('click', function (e) { var r = host.getBoundingClientRect(); (e.clientX - r.left) > r.width / 2 ? ctl.next() : ctl.prev(); });
    root.__ctl = ctl;
  }

  root.Presenter = { start: startPresenter, solo: startSolo };
})(window);
