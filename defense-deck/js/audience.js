/* ==========================================================================
   audience.js - the AUDIENCE window (projector).
   It never decides anything: it only draws the state sent by the presenter.
   Keyboard/mouse do nothing here, except  F  or double-click = fullscreen,
   so the slide can't be advanced by accident.
   ========================================================================== */
(function (root) {
  'use strict';
  var doc = root.document, DECK = root.DECK;

  function startAudience(app) {
    app.className = 'aud-root'; app.innerHTML = '';
    var host = doc.createElement('div'); host.className = 'aud-host'; app.appendChild(host);
    var view = new root.StageView(host, { live: true });
    var last = { slide: 0, step: 0, blank: false, zoom: null, seq: -1 };

    // start from the last position the presenter saved (if any), so a reload recovers
    try {
      var o = JSON.parse(root.localStorage.getItem('ljp-deck-state') || 'null');
      if (o && DECK.slides[o.slide]) { last.slide = o.slide; last.step = Math.min(o.step, DECK.slides[o.slide].steps.length - 1); }
    } catch (e) {}
    view.show(last.slide, last.step);

    function apply(m) {
      var slideChanged = m.slide !== last.slide;
      view.show(m.slide, m.step, { animate: !slideChanged });
      view.setBlank(m.blank);
      view.setZoom(m.zoom);
      last = { slide: m.slide, step: m.step, blank: m.blank, zoom: m.zoom, seq: m.seq };
    }

    root.Sync.init('audience', function (m) {
      if (m.t === 'state') {
        if (m.slide === last.slide && m.step === last.step && m.blank === last.blank && JSON.stringify(m.zoom) === JSON.stringify(last.zoom)) { last.seq = m.seq; return; }
        apply(m);
      } else if (m.t === 'video') {
        root.Video.command(m.a);
      }
    });
    root.Video.onStatus = function (st) { root.Sync.send({ t: 'vstat', s: st.state, time: st.time, dur: st.dur, muted: st.muted }); };
    // tell the presenter we exist (now, and every 2 seconds)
    root.Sync.send({ t: 'hello' });
    setInterval(function () { root.Sync.send({ t: 'ping', seq: last.seq }); }, 2000);

    function fullscreen() {
      try {
        if (doc.fullscreenElement) doc.exitFullscreen(); else doc.documentElement.requestFullscreen();
      } catch (e) {}
    }
    doc.addEventListener('keydown', function (e) {
      if (e.key === 'f' || e.key === 'F') { fullscreen(); }
      if (e.key !== 'F11' && e.key !== 'F5' && !e.ctrlKey && !e.metaKey && !e.altKey) e.preventDefault();   // swallow navigation keys
    });
    host.addEventListener('dblclick', fullscreen);
  }
  root.Audience = { start: startAudience };
})(window);
