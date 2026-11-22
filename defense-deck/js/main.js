/* ==========================================================================
   main.js - decides which "mode" this window is, then starts it.
       index.html            -> PRESENTER window (the one you control)
       index.html#audience   -> AUDIENCE window (projector; opened with key A)
       index.html#solo       -> SOLO: one window, keyboard control
       index.html?print      -> PRINT: all slides, every step visible (Ctrl+P -> Save as PDF)
   ========================================================================== */
(function (root) {
  'use strict';
  var doc = root.document;
  var hash = root.location.hash.toLowerCase();
  var isPrint = /[?&]print/.test(root.location.search);
  var mode = isPrint ? 'print' : hash.indexOf('audience') >= 0 ? 'audience' : hash.indexOf('solo') >= 0 ? 'solo' : 'presenter';
  doc.documentElement.className = 'mode-' + mode;
  doc.title = mode === 'audience' ? 'Defense deck - AUDIENCE' : mode === 'presenter' ? 'Defense deck - PRESENTER' : 'Defense deck';

  function startPrint(app) {
    app.innerHTML = '';
    root.DECK.slides.forEach(function (s, i) {
      var page = doc.createElement('div'); page.className = 'print-page' + (s.appendix ? ' appx' : '');
      var host = doc.createElement('div'); host.className = 'print-host'; page.appendChild(host); app.appendChild(page);
      var v = new root.StageView(host, { inertLinks: false });
      v.show(i, s.steps.length - 1);
    });
  }

  function go() {
    var app = doc.getElementById('app');
    if (mode === 'audience') root.Audience.start(app);
    else if (mode === 'solo') root.Presenter.solo(app);
    else if (mode === 'print') startPrint(app);
    else root.Presenter.start(app);
    root.addEventListener('hashchange', function () { if (!isPrint) root.location.reload(); });
  }

  // Wait for the bundled fonts, so hidden-element sizes are measured with the final font.
  var started = false;
  function once() { if (started) return; started = true; go(); }
  try {
    var faces = ['400 20px "Open Sans"', 'italic 400 20px "Open Sans"', '600 20px "Open Sans"', '700 20px "Open Sans"', 'italic 700 20px "Open Sans"'];
    Promise.all(faces.map(function (f) { return doc.fonts.load(f); })).then(once, once);
    setTimeout(once, 2500);   // never block forever
  } catch (e) { once(); }
})(window);
