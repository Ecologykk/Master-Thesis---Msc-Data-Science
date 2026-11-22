/* ==========================================================================
   video.js - the app-demo video on the "Judge Validation" slide.

   RULES
   - Never autoplays. It plays when the presenter presses V, or when somebody
     clicks the video area / the big play button (in the AUDIENCE window or solo mode).
   - File: assets/video/app_demo.mp4 (H.264 + AAC, "faststart": the index is at the
     start of the file, so the browser can begin immediately). If the file is missing
     the slide keeps its poster and nothing breaks.
   - Only the AUDIENCE (or solo) window mounts the real <video>; previews show the poster.

   BROWSER AUTOPLAY RULE (why a click may be needed)
   Browsers only allow sound after the person clicked inside THAT window once.
   If the audience window has not been clicked, V starts the video MUTED (allowed),
   and a click anywhere in the audience window turns the sound on. If even muted
   playback is refused, a "click here" button appears and the presenter window shows
   "Video: BLOCKED, click once in the audience window".
   The audience reports its state to the presenter (status chip) so you can see it.
   ========================================================================== */
(function (root) {
  'use strict';
  var doc = root.document;
  var Video = {
    current: null,      // the <video> element, if mounted
    panel: null,
    onStatus: null,     // set by audience.js: function ({state, time, dur})
    lastReport: 0,

    report: function (state, force) {
      var v = Video.current, now = Date.now();
      if (!force && state === 'playing' && now - Video.lastReport < 900) return;
      Video.lastReport = now;
      if (Video.onStatus) Video.onStatus({ state: state, time: v ? v.currentTime : 0, dur: v && isFinite(v.duration) ? v.duration : 0, muted: v ? v.muted : false });
    },

    /** Called by StageView after a slide is drawn (audience / solo only). */
    mount: function (view, slide) {
      Video.current = null; Video.panel = null;
      var panel = view.zoom.querySelector('.video-panel[data-video]');
      if (!panel) return;
      Video.panel = panel;
      var v = doc.createElement('video');
      v.preload = 'auto'; v.playsInline = true; v.controls = false; v.muted = false;
      var poster = panel.getAttribute('data-poster'); if (poster) v.poster = poster;
      v.addEventListener('loadedmetadata', function () { panel.classList.add('has-video'); Video.report('ready', true); });
      v.addEventListener('play', function () { panel.classList.add('is-playing'); panel.classList.remove('blocked'); Video.report('playing', true); });
      v.addEventListener('pause', function () { panel.classList.remove('is-playing'); Video.report('paused', true); });
      v.addEventListener('ended', function () { panel.classList.remove('is-playing'); Video.report('ended', true); });
      v.addEventListener('timeupdate', function () { Video.report('playing'); });
      v.addEventListener('error', function () { panel.classList.remove('has-video', 'is-playing'); v.remove(); Video.current = null; Video.report('error', true); });
      v.src = panel.getAttribute('data-video');
      panel.insertBefore(v, panel.firstChild);
      Video.current = v;
      // big play button + "blocked" message live in the panel (see slide html); clicking anywhere on the panel toggles play
      panel.addEventListener('click', function (e) { e.stopPropagation(); Video.command('toggle', true); });
      // any click inside this window = the browser's "user gesture": switch the sound on if it was started muted
      doc.addEventListener('click', Video.unmute, true);
    },

    /** after a user click: unmute (the browser now allows sound) */
    unmute: function () { var v = Video.current; if (v && v.muted) { try { v.muted = false; } catch (e) {} } },

    /** action: 'toggle' | 'play' | 'pause' | 'restart'. byClick = the user clicked in this window (gesture present). */
    command: function (action, byClick) {
      var v = Video.current;
      if (!v) { Video.report('missing', true); return; }
      function play() {
        var p;
        try { p = v.play(); } catch (e) { p = null; }
        if (p && p.catch) p.catch(function () {
          // sound refused: try muted (always allowed)
          v.muted = true;
          var p2; try { p2 = v.play(); } catch (e) { p2 = null; }
          if (p2 && p2.catch) p2.catch(function () { if (Video.panel) Video.panel.classList.add('blocked'); Video.report('blocked', true); });
        });
      }
      try {
        if (action === 'restart') { v.currentTime = 0; play(); }
        else if (action === 'pause' || (action === 'toggle' && !v.paused)) v.pause();
        else play();
      } catch (e) {}
    }
  };
  root.Video = Video;
})(window);
