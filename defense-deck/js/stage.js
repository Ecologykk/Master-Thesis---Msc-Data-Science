/* ==========================================================================
   stage.js - draws a slide at a given reveal STEP inside a scaled 1920x1080 box.

   Used for: the audience window, the two previews in the presenter window
   (current / next), the overview thumbnails and the print mode.

   HOW "NOTHING LEAKS TO THE AUDIENCE" WORKS
   Elements with data-at="N" are "hidden" while step < N. A hidden element is
   not just made invisible: its children are REMOVED from the page and kept in a
   JavaScript variable (el._stash). Only an empty, invisible "shell" with the
   same size stays behind so the layout does not jump when it appears.
   => find-in-page (Ctrl+F), select-all and the DOM inspector see nothing.
   When the step reaches N the children are put back (with a short fade).
   ========================================================================== */
(function (root) {
  'use strict';
  var DECK = root.DECK;
  var Layouts = root.Layouts;
  var reduceMotion = false;
  try { reduceMotion = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}

  /* ---- hide / show one element ------------------------------------------ */
  function strip(el) {
    if (!(el instanceof root.SVGElement)) {          // HTML: remember its size so layout stays put
      el.style.minWidth = el.offsetWidth + 'px';
      el.style.minHeight = el.offsetHeight + 'px';
      el.style.visibility = 'hidden';
    }
    var frag = root.document.createDocumentFragment();
    while (el.firstChild) frag.appendChild(el.firstChild);
    el._stash = frag;
  }
  function restore(el, animate) {
    if (!(el instanceof root.SVGElement)) {
      el.style.minWidth = ''; el.style.minHeight = ''; el.style.visibility = '';
    }
    el.appendChild(el._stash);
    el._stash = undefined;
    if (animate && !reduceMotion) {
      el.classList.add('fade-in');
      setTimeout(function () { el.classList.remove('fade-in'); }, 260);
    }
  }

  /** Make the DOM under `rootEl` show exactly the content for `step`. */
  function applyStep(rootEl, step, animate) {
    var changed = true, guard = 0;
    while (changed && guard++ < 25) {              // loop: restoring a parent can reveal nested elements
      changed = false;
      var list = rootEl.querySelectorAll('[data-at]');
      for (var i = 0; i < list.length; i++) {
        var el = list[i];
        if (!el.isConnected) continue;             // was removed with a hidden ancestor
        var hide = Number(el.getAttribute('data-at')) > step;
        var stripped = el._stash !== undefined;
        if (hide && !stripped) { strip(el); changed = true; }
        else if (!hide && stripped) { restore(el, animate); changed = true; }
      }
    }
    // class switches: data-cls="2:b" -> class "b" when step >= 2; "1-1:on" -> only while 1 <= step <= 1
    var sw = rootEl.querySelectorAll('[data-cls]');
    for (var k = 0; k < sw.length; k++) {
      var parts = sw[k].getAttribute('data-cls').split(';');
      for (var p = 0; p < parts.length; p++) {
        var kv = parts[p].split(':');
        var ab = kv[0].split('-'), on = ab.length > 1 ? (step >= Number(ab[0]) && step <= Number(ab[1])) : step >= Number(kv[0]);
        sw[k].classList.toggle(kv[1], on);
      }
    }
  }

  /* ---- StageView --------------------------------------------------------- */
  function StageView(host, opts) {
    opts = opts || {};
    this.host = host;
    this.opts = opts;                                // {live:true => mount video, inertLinks:true}
    host.classList.add('stage-host');
    this.stage = root.document.createElement('div');
    this.stage.className = 'stage';
    this.zoom = root.document.createElement('div');
    this.zoom.className = 'zoom-layer';
    this.stage.appendChild(this.zoom);
    host.appendChild(this.stage);
    this.slideIndex = -1; this.step = -1;
    var self = this;
    if (opts.inertLinks !== false) {                 // links never navigate on the audience / previews
      this.stage.addEventListener('click', function (e) {
        var a = e.target.closest && e.target.closest('a');
        if (a) e.preventDefault();
      });
    }
    this._fit = function () { self.fit(); };
    if (root.ResizeObserver) { this._ro = new root.ResizeObserver(this._fit); this._ro.observe(host); }
    else root.addEventListener('resize', this._fit);
    this.fit();
  }
  StageView.prototype.fit = function () {
    var w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    var s = Math.min(w / 1920, h / 1080);
    var ox = (w - 1920 * s) / 2, oy = (h - 1080 * s) / 2;
    this.stage.style.transform = 'translate(' + ox + 'px,' + oy + 'px) scale(' + s + ')';
    this.scale = s;
  };
  /** Show slide `index` at `step`. highlight = step number whose new elements get a dashed outline. */
  StageView.prototype.show = function (index, step, o) {
    o = o || {};
    var slides = DECK.slides, slide = slides[index];
    if (!slide) return;
    var animate = false;
    if (slide.keep) { step = slide.keep[step]; if (o.highlight != null) o = { animate: o.animate, highlight: slide.keep[o.highlight] == null ? null : slide.keep[o.highlight], hiFrom: o.highlight > 0 ? slide.keep[o.highlight - 1] : -1 }; }
    if (index !== this.slideIndex) {
      this.zoom.innerHTML = Layouts.renderSlideHTML(slide);
      this.slideIndex = index;
      this.slideEl = this.zoom.firstChild;
      this.video = null;
      applyStep(this.zoom, step, false);
      if (this.opts.live && root.Video) root.Video.mount(this, slide);
    } else {
      animate = !!o.animate && step > this.step;
      applyStep(this.zoom, step, animate);
    }
    if (slide.live && root.Live && root.Live[slide.live]) {
      root.Live[slide.live](this.zoom, step, { animate: animate, prev: this.step }, slide);
    }
    this.step = step;
    // outline whatever the NEXT key press will reveal (presenter preview only)
    var old = this.zoom.querySelectorAll('.nx-new');
    for (var i = 0; i < old.length; i++) old[i].classList.remove('nx-new');
    if (o.highlight != null) {
      var lo = o.hiFrom == null ? o.highlight - 1 : o.hiFrom, nw = this.zoom.querySelectorAll('[data-at]');
      for (var j = 0; j < nw.length; j++) { var at = Number(nw[j].getAttribute('data-at')); if (at > lo && at <= o.highlight) nw[j].classList.add('nx-new'); }
    }
  };
  StageView.prototype.setBlank = function (b) { this.host.classList.toggle('blank', !!b); };
  /** zoom = null | {x:0..1, y:0..1, s:number}. */
  StageView.prototype.setZoom = function (z) {
    if (!z) { this.zoom.style.transform = 'none'; return; }
    this.zoom.style.transformOrigin = (z.x * 100) + '% ' + (z.y * 100) + '%';
    this.zoom.style.transform = 'scale(' + z.s + ')';
  };

  root.StageView = StageView;
  root.applyStep = applyStep;
})(window);
