/* ==========================================================================
   controller.js - the deck's "brain": where are we (slide, step), is the
   audience blanked or frozen, zoomed...  No drawing happens here.

   Terms
     slide index : position in DECK.slides (0..32 main, 33.. appendix)
     step        : reveal state inside a slide (0 = what shows on entry)
     frozen      : audience keeps showing a snapshot while the presenter browses
   ========================================================================== */
(function (root) {
  'use strict';
  var STORE = 'ljp-deck-state';

  function Controller(deck, onChange) {
    this.deck = deck;
    this.onChange = onChange || function () {};
    this.slide = 0; this.step = 0;
    this.blank = false;
    this.frozen = null;            // {slide, step} while frozen
    this.zoom = null;              // {x,y,s} or null
    this.returnTo = null;          // where to go back to after visiting an appendix slide
    this.mainCount = deck.slides.filter(function (s) { return !s.appendix; }).length;
    this.lastMain = this.mainCount - 1;
  }
  var P = Controller.prototype;

  P.stepsOf = function (i) { return this.deck.slides[i].steps.length; };
  P.isAppendix = function (i) { return !!this.deck.slides[i === undefined ? this.slide : i].appendix; };
  P._set = function (slide, step) {
    this.slide = slide; this.step = step; this.zoom = null; this._changed();
  };
  P._changed = function () { this.save(); this.onChange(); };

  /** next key press: reveal the next step, or move on to the next slide. */
  P.next = function () {
    var n = this.stepsOf(this.slide);
    if (this.step < n - 1) return this._set(this.slide, this.step + 1);
    if (this.isAppendix()) return;                       // appendix is a dead end on purpose
    if (this.slide < this.lastMain) this._set(this.slide + 1, 0);
  };
  /** previous: hide the last revealed step, or go to the previous slide (fully revealed). */
  P.prev = function () {
    if (this.step > 0) return this._set(this.slide, this.step - 1);
    if (this.isAppendix()) return;
    if (this.slide > 0) this._set(this.slide - 1, this.stepsOf(this.slide - 1) - 1);
  };
  P.nextSlide = function () {
    if (this.isAppendix()) return;
    if (this.slide < this.lastMain) this._set(this.slide + 1, 0);
  };
  P.prevSlide = function () {
    if (this.isAppendix()) return;
    if (this.slide > 0) this._set(this.slide - 1, 0);
  };
  /** jump (from overview / go-to). Entering an appendix slide remembers where we were. */
  P.goto = function (i, step) {
    if (i < 0 || i >= this.deck.slides.length) return;
    if (this.isAppendix(i) && !this.isAppendix()) this.returnTo = { slide: this.slide, step: this.step };
    if (!this.isAppendix(i)) this.returnTo = null;
    this._set(i, Math.max(0, Math.min(step || 0, this.stepsOf(i) - 1)));
  };
  P.returnFromAppendix = function () {
    if (!this.isAppendix()) return;
    var r = this.returnTo || { slide: this.lastMain, step: 0 };
    this.returnTo = null; this._set(r.slide, r.step);
  };
  /** resolve "13", "A2", slide id... to a slide index (or -1). */
  P.find = function (q) {
    q = String(q).trim().toLowerCase();
    if (!q) return -1;
    var s = this.deck.slides;
    for (var i = 0; i < s.length; i++) {
      if (String(s[i].id).toLowerCase() === q) return i;
      if (s[i].number != null && String(s[i].number) === q) return i;
      if (s[i].badge && String(s[i].badge).toLowerCase() === q) return i;
    }
    if (q === '0') return 0;
    if (q === 'last') return this.lastMain;
    return -1;
  };

  P.toggleBlank = function () { this.blank = !this.blank; this._changed(); };
  P.toggleFreeze = function () {
    this.frozen = this.frozen ? null : { slide: this.slide, step: this.step };
    this._changed();
  };
  /** Esc Esc: unblank, unfreeze and go back to the last position the audience was showing. */
  P.panic = function () {
    if (this.frozen) { this.slide = this.frozen.slide; this.step = this.frozen.step; }
    this.frozen = null; this.blank = false; this.zoom = null; this._changed();
  };
  P.toggleZoom = function (x, y) {
    this.zoom = this.zoom ? null : { x: x == null ? 0.5 : x, y: y == null ? 0.5 : y, s: 1.75 };
    this._changed();
  };

  /** What the audience window should show right now. */
  P.audienceState = function () {
    var a = this.frozen || { slide: this.slide, step: this.step };
    return { slide: a.slide, step: a.step, blank: this.blank, zoom: this.frozen ? null : this.zoom };
  };

  P.save = function () {
    try { root.localStorage.setItem(STORE, JSON.stringify({ slide: this.slide, step: this.step })); } catch (e) {}
  };
  P.restore = function () {
    try {
      var o = JSON.parse(root.localStorage.getItem(STORE) || 'null');
      if (o && this.deck.slides[o.slide]) { this.slide = o.slide; this.step = Math.min(o.step, this.stepsOf(o.slide) - 1); }
    } catch (e) {}
  };
  P.reset = function () { this.blank = false; this.frozen = null; this.zoom = null; this.returnTo = null; this._set(0, 0); };

  root.Controller = Controller;
})(window);
