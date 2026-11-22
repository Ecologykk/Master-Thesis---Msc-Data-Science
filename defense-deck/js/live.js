/* ==========================================================================
   live.js - the ANIMATED slide ("A Possible Agentic System").

   HOW IT WORKS (plain words)
   A slide with  live: "agenticLive"  gets this function called every time its
   step changes. For a given step the function fills the texts and sets the CSS
   classes of THAT step, so going BACK a step, or reloading a window, always gives
   the same picture. Animations are only decoration: they are started by adding a
   CSS class (css/live.css) and the resting state does not depend on them. With
   "reduce motion" set in the operating system, no animation is started at all.

   Only the content of steps already reached is written into the page, so the
   audience window never contains text about stages not yet shown.

   Steps:  0 = the case enters
           1..8 = the eight stages of the Portuguese civil process
           9 = the two memories                      10 = the decision stays with the judge
   The content of each stage (its small specialised agents, texts...) lives in
   data/slides.js (slide.liveData), NOT here.
   ========================================================================== */
(function (root) {
  'use strict';

  var reduce = false;
  try { reduce = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}

  function q(el, sel) { return el.querySelector(sel); }
  function qa(el, sel) { return Array.prototype.slice.call(el.querySelectorAll(sel)); }
  /** restart a CSS animation class on an element */
  function play(el, cls, delay) {
    if (!el) return;
    el.classList.remove(cls); void el.offsetWidth;
    el.style.animationDelay = (delay || 0) + 's';
    el.classList.add(cls);
  }
  function clearAnim(el) {
    qa(el, '[class*="lv-a-"]').forEach(function (n) {
      n.className = n.className.replace(/\blv-a-[\w-]+/g, '').replace(/\s+/g, ' ').trim(); n.style.animationDelay = '';
    });
  }

  var Live = {
    /**
     * el    : the element that contains the slide
     * step  : current step (0-based)
     * o     : {animate:boolean, prev:number}
     * slide : the slide object (slide.liveData has the stage table)
     */
    agenticLive: function (el, step, o, slide) {
      var root_ = q(el, '.lv'); if (!root_) return;
      var D = slide.liveData, S = D.stages, N = S.length;       // N = 8 stages
      var animate = !!(o && o.animate) && !reduce;
      clearAnim(root_);

      var STEP_GATE = -1, STEP_RECAP = -1, STEP_MEM = N + 1, STEP_END = N + 2;   // gating and recap steps were removed
      var n = (step >= 1 && step <= N) ? step : 0;               // current stage number (0 = none)
      var st = n ? S[n - 1] : null;
      var gating = step === STEP_GATE, memory = step === STEP_MEM, recap = step === STEP_RECAP, closing = step === STEP_END;
      var human = !!(st && st.kind === 'human');

      /* ---- progress rail: done / active / locked (stage gating) ---- */
      var railN = n || (gating ? 4 : 0);                           // the gating example uses stage 4
      qa(root_, '.lv-node').forEach(function (nd, i) {
        var k = i + 1, cls = 'lv-node';
        if (recap || closing || memory) cls += (k === 6 && !memory ? ' done human' : ' done');
        else if (k < railN) cls += ' done';
        else if (k === railN) cls += ' active' + (k === 6 ? ' human' : '');
        else cls += ' locked';
        nd.className = cls;
      });
      q(root_, '.lv-rail-fill').style.width = Math.round(((recap || closing || memory) ? 1 : railN ? (railN - 1) / (N - 1) : 0) * 1680) + 'px';

      /* ---- heading + gating sentence ---- */
      var head = q(root_, '.lv-head-t'), art = q(root_, '.lv-head-art'), gate = q(root_, '.lv-gate');
      var ht = D.headings;
      if (st) { head.textContent = st.n + ' · ' + st.name; art.textContent = st.art; }
      else { head.textContent = gating ? ht.gating : memory ? ht.memory : recap ? ht.recap : closing ? ht.closing : ht.start; art.textContent = ''; }
      gate.textContent = '';

      /* ---- input card ---- */
      q(root_, '.lv-in-body').textContent = st ? st.input : gating ? D.gatingInput : memory ? D.memoryInput : recap ? D.recapInput : closing ? D.closingInput : D.startInput;

      /* ---- the small specialised agents (cards, in sequence) ---- */
      var rows = qa(root_, '.lv-row');
      var cards = st ? st.agents : gating ? D.gatingCards : memory ? D.memoryCards : recap ? D.recapCards : closing ? D.closingCards : D.startCards;
      rows.forEach(function (r, i) {
        var c = cards[i];
        if (!c) { r.className = 'lv-row off'; q(r, '.lv-rtag').style.visibility = 'hidden'; return; }
        r.className = 'lv-row' + (c.cls ? ' ' + c.cls : '') + (cards[i + 1] ? ' has-next' : '') + (human && i === 0 ? ' human' : '');
        q(r, '.lv-num').textContent = c.num == null ? String(i + 1) : c.num;
        q(r, '.lv-rname').textContent = c.name;
        q(r, '.lv-rjob').textContent = c.job;
        var tg = q(r, '.lv-rtag'); tg.textContent = c.tool || ''; tg.style.visibility = c.tool ? 'visible' : 'hidden';
      });

      /* ---- local memory (this stage only) ---- */
      q(root_, '.lv-local-body').textContent = st ? st.local : gating ? D.gatingLocal : memory ? D.memoryLocal : (recap || closing) ? D.clearedLocal : D.emptyLocal;

      /* ---- global memory (grows across stages) ---- */
      var last = (recap || closing || memory) ? N : n;
      if (gating) last = 3;                                       // example: stage 4 sees stages 1-3 only
      var list = q(root_, '.lv-glist'); list.innerHTML = '';
      for (var i = 0; i < last; i++) {
        var li = root.document.createElement('div'); li.className = 'lv-gitem' + (S[i].kind === 'human' ? ' human' : ''); li.textContent = S[i].n + ' · ' + S[i].glob;
        if (i === last - 1 && animate && st) li.classList.add('lv-a-fade');
        list.appendChild(li);
      }
      q(root_, '.lv-gbar-fill').style.height = ((last / N) * 100) + '%';

      /* ---- output artifact ---- */
      q(root_, '.lv-out-body').textContent = st ? st.out : gating ? D.gatingOut : memory ? D.memoryOut : recap ? D.recapOut : closing ? D.closingOut : D.startOut;

      /* ---- one-line note ---- */
      var note = q(root_, '.lv-note');
      note.textContent = st ? st.note : gating ? D.gatingNote : memory ? D.memoryNote : recap ? D.recapNote : closing ? D.closingNote : D.startNote;
      note.className = 'lv-note' + (human || closing ? ' human' : '') + (st && n === N ? ' thesis' : '');

      /* ---- animations (decoration only) ---- */
      if (animate && n) {
        play(q(root_, '.lv-in'), 'lv-a-in', 0);
        if (n > 1) {
          var dot = q(root_, '.lv-raildot');
          dot.style.left = (120 + (n - 2) * 240) + 'px'; dot.style.setProperty('--dx', '240px'); play(dot, 'lv-a-rail', 0);
        }
        var t = 0.6;
        if (!human) {
          rows.forEach(function (r, i) { if (cards[i]) { play(r, "lv-a-pop", t); t += 0.6; } });
          play(q(root_, '.lv-dot-a'), 'lv-a-dota', t); play(q(root_, '.lv-dot-b'), 'lv-a-dotb', t + 0.3);
          play(q(root_, '.lv-local'), 'lv-a-flash', t + 0.4);
          play(q(root_, '.lv-out'), 'lv-a-out', t + 0.9);
        } else {
          play(rows[0], 'lv-a-pop', 0.7); play(q(root_, '.lv-note'), 'lv-a-flash', 1.2); play(q(root_, '.lv-out'), 'lv-a-out', 1.8);
        }
      } else if (animate) {
        play(q(root_, '.lv-note'), 'lv-a-flash', 0.1);
        rows.forEach(function (r, i) { if (cards[i]) play(r, 'lv-a-pop', 0.2 + i * 0.3); });
      }
    }
  };
  root.Live = Live;
})(window);
