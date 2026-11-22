/* ==========================================================================
   layouts.js  -  turns ONE slide (data from data/slides.js) into HTML.

   Every slide is drawn on a fixed 1920x1080 "stage":
       header band (150px)  -> title, LASIGE logo area, slide number
       content band (840px) -> what each layout below draws
       footer band  (90px)  -> university logos

   A "layout" is just a function:  (slide) => HTML string for the content band.
   Add a new kind of slide by adding a function to LAYOUTS below.
   Pure string building (no DOM access) so it also runs in Node for the checker.
   ========================================================================== */
(function (root) {
  'use strict';
  var UI = root.UI;
  var at = UI.at;

  /* ---- small shared pieces ------------------------------------------------ */

  function captionBand(c, extraStyle) {
    if (!c) return '';
    return '<div class="caption ' + (c.cls || '') + '"' + at(c.at) + ' style="' + (extraStyle || '') + '">' + c.html + '</div>';
  }

  /* ---- logos -------------------------------------------------------------
     Two real files are expected (DECK.meta.logos in data/slides.js):
       header: logos/LASIGE.png               (top right of every slide)
       footer: logos/U_lisboa-fcul-fct.png    (ONE image with the three university logos)
     Both are meant to be light/white artwork on a transparent background (for the maroon bands).
     While a file is flagged usable:false (e.g. it is blank), a labelled placeholder is drawn instead. */
  function logoCfg() { var m = (root.DECK && root.DECK.meta && root.DECK.meta.logos) || {}; return m; }
  function headerLogo(extra) {
    var c = logoCfg().header;
    if (c && c.usable) return '<img class="hl-img' + (extra ? ' ' + extra : '') + '" alt="LASIGE" src="' + c.src + '">';
    return '<span class="logo-slot ph-only"><span class="ph">LASIGE</span></span>';
  }
  function footerLogo(extra) {
    var c = logoCfg().footer;
    if (c && c.usable) return '<img class="fl-img' + (extra ? ' ' + extra : '') + '" alt="Universidade de Lisboa, Ciências ULisboa, FCT" src="' + c.src + '">';
    return '<span class="logo-slot ph-only wide"><span class="ph">Universidade de Lisboa · Ciências ULisboa · FCT</span></span>';
  }
  function footerLogos() { return '<div class="ftr">' + footerLogo() + '</div>'; }

  /* ---- layouts ------------------------------------------------------------ */

  var LAYOUTS = {};

  /** Cover and closing slide (no header band, full maroon). */
  LAYOUTS.cover = function (s) {
    var d = s.cover;
    return '<div class="cover">' +
      '<div class="cover-logos">' + headerLogo('cover-h') + footerLogo('cover-f') + '</div>' +
      '<h1 class="cover-title">' + d.title + '</h1>' +
      '<div class="cover-rule"></div>' +
      '<div class="cover-by">' + d.by + '</div>' +
      (d.supervisors ? '<div class="cover-sup">' + d.supervisors + '</div>' : '') +
      '</div>';
  };

  /** Rows of pills with a coloured badge on the left (RQs, objectives, limitations...). */
  LAYOUTS.pillRows = function (s) {
    var rowH = s.rowH || 170, gap = s.gap == null ? 22 : s.gap;
    var h = '<div class="flow pad" style="gap:' + gap + 'px;justify-content:' + (s.justify || 'center') + '">';
    if (s.kicker) h += '<div class="kicker"' + at(s.kicker.at) + '>' + s.kicker.html + '</div>';
    s.rows.forEach(function (r, i) {
      var tone = r.tone || s.tone || 'orange';
      h += '<div class="pill-row tone-' + tone + (r.cls ? ' ' + r.cls : '') + '"' + at(r.at) +
        ' style="height:' + rowH + 'px;' + (r.badgeColor ? '--badge:' + r.badgeColor + ';' : '') + '">' +
        '<div class="pill-badge" style="width:' + (s.badgeW || 250) + 'px">' + r.badge + '</div>' +
        '<div class="pill-text" style="font-size:' + (s.fs || 34) + 'px">' + r.html + '</div></div>';
    });
    if (s.caption) h += captionBand(s.caption);
    return h + '</div>';
  };

  /** Status rows: green / amber / grey. The WORD is always shown, never colour alone. */
  LAYOUTS.statusRows = function (s) {
    var h = '<div class="flow pad" style="gap:22px">';
    s.rows.forEach(function (r) {
      h += '<div class="pill-row status-' + r.status + '"' + at(r.at) + ' style="height:' + (r.h || 170) + 'px">' +
        '<div class="pill-badge" style="width:' + (s.badgeW || 360) + 'px">' + r.badge + '</div>' +
        '<div class="pill-text" style="font-size:' + (s.fs || 33) + 'px">' + r.html + '</div></div>';
    });
    return h + '</div>';
  };

  /** Generic columns of cards, optional intro line above and caption band below. */
  LAYOUTS.cards = function (s) {
    var h = '<div class="flow pad" style="gap:26px">';
    if (s.intro) h += '<div class="intro"' + at(s.intro.at) + '>' + s.intro.html + '</div>';
    h += '<div class="cards" style="grid-template-columns:' + (s.columns || 'repeat(' + s.cols.length + ',1fr)') + ';height:' + (s.cardsH || 520) + 'px">';
    s.cols.forEach(function (c) {
      h += '<div class="card ' + (c.cls || '') + '"' + at(c.at) + '><h3>' + c.title + '</h3><div class="card-body" style="font-size:' + (s.fs || 29) + 'px">' + c.html + '</div></div>';
    });
    h += '</div>';
    if (s.caption) h += captionBand(s.caption);
    return h + '</div>';
  };

  /** Big statements revealed one by one (Future Work, The Answer...). */
  LAYOUTS.stat = function (s) {
    var h = '<div class="flow pad stat" style="gap:' + (s.gap || 46) + 'px;justify-content:center">';
    s.lines.forEach(function (l) {
      h += '<div class="stat-line ' + (l.cls || '') + '"' + at(l.at) + ' style="font-size:' + (l.fs || 52) + 'px">' + l.html + '</div>';
    });
    return h + '</div>';
  };

  /** Two columns: left card and right card, with optional caption. */
  LAYOUTS.compare = function (s) {
    var h = '<div class="flow pad" style="gap:26px">';
    if (s.intro) h += '<div class="intro"' + at(s.intro.at) + '>' + s.intro.html + '</div>';
    h += '<div class="cards" style="grid-template-columns:1fr 1fr;height:' + (s.cardsH || 560) + 'px">';
    s.cols.forEach(function (c) {
      h += '<div class="card big ' + (c.cls || '') + '"' + at(c.at) + '><h3>' + c.title + '</h3><div class="card-body" style="font-size:' + (s.fs || 34) + 'px">' + c.html + '</div></div>';
    });
    h += '</div>';
    if (s.caption) h += captionBand(s.caption);
    return h + '</div>';
  };

  /** Headline Results: aligned table, DV rows light, BoC rows darker (colour = domain only). */
  LAYOUTS.results = function (s) {
    var h = '<div class="flow pad" style="gap:12px">';
    h += '<div class="tr th" style="grid-template-columns:' + s.cols + '">' + s.head.map(function (x) { return '<div>' + x + '</div>'; }).join('') + '</div>';
    s.rows.forEach(function (r) {
      h += '<div class="tr dom-' + r.domain + '"' + at(r.at) + ' style="grid-template-columns:' + s.cols + '">' +
        r.cells.map(function (c) {
          if (typeof c === 'string') return '<div>' + c + '</div>';
          return '<div class="cell"' + UI.clsAt(c.cl) + '>' + c.t + '</div>';
        }).join('') + '</div>';
    });
    if (s.caption) h += captionBand(s.caption, 'font-size:27px;font-style:normal;font-weight:600');
    return h + '</div>';
  };

  /** Literature comparison (Scale and methods, Not Capability). */
  LAYOUTS.studies = function (s) {
    var h = '<div class="flow pad" style="gap:12px">';
    h += '<div class="tr th studies" style="grid-template-columns:' + s.cols + '">' + s.head.map(function (x) { return '<div>' + x + '</div>'; }).join('') + '</div>';
    s.rows.forEach(function (r) {
      h += '<div class="study ' + (r.cls || '') + '"' + at(r.at) + '>' +
        '<div class="study-main" style="grid-template-columns:' + s.cols + '">' +
        '<div class="study-name">' + r.study + '</div><div>' + r.size + '</div><div>' + r.result + '</div></div>' +
        '<div class="study-io">' + r.io + '</div></div>';
    });
    if (s.caption) h += captionBand(s.caption);
    return h + '</div>';
  };

  /** Judges' comments: intro band + three quote cards. */
  LAYOUTS.judges = function (s) {
    var h = '<div class="flow pad" style="gap:28px">';
    h += '<div class="caption top"' + at(s.intro.at) + '>' + s.intro.html + '</div>';
    h += '<div class="cards" style="grid-template-columns:repeat(3,1fr);height:' + (s.cardsH || 560) + 'px;gap:30px">';
    s.cards.forEach(function (c) {
      h += '<div class="qcard"' + at(c.at) + '><div class="qbadge">' + c.badge + '</div>' +
        '<div class="qtext">' + c.quotes.map(function (q) { return '<p>&ldquo;' + q + '&rdquo;</p>'; }).join('') + '</div>' +
        '<div class="qmeta">' + c.meta + '</div></div>';
    });
    return h + '</div></div>';
  };

  /** Free-form: html is positioned by the slide itself (diagrams, evidence...). */
  LAYOUTS.custom = function (s) { return typeof s.html === 'function' ? s.html(s) : s.html; };

  /* ---- frame (header + content + footer) --------------------------------- */

  function renderSlideHTML(s) {
    var isCover = s.layout === 'cover';
    var body = (LAYOUTS[s.layout] || LAYOUTS.custom)(s);
    if (s.footnote && !isCover) body += UI.box({ x: 80, y: 786, w: 1760, h: 40, cls: 'src fnote', style: 'font-size:20px;color:#6b6560;line-height:40px;white-space:nowrap' }, s.footnote);
    if (isCover) return '<div class="slide cover-slide">' + body + '</div>';
    return '<div class="slide">' +
      '<div class="hdr"><h1>' + s.title + '</h1>' +
      '<div class="hdr-logo">' + headerLogo() + '</div>' +
      (s.number != null ? '<div class="num">' + s.number + '</div>' : (s.badge ? '<div class="num appx">' + s.badge + '</div>' : '')) +
      '</div>' +
      '<div class="content">' + body + '</div>' + footerLogos() + '</div>';
  }

  root.Layouts = { LAYOUTS: LAYOUTS, renderSlideHTML: renderSlideHTML, captionBand: captionBand };
  if (typeof module !== 'undefined') module.exports = root.Layouts;
})(typeof window !== 'undefined' ? window : globalThis);
