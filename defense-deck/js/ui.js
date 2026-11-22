/* ==========================================================================
   ui.js  -  tiny helpers that turn pieces of slide content into HTML strings.

   WHY THIS FILE EXISTS
   data/slides.js holds the TEXT of the deck. To avoid repeating long HTML
   everywhere, slides.js calls these small helpers (UI.link, UI.box, ...).
   Nothing here touches the page: every function just returns a string, so
   the same file also runs inside Node (used by tools/check-deck.js).

   THE REVEAL RULE (important)
   Any element with  data-at="N"  is only shown from step N onwards.
   Before that, the audience window removes its CONTENT from the page
   (see js/stage.js), so hidden text can't be found, selected or read.
   ========================================================================== */
(function (root) {
  'use strict';

  /** data-at attribute (omitted when n is 0/undefined = visible from the start). */
  function at(n) { return n ? ' data-at="' + n + '"' : ''; }

  /** Optional class switch: cls('2:best') adds class "best" once step >= 2. */
  function clsAt(spec) { return spec ? ' data-cls="' + spec + '"' : ''; }

  /** A hyperlink. In the audience window links are inert (see stage.js). */
  function link(url, text, cls) {
    return '<a class="lnk' + (cls ? ' ' + cls : '') + '" href="' + url + '" target="_blank" rel="noopener">' + text + '</a>';
  }

  /**
   * Absolutely positioned box on the 1920x840 content area.
   * o = {x,y,w,h, cls, at, style, id}; inner = HTML string.
   * Use for diagrams, where we want full control of positions.
   */
  function box(o, inner) {
    // "center" boxes are flex columns: wrap the content in ONE div so <br>/text stay together
    if ((o.cls || '').indexOf('center') >= 0) inner = '<div>' + (inner || '') + '</div>';
    var s = 'left:' + o.x + 'px;top:' + o.y + 'px;';
    if (o.w != null) s += 'width:' + o.w + 'px;';
    if (o.h != null) s += 'height:' + o.h + 'px;';
    if (o.style) s += o.style;
    return '<div class="abs ' + (o.cls || '') + '"' + at(o.at) + clsAt(o.cl) +
      (o.attrs ? ' ' + o.attrs : '') + ' style="' + s + '">' + (inner || '') + '</div>';
  }

  /**
   * Full-area SVG layer (1920x840) for arrows/lines drawn behind or over boxes.
   * Children are strings made with UI.line / UI.arrow / UI.path.
   */
  function svgLayer(inner, z) {
    return '<svg class="abs svg-layer" width="1920" height="840" viewBox="0 0 1920 840" style="left:0;top:0;' +
      (z != null ? 'z-index:' + z + ';' : '') + '" xmlns="http://www.w3.org/2000/svg">' +
      '<defs>' +
      '<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#6c2118"/></marker>' +
      '<marker id="ahg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8a8f94"/></marker>' +
      '</defs>' + inner + '</svg>';
  }

  /** Straight arrow. o = {at, color, w, dash, grey, noHead} */
  function arrow(x1, y1, x2, y2, o) {
    o = o || {};
    var col = o.color || (o.grey ? '#8a8f94' : '#6c2118');
    return '<g' + at(o.at) + '><line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="' + col +
      '" stroke-width="' + (o.w || 4) + '"' + (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') +
      (o.noHead ? '' : ' marker-end="url(#' + (o.grey ? 'ahg' : 'ah') + ')"') + '/></g>';
  }

  /** Polyline/elbow arrow from a list of [x,y] points. */
  function path(points, o) {
    o = o || {};
    var col = o.color || (o.grey ? '#8a8f94' : '#6c2118');
    var d = points.map(function (p, i) { return (i ? 'L' : 'M') + p[0] + ',' + p[1]; }).join(' ');
    return '<g' + at(o.at) + '><path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + (o.w || 4) +
      '" stroke-linejoin="round"' + (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') +
      (o.noHead ? '' : ' marker-end="url(#' + (o.grey ? 'ahg' : 'ah') + ')"') + '/></g>';
  }

  root.UI = { at: at, clsAt: clsAt, link: link, box: box, svgLayer: svgLayer, arrow: arrow, path: path };
  if (typeof module !== 'undefined') module.exports = root.UI;
})(typeof window !== 'undefined' ? window : globalThis);
