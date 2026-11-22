/* ==========================================================================
   tools/check-deck.js  -  OPTIONAL checker (needs Node, which the deck itself does NOT need).

   Run from Git Bash, from the folder "Tese de Mestrado":
       node defense-deck/tools/check-deck.js
       node defense-deck/tools/check-deck.js --text      (also prints each slide's text)

   What it checks
     - exactly 35 main slides + the appendix slides (A1..)
     - unique ids, titles, at least one step per slide
     - every data-at="N" inside a slide is a valid step number (< number of steps)
     - every image referenced exists in assets/
     - every URL required by the brief is present in the deck data
     - lists logo / video files that are still missing (expected: see ASSETS_TODO.md)
   Exit code 0 = no errors, 1 = errors.
   ========================================================================== */
'use strict';
const fs = require('fs');
const path = require('path');
global.window = global;                         // the deck files expect a browser-like "window"
const ROOT = path.join(__dirname, '..');
require(path.join(ROOT, 'js', 'ui.js'));
require(path.join(ROOT, 'data', 'slides.js'));
require(path.join(ROOT, 'js', 'layouts.js'));
const DECK = global.DECK, Layouts = global.Layouts;

let errors = 0, warnings = 0;
const err = (m) => { errors++; console.log('  ERROR  ' + m); };
const warn = (m) => { warnings++; console.log('  warn   ' + m); };
const strip = (h) => h.replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&ldquo;|&rdquo;/g, '"').replace(/&quot;/g, '"').replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/\s+/g, ' ').trim();

const main = DECK.slides.filter(s => !s.appendix), appx = DECK.slides.filter(s => s.appendix);
console.log(`Slides: ${main.length} main + ${appx.length} appendix`);
if (main.length !== 20) err(`expected 20 main slides (short deck), found ${main.length}`);
if (appx.length !== 29) err(`expected 29 backup slides (4 original + 25 moved), found ${appx.length}`);
if (DECK.slides.slice(0, main.length).some(s => s.appendix)) err('appendix slides must come after all main slides');

const ids = new Set();
let totalSteps = 0;
const allHtml = [];
DECK.slides.forEach((s, i) => {
  const tag = `#${i} ${s.id}`;
  if (ids.has(s.id)) err(`${tag}: duplicate id`); ids.add(s.id);
  if (!s.title) err(`${tag}: missing title`);
  if (!s.steps || !s.steps.length) { err(`${tag}: no steps`); return; }
  totalSteps += s.steps.length;
  if (!s.appendix && s.number != null && s.number !== i) err(`${tag}: printed number ${s.number} != index ${i}`);
  if (!(typeof s.notes === 'string' || Array.isArray(s.notes))) err(`${tag}: notes must be a list of bullets (or empty)`);
  if (Array.isArray(s.notes)) { if (s.notes.length > 10) warn(`${tag}: ${s.notes.length} bullets (aim for 3-7)`); s.notes.forEach(b => { if (b.split(/\s+/).length > 16) warn(`${tag}: long bullet: ${b.slice(0, 50)}...`); }); }
  let html;
  try { html = Layouts.renderSlideHTML(s); } catch (e) { err(`${tag}: render failed: ${e.message}`); return; }
  allHtml.push(html);
  // data-at values must be valid step numbers (short deck: s.keep lists the OLD step states that remain)
  const oldN = s.oldN || s.steps.length;
  const bad = [...html.matchAll(/data-at="(\d+)"/g)].map(m => +m[1]).filter(n => n >= oldN);
  if (bad.length) err(`${tag}: data-at ${[...new Set(bad)].join(',')} but the slide only has ${oldN} steps (0..${oldN - 1})`);
  // every step number (except 0) should reveal something
  for (let n = 1; !s.live && !s.keep && n < s.steps.length; n++) {
    const reveals = html.includes(`data-at="${n}"`) || html.includes(`data-cls="${n}:`);
    if (!reveals) warn(`${tag}: step ${n} ("${s.steps[n].cue}") reveals nothing`);
  }
  // images exist
  for (const m of html.matchAll(/src="(assets\/[^"]+)"/g)) {
    if (/\.(svg)$/.test(m[1])) continue;
    if (!fs.existsSync(path.join(ROOT, m[1]))) { if (/logos\/|video\//.test(m[1])) continue; err(`${tag}: missing file ${m[1]}`); }
  }
  if (process.argv.includes('--text')) console.log(`--- #${i} ${s.title}\n${strip(html)}\n`);
});

// URLs required by the brief
const required = [
  'https://arxiv.org/abs/1809.06537', 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272287', 'https://arxiv.org/abs/2207.08823',
  'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRE:2025:123.22.5PACTX.E1.BA/', 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRL:2025:87.25.3PGSXL.A.L1.5.B5/',
  'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRP:2025:1627.20.0T8PVZ.P1.7A/', 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRL:2025:236.23.6PBHRT.L1.9.58/',
  'https://arxiv.org/abs/2305.04388', 'https://doi.org/10.5281/zenodo.22116675',
  'https://webapi-estatisticas.justica.gov.pt/api/public/ContentMedias?id=bf5c32c8-79cb-406c-9e7c-33bdd01acb34&lang=PT',
  'https://webapi-estatisticas.justica.gov.pt/api/public/ContentMedias?id=ac5fa82e-73b1-4339-9443-e0809d501838&lang=PT',
  'https://commission.europa.eu/document/download/e5772fb0-6b70-4b1c-b54c-def1c42363f0_en?filename=Country%20chapter%20-%20Portugal%20-%20EN.pdf',
  'https://dciap.ministeriopublico.pt/sites/default/files/documentos/pdf/resumo_acusacoes_ges-bes_vf.pdf',
  'https://www.rtp.pt/noticias/pais/mais-de-100-lesados-ja-morreram-aespera-de-ver-feita-justica_n1607773',
  'https://economy-finance.ec.europa.eu/document/download/295e2168-d2a6-4e13-9496-a65e01a24bbb_en?filename=PT_SWD_2026_222_1_EN_autre_document_travail_service_part1_v3.pdf',
  'https://commission.europa.eu/document/download/5a482f87-1f24-47bd-8595-d25f1ca29c6a_en?filename=2025%20Rule%20of%20Law%20Report%20-%20Country%20Chapter%20Portugal.pdf',
  'http://ceur-ws.org/Vol-4089/paper2.pdf', 'https://arxiv.org/abs/2510.27675', 'https://arxiv.org/abs/2202.12837', 'https://arxiv.org/abs/1705.07874',
  'https://arxiv.org/abs/2606.18728', 'https://arxiv.org/abs/2508.17322',
  'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRG:2025:65.24.0GEBRG.G1.AE/'
];
['552', '572', '591', '410', '603', '607', '644'].forEach(n => required.push(`https://www.pgdlisboa.pt/leis/lei_mostra_articulado.php?artigo_id=1959A0${n}&nid=1959&tabela=lei_velhas&pagina=1&ficha=1&so_miolo=1&nversao=14`));
const haystack = JSON.stringify(DECK.slides.map(s => s.sources)) + allHtml.join('\n');
required.forEach(u => { if (!haystack.includes(u)) err('required URL missing from the deck: ' + u); });

// logos: the flag DECK.meta.logos.*.usable must match the real file content
function pngInfo(file) {
  // minimal PNG reader (8-bit, non-interlaced, RGB or RGBA): returns {w,h,hasAlpha,nonWhite,nonTransparent}
  const zlib = require('zlib');
  const d = fs.readFileSync(file); let i = 8, idat = [], w = 0, h = 0, ct = 0, bd = 0, il = 0;
  while (i < d.length) { const len = d.readUInt32BE(i), type = d.toString('ascii', i + 4, i + 8);
    if (type === 'IHDR') { w = d.readUInt32BE(i + 8); h = d.readUInt32BE(i + 12); bd = d[i + 16]; ct = d[i + 17]; il = d[i + 20]; }
    if (type === 'IDAT') idat.push(d.slice(i + 8, i + 8 + len)); i += 12 + len; }
  if (bd !== 8 || il !== 0 || (ct !== 2 && ct !== 6)) return null;
  const bpp = ct === 6 ? 4 : 3, raw = zlib.inflateSync(Buffer.concat(idat)), stride = w * bpp;
  const prev = Buffer.alloc(stride), cur = Buffer.alloc(stride); let nonWhite = 0, nonTransparent = 0;
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)]; raw.copy(cur, 0, y * (stride + 1) + 1, y * (stride + 1) + 1 + stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0, b = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
      if (f === 1) cur[x] = (cur[x] + a) & 255; else if (f === 2) cur[x] = (cur[x] + b) & 255; else if (f === 3) cur[x] = (cur[x] + ((a + b) >> 1)) & 255;
      else if (f === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); cur[x] = (cur[x] + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c)) & 255; }
    }
    for (let x = 0; x < w; x++) { const o = x * bpp; if (cur[o] < 250 || cur[o + 1] < 250 || cur[o + 2] < 250) nonWhite++; if (bpp === 4 && cur[o + 3] > 8) nonTransparent++; }
    cur.copy(prev);
  }
  return { w, h, hasAlpha: bpp === 4, nonWhite, nonTransparent };
}
const LOGOS = (DECK.meta && DECK.meta.logos) || {};
Object.keys(LOGOS).forEach(k => {
  const f = path.join(ROOT, LOGOS[k].src);
  if (!fs.existsSync(f)) { if (LOGOS[k].usable) err(`logo ${k}: file ${LOGOS[k].src} is missing but flagged usable`); return; }
  let info = null; try { info = pngInfo(f); } catch (e) {}
  if (!info) { warn(`logo ${k}: could not analyse ${LOGOS[k].src} (not an 8-bit RGB/RGBA PNG?)`); return; }
  const blank = info.hasAlpha ? info.nonTransparent === 0 : info.nonWhite === 0;
  const note = `${LOGOS[k].src}: ${info.w}x${info.h}, ${info.hasAlpha ? 'RGBA' : 'RGB'}, ${info.hasAlpha ? info.nonTransparent + ' visible px' : info.nonWhite + ' non-white px'}`;
  if (blank && LOGOS[k].usable) err(`logo ${k} is flagged usable but the file looks BLANK (${note})`);
  else if (blank) console.log(`  todo   logo ${k} file is BLANK (entirely white/transparent), placeholder shown: ${note}`);
  else if (!LOGOS[k].usable) warn(`logo ${k} has content (${note}); set usable:true in data/slides.js meta.logos`);
  else console.log(`  logo ${k} ok: ${note}`);
});

// optional files
['assets/video/app_demo.mp4'].forEach(f => {
  if (!fs.existsSync(path.join(ROOT, f)) && !fs.existsSync(path.join(ROOT, f.replace('.png', '.svg')))) console.log('  todo   still missing (placeholder shown): ' + f);
});

console.log(`\n${DECK.slides.length} slides, ${totalSteps} reveal states, ${errors} errors, ${warnings} warnings`);
process.exit(errors ? 1 : 0);
