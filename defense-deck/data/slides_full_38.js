/* ==========================================================================
   data/slides.js  -  ALL the content of the defense deck, in ONE file.

   HOW TO READ / EDIT THIS FILE (beginner notes)
   - window.DECK.slides is a list. Each slide is an object with:
       id        short unique name (used by "G" go-to and by the checker)
       number    the number printed in the corner (null = unnumbered)
       title     text of the maroon header band
       layout    which drawing recipe from js/layouts.js to use
       steps     one entry per reveal state. steps[0] is what the audience sees
                 on entry; every extra entry is one more key press.
                 {cue: "short label for the presenter", say: optional one-liner}
       notes     speaker notes (shown ONLY in the presenter window). Blank line = new paragraph.
       sources   links shown (clickable) in the presenter window.
       ...plus the content that the layout needs (rows, cards, html ...)
   - Inside html, an element with  data-at="2"  appears at step 2 (steps are 0-based
     here, so data-at="2" = the third state).  UI.at(2) writes that attribute for you.
   - Content = same text as the Canva deck "FINAL DEFENSE - DRAFT v1" (source of truth).
   - Diagram slides use layout "custom": boxes are placed with UI.box({x,y,w,h}, ...)
     on a 1920 x 840 area (the white band of the slide).
   ========================================================================== */
(function (root) {
  'use strict';
  var UI = root.UI, link = UI.link, at = UI.at, box = UI.box, svg = UI.svgLayer, arrow = UI.arrow, path = UI.path;

  /* ---- URLs used in several places ---------------------------------------- */
  var URL = {
    stats103: 'https://webapi-estatisticas.justica.gov.pt/api/public/ContentMedias?id=bf5c32c8-79cb-406c-9e7c-33bdd01acb34&lang=PT',
    stats106: 'https://webapi-estatisticas.justica.gov.pt/api/public/ContentMedias?id=ac5fa82e-73b1-4339-9443-e0809d501838&lang=PT',
    ec: 'https://commission.europa.eu/document/download/e5772fb0-6b70-4b1c-b54c-def1c42363f0_en?filename=Country%20chapter%20-%20Portugal%20-%20EN.pdf',
    sem2026: 'https://economy-finance.ec.europa.eu/document/download/295e2168-d2a6-4e13-9496-a65e01a24bbb_en?filename=PT_SWD_2026_222_1_EN_autre_document_travail_service_part1_v3.pdf',
    ec2025: 'https://commission.europa.eu/document/download/5a482f87-1f24-47bd-8595-d25f1ca29c6a_en?filename=2025%20Rule%20of%20Law%20Report%20-%20Country%20Chapter%20Portugal.pdf',
    dciap: 'https://dciap.ministeriopublico.pt/sites/default/files/documentos/pdf/resumo_acusacoes_ges-bes_vf.pdf',
    rtp: 'https://www.rtp.pt/noticias/pais/mais-de-100-lesados-ja-morreram-aespera-de-ver-feita-justica_n1607773',
    gemini: 'https://ai.google.dev/gemini-api/docs/tokens',
    openai: 'https://community.openai.com/t/rules-of-thumb-for-number-of-source-code-characters-to-tokens/622947',
    long: 'https://arxiv.org/abs/1809.06537',
    menezes: 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0272287',
    bertalan: 'https://arxiv.org/abs/2207.08823',
    alfieri: 'http://ceur-ws.org/Vol-4089/paper2.pdf',
    hannan: 'https://arxiv.org/abs/2510.27675',
    min: 'https://arxiv.org/abs/2202.12837',
    shap: 'https://arxiv.org/abs/1705.07874',
    shapCase: 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRG:2025:65.24.0GEBRG.G1.AE/',
    case236: 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRL:2025:236.23.6PBHRT.L1.9.58/',
    turpin: 'https://arxiv.org/abs/2305.04388',
    j123: 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRE:2025:123.22.5PACTX.E1.BA/',
    j87: 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRL:2025:87.25.3PGSXL.A.L1.5.B5/',
    j1627: 'https://jurisprudencia.csm.org.pt/ecli/ECLI:PT:TRP:2025:1627.20.0T8PVZ.P1.7A/',
    zenodo: 'https://doi.org/10.5281/zenodo.22116675',
    legalworld: 'https://arxiv.org/abs/2606.18728',
    simcourt: 'https://arxiv.org/abs/2508.17322'
  };
  function cpc(n) { return 'https://www.pgdlisboa.pt/leis/lei_mostra_articulado.php?artigo_id=1959A0' + n + '&nid=1959&tabela=lei_velhas&pagina=1&ficha=1&so_miolo=1&nversao=14'; }

  /** Image placed inside the 1920x840 content area. */
  function img(src, x, y, w, h, atN, cls, alt) {
    return box({ x: x, y: y, w: w, h: h, at: atN }, '<img class="fig ' + (cls || '') + '" src="assets/img/' + src + '" alt="' + (alt || '') + '" style="width:100%;height:100%;object-fit:contain;background:#fff">');
  }

  var slides = [];

  /* ======================= 0  COVER ======================= */
  slides.push({
    id: 'cover', number: null, title: 'Legal Judgement Prediction and Explanation using LLMs in Portugal', layout: 'cover',
    cover: {
      title: 'Legal Judgement Prediction and Explanation using LLMs in Portugal',
      by: 'Helton Mendonça &nbsp;·&nbsp; Nº 56870',
      supervisors: '<b>Supervisor:</b> Prof. Dr Francisco Couto<br><b>Co-Supervisor:</b> Judge Dr Sofia Wengorovius'
    },
    steps: [{ cue: 'Cover' }], notes: '', sources: []
  });

  /* ======================= 1  CONTEXT & MOTIVATION (system level: pending cases, then Europe's view) ======================= */
  slides.push({
    id: 'context-1', number: 1, title: 'Context &amp; Motivation', layout: 'custom',
    steps: [{ cue: 'Portugal\'s own data: pending cases' }, { cue: 'Europe\'s view, 2026' }, { cue: 'A year earlier: 2025 vs 2026' }],
    notes: [
      "Motivation = efficiency of the justice system in general; DV and BoC are only my scope.",
      "605,550 pending at 31 Dec 2025: +4.1% vs 2024 (Nº 106, provisional); excludes sentence-execution courts.",
      "Pending is not the same as delayed (DGPJ footnote); 2024 revised: 583,890 (Nº 103) vs about 581.6k implied.",
      "Thesis cites 583,890 (Nº 103 basis): visible in the left chart.",
      "Scope caveat: the 2026 deterioration is in Administrative and Tax Courts, not the courts I study: a system-wide trend.",
      "Civil/commercial 2026: clearance rate below 100%, backlog increased further (p.7); 263 days first instance (all litigious civil/commercial cases).",
      "2025 edition: some further progress at first instance; serious concerns in second instance (SWD(2025) 922, p.6)."
    ],
    sources: [
      { label: 'DGPJ Destaque anual Nº 103 (Oct 2025)', url: URL.stats103 },
      { label: 'DGPJ Destaque Nº 106 (Apr 2026)', url: URL.stats106 },
      { label: 'EU Rule of Law Report 2026, Portugal chapter (SWD(2026) 922 final, 17.7.2026)', url: URL.ec },
      { label: 'EU Rule of Law Report 2025, Portugal chapter (SWD(2025) 922, 8.7.2025)', url: URL.ec2025 }
    ],
    html: function () {
      function quote(y, h, txt, tagTxt) {
        return box({ x: 80, y: y, w: 1760, h: h, style: 'border-left:12px solid #c3542b;background:#f6e0d6;border-radius:0 24px 24px 0;padding:14px 30px;display:flex;flex-direction:column;justify-content:center;gap:8px' },
          '<div style="font-size:36px;font-style:italic;font-weight:600;color:#6c2118;line-height:1.28">&ldquo;' + txt + '&rdquo;</div>' +
          '<div style="font-size:24px;font-weight:700;color:#394c60">' + tagTxt + '</div>');
      }
      /* ---- step 0: pending cases (visible only at step 0) ---- */
      var s0 =
        img('ctx_pending_n103.png', 208, 14, 504, 432, 0, '', 'DGPJ Destaque Nº 103: pending cases 31 Dec 2024 and 30 Jun 2025') +
        box({ x: 80, y: 450, w: 760, h: 36, style: 'font-size:26px;font-weight:700;color:#394c60;text-align:center' }, 'DGPJ Destaque Nº 103, Oct 2025') +
        box({ x: 80, y: 490, w: 760, h: 50, cls: 'src', style: 'font-size:19px' }, link(URL.stats103, URL.stats103)) +
        img('ctx_pending_n106.png', 1000, 14, 348, 432, 0, '', 'DGPJ Destaque Nº 106: pending cases 2025') +
        box({ x: 890, y: 450, w: 670, h: 36, style: 'font-size:26px;font-weight:700;color:#394c60;text-align:center' }, 'DGPJ Destaque Nº 106, Apr 2026') +
        box({ x: 890, y: 490, w: 670, h: 50, cls: 'src', style: 'font-size:19px' }, link(URL.stats106, URL.stats106)) +
        box({ x: 1600, y: 14, w: 270, h: 526, cls: 'dbox center', style: 'font-size:28px;font-weight:700;color:#6c2118;text-align:center;border-radius:28px;background:#f6e0d6;border-color:#c3542b' },
          '<span style="font-size:2em;color:#c3542b">605,550</span><br>pending at<br>31 Dec 2025<br><span style="color:#394c60">+4.1% vs 2024</span><br><span style="font-size:.85em;font-weight:600;color:#394c60">(provisional)</span>') +
        box({ x: 80, y: 570, w: 1790, h: 80, style: 'font-size:24px;line-height:1.3;color:#5b5651' },
          'Excludes sentence-execution courts. 2024 was revised between Nº 103 (583,890) and Nº 106 (~581.6k implied: 605,550 − 23,937 saldo desfavorável); 2025 is provisional.');
      /* ---- step 1: Europe's view, 2026 (only at step 1) ---- */
      var s1 =
        box({ x: 80, y: 14, w: 1100, h: 50, style: 'font-size:30px;font-weight:700;color:#394c60' }, 'EU Rule of Law Report 2026 · Portugal') +
        box({ x: 1200, y: 14, w: 640, h: 50, style: 'text-align:right' }, '<span class="tag" style="font-size:26px;background:#394c60">system-wide trend</span>') +
        quote(84, 230, '(…) There has been no further progress to improve the efficiency of the justice system, in particular of Administrative and Tax Courts, as the efficiency of these courts has further deteriorated.', 'Rule of Law Report 2026, p.2') +
        quote(334, 230, '[…] it is recommended to Portugal to take further steps to improve the digitalisation of justice, including by improving the online publication and digital accessibility of judgments […]', 'Rule of Law Report 2026, p.2') +
        box({ x: 80, y: 600, w: 1790, h: 70, cls: 'src', style: 'font-size:20px' }, 'SWD(2026) 922 final, Brussels 17.7.2026 · ' + link(URL.ec, URL.ec));
      /* ---- step 2: a year earlier ---- */
      function card(x, hdr, col, tagTxt, tagCol, txt) {
        return box({ x: x, y: 40, w: 800, h: 430, cls: 'dbox', style: 'padding:24px 30px;border-color:' + col + ';background:#fff;border-radius:26px' },
          '<div style="font-size:30px;font-weight:700;color:' + col + '">' + hdr + '</div>' +
          '<div style="margin:14px 0"><span class="tag" style="font-size:28px;background:' + tagCol + '">' + tagTxt + '</span></div>' +
          '<div style="font-size:30px;font-style:italic;line-height:1.3;color:#26221f">&ldquo;' + txt + '&rdquo;</div>');
      }
      var s2 =
        card(80, '2025 edition · SWD(2025) 922 · p.6', '#2f8a4c', 'some further progress', '#2f8a4c', 'There has been some further progress in increasing the efficiency of Administrative and Tax Courts at first instance …') +
        card(1040, '2026 edition · SWD(2026) 922 · p.2', '#a8322a', 'no further progress', '#a8322a', 'There has been no further progress to improve the efficiency of the justice system, in particular of Administrative and Tax Courts, …') +
        svg(arrow(890, 255, 1030, 255, { w: 12 })) +
        box({ x: 80, y: 500, w: 1760, h: 90, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:34px' }, 'The trend reversed while pending cases rose.') +
        box({ x: 80, y: 612, w: 1790, h: 70, cls: 'src', style: 'font-size:20px' }, link(URL.ec2025, URL.ec2025) + '<br>' + link(URL.ec, URL.ec));
      return box({ x: 0, y: 0, w: 1920, h: 760, cl: '0-0:on', cls: 'only' }, s0) +
        box({ x: 0, y: 0, w: 1920, h: 760, at: 1, cl: '1-1:on', cls: 'only' }, s1) +
        box({ x: 0, y: 0, w: 1920, h: 760, at: 2, cl: '2-2:on', cls: 'only' }, s2);
    }
  });

  /* ======================= 2  CONTEXT & MOTIVATION (BES) ======================= */
  slides.push({
    id: 'context-2', number: 2, title: 'Context &amp; Motivation', layout: 'custom',
    steps: [{ cue: 'BES case' }, { cue: 'The facts' }, { cue: 'RTP headline + sources' }, { cue: 'It also hits companies' }],
    notes: [
      "GES/BES: people affected by a criminal case that is still running after more than ten years.",
      "Step 4: 'People to companies': stakeholder statements, not causal proof.",
      "Limits: qualitative stakeholder and survey statements; association, not proven causal effect.",
      "263 days refers to all civil/commercial litigious cases, not just company disputes.",
      "INE 2025 survey (Semester report p.11): firms ranked the judicial system the biggest obstacle to their operations.",
      "Older, notes only: Banco de Portugal staff article (Pereira & Wemans, ree201701) citing the 2015 INE survey.",
      "Do not use: Doing Business (discontinued); the Hotel Neptuno case (unverified)."
    ],
    sources: [
      { label: 'DCIAP: summary of the GES/BES indictments (PDF)', url: URL.dciap },
      { label: 'RTP: more than 100 victims have died waiting', url: URL.rtp },
      { label: 'EU Semester Country Report Portugal 2026 (SWD(2026) 222 final, 3 June 2026), printed p.11', url: URL.sem2026 },
      { label: 'EU Rule of Law Report 2026, Portugal (printed pp.7 and 17)', url: URL.ec },
      { label: 'Banco de Portugal: Pereira & Wemans, Produtividade na justiça cível (older, notes only)', url: 'https://www.bportugal.pt/sites/default/files/anexos/papers/ree201701_p.pdf' }
    ],
    html: function () {
      return '' +
        img('ctx_bes.jpg', 70, 30, 760, 416, 0, '', 'BES') +
        box({ x: 900, y: 30, w: 940, h: 130, at: 1, style: 'display:flex;align-items:center;gap:30px;background:#f6e0d6;border-radius:65px;padding:0 40px 0 14px' },
          '<div style="flex:none;min-width:250px;height:100px;border-radius:50px;background:#c3542b;color:#fff;font-weight:700;font-size:50px;display:flex;align-items:center;justify-content:center;padding:0 26px">300+</div><div style="font-size:31px;line-height:1.25">Crimes (e.g., Corruption, Money Laundering, etc.)</div>') +
        box({ x: 900, y: 176, w: 940, h: 110, at: 1, style: 'display:flex;align-items:center;gap:30px;background:#f6e0d6;border-radius:55px;padding:0 40px 0 14px' },
          '<div style="flex:none;min-width:250px;height:84px;border-radius:42px;background:#c3542b;color:#fff;font-weight:700;font-size:50px;display:flex;align-items:center;justify-content:center;padding:0 26px">20+</div><div style="font-size:31px">defendants</div>') +
        box({ x: 900, y: 302, w: 940, h: 144, at: 1, style: 'display:flex;align-items:center;gap:30px;background:#f6e0d6;border-radius:72px;padding:0 40px 0 14px' },
          '<div style="flex:none;min-width:250px;height:110px;border-radius:55px;background:#c3542b;color:#fff;font-weight:700;font-size:44px;display:flex;align-items:center;justify-content:center;padding:0 22px">2014 → 2024</div><div style="font-size:30px;line-height:1.25">Process began in 2014, however, the first court appearance was in 2024. <b>No end date in sight.</b></div>') +
        box({ x: 900, y: 462, w: 940, h: 60, at: 1, cls: 'src' }, link(URL.dciap, URL.dciap)) +
        img('ctx_rtp_lesados.png', 70, 470, 620, 349, 2, '', 'RTP headline: more than 100 victims have already died waiting for justice') +
        box({ x: 730, y: 640, w: 1110, h: 100, at: 2, cls: 'src' }, link(URL.rtp, URL.rtp)) +
        /* step 3: People -> Companies (visible only at step 3) */
        box({ x: 0, y: 0, w: 1920, h: 840, at: 3, cl: '3-3:on', cls: 'only', style: 'background:#fff' },
          box({ x: 80, y: 10, w: 620, h: 70, cls: 'dbox fill center', style: 'font-size:30px;font-weight:700;text-align:center;border-radius:35px;opacity:.85' }, 'PEOPLE · the GES/BES case') +
          svg(arrow(710, 45, 830, 45, { w: 8 })) +
          box({ x: 840, y: 10, w: 1000, h: 70, cls: 'dbox dark center', style: 'font-size:34px;font-weight:700;text-align:center;border-radius:35px' }, 'It also hits companies') +
          box({ x: 80, y: 100, w: 1760, h: 196, style: 'background:#fff;border:4px solid #c3542b;border-radius:24px;padding:16px 30px;display:flex;flex-direction:column;justify-content:center;gap:10px' },
          '<div style="font-size:35px;font-style:italic;font-weight:600;color:#26221f;line-height:1.28">&ldquo;Businesses flag inefficiencies in the judicial system as one of the main barriers to investment.&rdquo;</div>' +
          '<div><span class="tag" style="font-size:24px">EU Semester Country Report 2026</span> <span style="font-size:24px;font-weight:600;color:#394c60">SWD(2026) 222 final · p.11</span></div>') +
          box({ x: 80, y: 310, w: 1760, h: 196, style: 'background:#fff;border:4px solid #c3542b;border-radius:24px;padding:16px 30px;display:flex;flex-direction:column;justify-content:center;gap:10px' },
          '<div style="font-size:35px;font-style:italic;font-weight:600;color:#26221f;line-height:1.28">&ldquo;Stakeholders continue to flag efficiency shortcomings in the justice system, especially in Administrative and Tax Courts, as a long-term obstacle to investment and the business environment.&rdquo;</div>' +
          '<div><span class="tag" style="font-size:24px">Rule of Law Report 2026</span> <span style="font-size:24px;font-weight:600;color:#394c60">p.7</span></div>') +
          box({ x: 80, y: 520, w: 1760, h: 196, style: 'background:#fff;border:4px solid #c3542b;border-radius:24px;padding:16px 30px;display:flex;flex-direction:column;justify-content:center;gap:10px' },
          '<div style="font-size:35px;font-style:italic;font-weight:600;color:#26221f;line-height:1.28">&ldquo;45% of companies are very or fairly confident that investments are protected by law and courts&rdquo;</div>' +
          '<div><span class="tag" style="font-size:24px">Rule of Law Report 2026</span> <span style="font-size:24px;font-weight:600;color:#394c60">p.17 · source: 2026 EU Justice Scoreboard</span></div>') +
          box({ x: 80, y: 726, w: 1790, h: 90, cls: 'src', style: 'font-size:19px' }, link(URL.sem2026, URL.sem2026) + '<br>' + link(URL.ec, URL.ec)));
    }
  });

  /* ======================= 3  THREE RESEARCH QUESTIONS ======================= */
  slides.push({
    id: 'rqs', number: 3, title: 'Three research questions', layout: 'pillRows', tone: 'orange', rowH: 190, badgeW: 200, fs: 35,
    rows: [
      { badge: 'RQ1', html: 'Can LLMs outperform the LegalBERTimbau DL baseline in predicting outcomes for both types of cases?' },
      { badge: 'RQ2', html: 'Beyond prediction, can the models explain how they reached their decisions, and do these explanations align with legal reasoning?', at: 1 },
      { badge: 'RQ3', html: 'How does performance and explainability differ between a sensitive/human-centered domain like domestic violence and a documentary domain like breach of contract?', at: 2 }
    ],
    steps: [{ cue: 'RQ1' }, { cue: 'RQ2' }, { cue: 'RQ3' }], notes: '', sources: []
  });

  /* ======================= 4  THREE OBJECTIVES ======================= */
  slides.push({
    id: 'objectives', number: 4, title: 'Three objectives', layout: 'pillRows', tone: 'maroon', rowH: 190, badgeW: 200, fs: 36,
    rows: [
      { badge: 'O1', html: 'Establish a Portugal-specific Legal Judgement Prediction (LJP) baseline (the first to best of my knowledge).' },
      { badge: 'O2', html: 'Experiment &amp; evaluate DL/LLM techniques, recommend next steps.', at: 1 },
      { badge: 'O3', html: 'Explore XAI, validated with a legal expert.', at: 2 }
    ],
    steps: [{ cue: 'O1' }, { cue: 'O2' }, { cue: 'O3' }], notes: '', sources: []
  });

  /* ======================= 5  RESEARCH GAP & CONTRIBUTIONS ======================= */
  slides.push({
    id: 'gap', number: 5, title: 'Research Gap &amp; Contributions', layout: 'custom',
    steps: [{ cue: 'Current literature' }, { cue: 'Gap: Explainability + Legal Expert Validation' }, { cue: 'Gap: Portugal Specific + Domain Comparisons' }],
    notes: '', sources: [],
    html: function () {
      function lit(y, t) { return box({ x: 100, y: y, w: 640, h: 150, cls: 'dbox fill center', style: 'font-size:42px;font-weight:700;text-align:center;border-radius:75px;color:#394c60;border-color:#394c60;background:#e5eaf0' }, t); }
      function gap(x, y, t, a) { return box({ x: x, y: y, w: 420, h: 220, at: a, cls: 'dbox salmon center', style: 'font-size:42px;font-weight:700;text-align:center;color:#6c2118;border-radius:36px' }, t); }
      return '' +
        box({ x: 80, y: 40, w: 680, h: 90, cls: 'dbox dark center', style: 'font-size:44px;font-weight:700;text-align:center;border-radius:45px;background:#394c60;border-color:#394c60' }, 'Current literature') +
        lit(170, 'Data') + lit(350, 'ML/DL/LLMs') + lit(530, 'Performance Metrics') +
        svg(arrow(790, 440, 960, 440, { at: 1, w: 8 })) +
        box({ x: 980, y: 40, w: 860, h: 90, at: 1, cls: 'dbox dark center', style: 'font-size:44px;font-weight:700;text-align:center;border-radius:45px;background:#c3542b;border-color:#c3542b' }, 'Existing Gap') +
        gap(980, 170, 'Explainability', 1) + gap(1420, 170, 'Legal Expert Validation', 1) +
        gap(980, 420, 'Portugal Specific', 2) + gap(1420, 420, 'Domain Comparisons', 2);
    }
  });

  /* ======================= 6  DATA ======================= */
  slides.push({
    id: 'data', number: 6, title: 'Data', layout: 'custom',
    steps: [{ cue: 'Sources, pros and cons' }, { cue: 'Two domains, various courts' }, { cue: 'Easy vs hard verdict extraction' }],
    notes: '', sources: [
      { label: 'dgsi.pt', url: 'https://www.dgsi.pt/' },
      { label: 'jurisprudencia.csm.org.pt', url: 'https://jurisprudencia.csm.org.pt/' }
    ],
    html: function () {
      return '' +
        box({ x: 80, y: 20, w: 640, h: 190, cls: 'dbox dark center', style: 'font-size:32px;font-weight:600' }, '<h4 style="font-size:34px">Data Sources</h4>dgsi.pt &amp; jurisprudencia.csm.org.pt') +
        box({ x: 750, y: 20, w: 520, h: 190, cls: 'dbox out center', style: 'font-size:29px' }, '<h4 style="font-size:32px">Pros</h4>Public, pseudonymized &amp; managed by trusted entities') +
        box({ x: 1300, y: 20, w: 540, h: 190, cls: 'dbox flag center', style: 'font-size:29px;border-style:solid' }, '<h4 style="font-size:32px">Cons</h4>Low volume &amp; quality, lack of documentation &amp; communication') +
        box({ x: 80, y: 235, w: 880, h: 120, at: 1, cls: 'dbox fill center', style: 'font-size:33px' }, '<h4 style="font-size:33px">Two Domains</h4>Domestic Violence &amp; Breach of Contract') +
        box({ x: 990, y: 235, w: 850, h: 120, at: 1, cls: 'dbox fill center', style: 'font-size:33px' }, '<h4 style="font-size:33px">Various Courts</h4>Second Instances, Supreme Court &amp; Peace Courts') +
        /* easy: structured document (mirrors manuscript figure "Structured vs Prose") */
        box({ x: 80, y: 380, w: 880, h: 440, at: 2, cls: 'dbox out', style: 'font-size:23px;padding:16px 26px' },
          '<h4 style="font-size:30px;margin-bottom:10px">Easy to extract verdict</h4>' +
          '<b>I. Relatório</b> <i>(...) foi decidido (no que ora releva): (...)</i><br><span class="src" style="font-size:19px">[judges contextualize the current case]</span><br>' +
          '<b>II. Fundamentação</b> <i>1 - Delimitação do objeto do recurso. (...) 2 - A decisão recorrida. (...)</i><br><span class="src" style="font-size:19px">[the issues judged; proven facts; legal analysis]</span><br>' +
          '<b>III. Dispositivo</b><br><span style="background:#fff3a0;border-radius:6px;padding:2px 6px;font-weight:700;font-style:italic">Nos termos expostos, nega-se provimento ao recurso interposto pelo arguido, mantendo-se, consequentemente, o decidido no acórdão recorrido. (...)</span><br><span class="src" style="font-size:19px">[final decision]</span>') +
        box({ x: 990, y: 380, w: 850, h: 440, at: 2, cls: 'dbox flag', style: 'font-size:23px;padding:16px 26px;border-style:solid' },
          '<h4 style="font-size:30px;margin-bottom:10px">Hard to extract verdict</h4>' +
          '<i>Acordam os Juízes da Secção Criminal do Tribunal da Relação de Évora em apreciar o recurso interposto pelo arguido (...) </i><br><span class="src" style="font-size:19px">[no section titles]</span><br>' +
          '<i>O recorrente insurge-se contra a qualificação jurídica dos factos (...) Da análise conjunta da prova produzida em julgamento resulta que (...)</i><br><span class="src" style="font-size:19px">[facts and legal analysis run together in prose]</span><br>' +
          '<span style="background:#fff3a0;border-radius:6px;padding:2px 6px;font-weight:700;font-style:italic">Nos termos expostos, nega-se provimento ao recurso (...)</span><br><span class="src" style="font-size:19px">[only cue: the decision sits at the bottom, after a formal connector such as &ldquo;Pelo exposto...&rdquo;]</span>');
    }
  });

  /* ======================= 7  INITIAL PREPROCESSING ======================= */
  slides.push({
    id: 'preprocessing', number: 7, title: 'Initial Preprocessing', layout: 'custom',
    steps: [{ cue: 'Raw documents' }, { cue: '1. Scraping and verdict extraction' }, { cue: '2. Cleaning and filtering' }, { cue: '3. Target class engineering' }],
    notes: '',
    sources: [],
    html: function () {
      var X = [80, 680, 1280], W = 560;
      function head(i, t, a) { return box({ x: X[i], y: 150, w: W, h: 76, at: a, cls: 'dbox dark center', style: 'font-size:31px;font-weight:700;border-radius:38px' }, t); }
      return '' +
        box({ x: 80, y: 20, w: 1760, h: 96, cls: 'dbox fill center', style: 'font-size:34px;text-align:center;border-color:#394c60' },
          '<b>Raw Documents (JSON): 6,627 total</b> &nbsp;(4,368 BoC / 2,259 DV)') +
        svg(arrow(360, 116, 360, 146, { w: 5 }) +
          arrow(X[0] + W + 4, 188, X[1] - 6, 188, { at: 2, w: 5 }) + arrow(X[1] + W + 4, 188, X[2] - 6, 188, { at: 3, w: 5 })) +
        head(0, '1. Data Scraping', 1) + head(1, '2. Cleaning &amp; Filtering', 2) + head(2, '3. Target Class Engineering', 3) +
        /* column 1 */
        box({ x: X[0], y: 246, w: W, h: 100, at: 1, cls: 'dbox s24' }, '<b>Sources:</b> dgsi.pt &amp; jurisprudencia.csm.org.pt') +
        box({ x: X[0], y: 360, w: W, h: 130, at: 1, cls: 'dbox fill s24' }, '<h4>Verdict Extraction</h4>using regex rules on final 3rd of each document') +
        box({ x: X[0], y: 504, w: W, h: 80, at: 1, cls: 'dbox fill s24 center' }, '<h4 style="margin:0">Metadata Extraction</h4>') +
        box({ x: X[0], y: 600, w: W, h: 110, at: 1, cls: 'dbox flag center s24' }, '<b>Failed on ~5%</b> of each domain documents') +
        /* column 2 */
        box({ x: X[1], y: 246, w: W, h: 230, at: 2, cls: 'dbox s24' }, '<h4>Drop</h4><ul><li>cases failed decision extraction</li><li>case with no full text (usually older cases)</li><li>duplicate cases, by case id</li></ul>') +
        box({ x: X[1], y: 496, w: W, h: 170, at: 2, cls: 'dbox salmon center', style: 'font-size:30px' }, '<b>BoC: 1,938</b> (55.6% drop)<br><b>DV: 1,204</b> (46.7% drop)') +
        /* column 3 */
        box({ x: X[2], y: 246, w: W, h: 230, at: 3, cls: 'dbox s24' }, '<h4>Heuristic cues</h4><i>&ldquo;parcialmente provido&rdquo;, &ldquo;procedente em parte&rdquo;, &ldquo;provido&rdquo;, &ldquo;procedente&rdquo;, &ldquo;revogado&rdquo;, &ldquo;improcedente&rdquo;, &ldquo;mantido&rdquo;</i>') +
        box({ x: X[2], y: 496, w: W, h: 250, at: 3, cls: 'dbox out', style: 'font-size:27px' },
          '<h4>Post-Class Engineering Data</h4><b>BoC: 1,802</b> (FAVORABLE / UNFAVORABLE / PARTIAL)<br><b>DV: 1,126</b> (KEPT / ALTERED) <i>— Judge Sofia\'s correction</i>');
    }
  });

  /* ======================= 8  METHODOLOGY ======================= */
  slides.push({
    id: 'methodology', number: 8, title: 'Methodology Trimmed, Not Abandoned', layout: 'custom',
    steps: [{ cue: 'The full planned design' }, { cue: 'Secondary branch and ML (SVM + TF-IDF) removed' }],
    notes: '',
    sources: [],
    html: function () {
      var X0 = 80, X1 = 1040, W = 800;
      /* method box; "cl" = when it turns grey ("removed"); a REMOVED tag appears at the same step */
      function m(x, y, h, cls, cl, gone, t) {
        return box({ x: x, y: y, w: W, h: h, cls: 'dbox ' + cls, cl: cl, style: 'font-size:27px;position:absolute' },
          '<div>' + t + '</div>' + (gone ? '<span class="rm-tag" data-at="' + gone + '">REMOVED</span>' : ''));
      }
      return '' +
        box({ x: 80, y: 14, w: 1760, h: 96, cls: 'dbox brown center', style: 'font-size:31px;text-align:center' },
          '<b>Temporal based split</b><br>(hold-out by decision date; 50 newest cases per domain for testing)') +
        box({ x: 660, y: 128, w: 600, h: 90, cls: 'dbox brown center', style: 'font-size:30px;text-align:center' }, '<b>Training set</b><br>Preprocessed Documents') +
        /* legend (appears when something is removed) */
        box({ x: 1330, y: 142, w: 240, h: 58, at: 1, cls: 'dbox center', style: 'font-size:25px;font-weight:700;text-align:center;border-radius:29px' }, 'Active') +
        box({ x: 1590, y: 142, w: 250, h: 58, at: 1, cls: 'dbox grey center', style: 'font-size:25px;font-weight:700;text-align:center;border-radius:29px;border-style:dashed' }, 'Removed') +
        svg(arrow(960, 110, 960, 126, { w: 5 }) +
          path([[960, 218], [960, 256], [480, 256], [480, 288]], { w: 5 }) +
          path([[960, 218], [960, 256], [1440, 256], [1440, 288]], { w: 5, color: '#c3542b' }) +
          path([[960, 218], [960, 256], [1440, 256], [1440, 288]], { at: 1, w: 5, grey: true })) +
        box({ x: X0, y: 296, w: W, h: 62, cls: 'dbox dark center', style: 'font-size:32px;font-weight:700;text-align:center;border-radius:31px' }, 'Main Branch') +
        box({ x: X1, y: 296, w: W, h: 62, cls: 'dbox dark center', cl: '1:gone', style: 'font-size:32px;font-weight:700;text-align:center;border-radius:31px;background:#c3542b;border-color:#c3542b' }, 'Secondary Branch') +
        m(X0, 374, 136, '', '1:gone', 1, '<h4>ML</h4>SVM + TF-IDF') +
        m(X0, 522, 90, '', null, 0, '<h4>DL</h4>Legal BERTimbau') +
        m(X0, 624, 190, '', null, 0, '<h4>LLMs</h4>DeepSeek-R1-0528-Qwen3-8B<br>Zero-shot + Few-shot<br>(native CoT; RAG not implemented)') +
        m(X1, 374, 136, 'salmon', '1:gone', 1, '<h4>ML</h4>LR / NB / etc. + TF–IDF<br>SVM / LR / NB + Legal BERTimbau embeddings') +
        m(X1, 522, 90, 'salmon', '1:gone', 1, '<h4>DL</h4>BERTimbau / Legal BERT') +
        m(X1, 624, 190, 'salmon', '1:gone', 1, '<h4>LLMs</h4>Llama-3.1-8B-Instruct / Ministral-3-8B-Instruct-2512<br>(same prompts) + optional fine-tuning');
    }
  });

  /* ======================= 9  DOCUMENT LENGTH ======================= */
  slides.push({
    id: 'doc-length', number: 9, title: 'Document length was a major bottleneck', layout: 'custom',
    steps: [{ cue: 'DV numbers' }, { cue: 'BoC numbers + rule of thumb' }],
    notes: '',
    sources: [{ label: 'Gemini API docs: tokens', url: URL.gemini }, { label: 'OpenAI community: rules of thumb for characters to tokens', url: URL.openai }],
    html: function () {
      function tile(x, a, tag, chars, toks, barW) {
        return box({ x: x, y: 30, w: 840, h: 470, at: a, cls: 'card', style: 'padding:30px 44px' },
          '<h3 style="font-size:40px;border:0;margin:0 0 4px">' + tag + '</h3>' +
          '<div style="font-size:60px;font-weight:700;color:#6c2118;line-height:1.1">' + chars + '</div><div style="font-size:30px;margin-bottom:14px">characters avg</div>' +
          '<div style="font-size:60px;font-weight:700;color:#c3542b;line-height:1.1">' + toks + '</div><div style="font-size:30px;margin-bottom:22px">tokens avg</div>' +
          '<div style="height:34px;border-radius:17px;background:#c3542b;width:' + barW + 'px"></div>');
      }
      return '' +
        tile(80, 0, 'DV', '~82,500', '~20,625', 750) +
        tile(1000, 1, 'BoC', '~43,200', '~10,800', 393) +
        box({ x: 80, y: 540, w: 1760, h: 270, at: 1, style: 'font-size:28px;line-height:1.4;color:#394c60' },
          'assume 1 token ~ 4 characters according to leading labs rule of thumb:<br><span class="src" style="font-size:24px">' +
          link(URL.gemini, URL.gemini) + '<br>' + link(URL.openai, URL.openai) + '</span>');
    }
  });

  /* ======================= 10  BERT: SLIDING WINDOW ======================= */
  slides.push({
    id: 'bert-window', number: 10, title: 'BERT: Sliding Window', layout: 'custom',
    steps: [{ cue: 'Judgment text and overlapping windows' }, { cue: 'Keep 16 windows' }, { cue: 'LegalBERTimbau, mean pooling, classifier' }, { cue: 'Limitation' }],
    notes: '', sources: [],
    html: function () {
      var BX = function (i) { return 80 + i * 298; };
      function st(i, a, cls, t) { return box({ x: BX(i), y: 462, w: 270, h: 200, at: a, cls: 'dbox ' + cls + ' s22', style: 'font-size:22.5px' }, t); }
      function win(x, y, w, a, label) {
        return box({ x: x, y: y, w: w, h: 52, at: a, cls: 'win', style: 'border-radius:10px' }, '') +
          box({ x: x + w + 16, y: y + 8, w: 1000, h: 40, at: a, style: 'font-size:25px;color:#394c60;white-space:nowrap' }, label);
      }
      return '' +
        box({ x: 80, y: 6, w: 1760, h: 54, style: 'font-size:32px;font-weight:700;color:#394c60' }, 'Documents exceed BERT\'s 512-token limit → slide overlapping windows over the text') +
        box({ x: 140, y: 76, w: 1640, h: 36, style: 'font-size:26px;font-weight:700;color:#6c2118' }, 'Document = L tokens (≈ 10.8k BoC · 20.6k DV on average)') +
        box({ x: 140, y: 116, w: 1640, h: 54, cls: 'docbar' }, '') +
        win(140, 190, 330, 0, 'Window 1 · 512 tokens') +
        win(388, 256, 330, 0, 'Window 2 starts 384 tokens later (128-token overlap)') +
        win(636, 322, 330, 0, 'Window 3') +
        box({ x: 884, y: 388, w: 800, h: 40, style: 'font-size:25px;color:#394c60;font-weight:600' }, '… up to N windows (16 kept)') +
        st(0, 0, 'fill', '<h4>Judgment text</h4>Relatório + Fundamentação (Dispositivo removed)') +
        st(1, 0, '', '<h4>Sliding windows</h4>512 tokens per window, stride 128 → N windows') +
        st(2, 1, '', '<h4>Select windows</h4>If N &gt; 16, keep 16 equally spaced windows') +
        st(3, 2, 'salmon', '<h4>LegalBERTimbau</h4>Each window → one [CLS] vector') +
        st(4, 2, 'salmon', '<h4>Mean pooling</h4>Window vectors averaged into one document vector') +
        st(5, 2, 'out', '<h4>Classifier head</h4>Linear layer → outcome class (2 for DV, 3 for BoC)') +
        svg(arrow(BX(0) + 272, 562, BX(1) - 2, 562, { w: 4 }) + arrow(BX(1) + 272, 562, BX(2) - 2, 562, { at: 1, w: 4 }) +
          arrow(BX(2) + 272, 562, BX(3) - 2, 562, { at: 2, w: 4 }) + arrow(BX(3) + 272, 562, BX(4) - 2, 562, { at: 2, w: 4 }) +
          arrow(BX(4) + 272, 562, BX(5) - 2, 562, { at: 2, w: 4 })) +
        box({ x: 80, y: 680, w: 1760, h: 46, at: 1, style: 'font-size:29px;font-weight:700;color:#6c2118;text-align:center' }, '16 windows cover ≈ 6,272 unique tokens · ~4–6 h per training epoch on local hardware.') +
        box({ x: 80, y: 736, w: 1760, h: 92, at: 3, cls: 'limit-band', style: 'font-size:26px;padding:12px 30px;display:flex;align-items:center' },
          '<div><b>Limitation:</b> information loss — the other windows are discarded, and mean pooling weighs every window equally, ignoring position and document structure.</div>');
    }
  });

  /* ======================= 11  LLMs: SMART SUMMARIZATION ======================= */
  slides.push({
    id: 'smart-summarization', number: 11, title: 'LLMs: Smart Summarization', layout: 'custom',
    steps: [{ cue: 'Split, encode, score' }, { cue: 'Zoom: the three scoring components' }, { cue: 'Select and reorder' }, { cue: 'Compact summary + numbers' }, { cue: 'Limitations' }],
    notes: [
      "Pipeline: split, score, select, reorder, compact summary.",
      "Zoom: score = 0.52 query + 0.26 anchors + 0.22 centroid.",
      "Query: 5 Portuguese sentences on why courts keep or change a decision.",
      "Anchors: phrases courts use for facts, norms, proof, reasoning.",
      "Say out loud: hand-engineered with AI help; not validated by legal experts.",
      "About 5,400 tokens on average; 9,000 cap never reached.",
      "A local engineering solution, not a scientific contribution."
    ],
    sources: [],
    html: function () {
      var BX = function (i) { return 80 + i * 298; };
      function st(i, a, cls, t) { return box({ x: BX(i), y: 130, w: 270, h: 230, at: a, cls: 'dbox ' + cls + ' s22', style: 'font-size:23px' }, t); }
      function tile(i, big, small) {
        return box({ x: 80 + i * 600, y: 410, w: 560, h: 190, at: 3, cls: 'tile', style: 'display:flex;flex-direction:column;justify-content:center' },
          '<div class="big">' + big + '</div><div class="small">' + small + '</div>');
      }
      /* real anchor phrases from src/modeling/llms/summarizer.py (3 per category, Portuguese as in the code) */
      var anchors = [
        ['facts', 'factos provados · apurou-se que · resulta dos autos'],
        ['norms', 'nos termos do artigo · ao abrigo do · à luz do'],
        ['proof', 'prova testemunhal · convicção do tribunal · não se provou'],
        ['reasoning', 'entende o tribunal · conclui-se que · cumpre decidir'],
        ['case law', 'jurisprudência · acórdão do · em casos semelhantes'],
        ['conflict', 'não assiste razão · carece de fundamento · não se verifica']
      ];
      var arows = anchors.map(function (a) {
        return '<div style="display:grid;grid-template-columns:150px 1fr;column-gap:12px;align-items:baseline;margin:0 0 9px"><span class="tag" style="font-size:22px;text-align:center;background:#c3542b">' + a[0] + '</span><i style="font-size:25px">' + a[1] + '</i></div>';
      }).join('');
      function col(x, w, wt, title, body) {
        return box({ x: x, y: 190, w: w, h: 560, cls: 'card', style: 'padding:20px 26px' },
          '<h3 style="font-size:27px;margin-bottom:12px">' + title + ' <span style="color:#c3542b">' + wt + '</span></h3><div class="card-body" style="font-size:25px;line-height:1.3">' + body + '</div>');
      }
      return '' +
        box({ x: 80, y: 14, w: 1760, h: 90, cls: 'caption top', style: 'display:flex;align-items:center;justify-content:center;font-size:32px' },
          'Enlarging the window hit memory limits without curing dilution → select the relevant sentences') +
        st(0, 0, 'fill', '<h4>Judgment text</h4>Relatório + Fundamentação') +
        st(1, 0, '', '<h4>Split &amp; encode</h4>Sentences encoded with frozen LegalBERTimbau (mean pooling)') +
        st(2, 0, '', '<h4>Score sentences</h4>0.52 query similarity + 0.26 legal-anchor patterns + 0.22 centroid similarity') +
        st(3, 2, 'salmon', '<h4>Select</h4>Highest-ranked first, within a 9,000-token budget + anchor guardrails') +
        st(4, 2, 'salmon', '<h4>Reorder</h4>Back to original position, keeping the court\'s narrative order') +
        st(5, 3, 'out', '<h4>Compact summary</h4>Fed into zero-shot and few-shot prompts') +
        svg(arrow(BX(0) + 272, 245, BX(1) - 2, 245, { w: 4 }) + arrow(BX(1) + 272, 245, BX(2) - 2, 245, { w: 4 }) +
          arrow(BX(2) + 272, 245, BX(3) - 2, 245, { at: 2, w: 4 }) + arrow(BX(3) + 272, 245, BX(4) - 2, 245, { at: 2, w: 4 }) +
          arrow(BX(4) + 272, 245, BX(5) - 2, 245, { at: 3, w: 4 })) +
        tile(0, '≈ 5,400', 'tokens on average after summarization (DV and BoC)') +
        tile(1, '9,000', 'token cap — no test case reached it') +
        tile(2, '46.8% · 51.2%', 'average reduction rate, DV · BoC') +
        box({ x: 80, y: 640, w: 1760, h: 180, at: 4, cls: 'limit-band', style: 'font-size:29px;display:flex;align-items:center' },
          '<div><b>Limitations:</b> information loss (not quantified per case) and many manual engineering choices — templates, anchor keywords, weights — never validated by judges or peers.<br><b>Hence a local engineering solution, not a scientific contribution.</b></div>') +
        /* ZOOM on "Score sentences": shown only at step 1 (data-cls range), content present from step 1 on */
        box({ x: 40, y: 112, w: 1840, h: 712, at: 1, cl: '1-1:zoom-on', cls: 'zoom-ov' },
          box({ x: 0, y: 0, w: 1826, h: 70, cls: 'dbox dark center', style: 'font-size:32px;font-weight:700;text-align:center;border-radius:35px;position:absolute' },
            'Score sentences = 0.52 query + 0.26 anchors + 0.22 centroid') +
          col(0, 560, '0.52', 'QUERY SIMILARITY',
            'The query = 5 short Portuguese sentences on <i>why a court keeps or changes a decision</i>, averaged into one vector.<br><br><i style="color:#6c2118">&ldquo;fundamentos jurídicos para manter ou alterar a decisão recorrida&rdquo;</i><br><br><i style="color:#6c2118">&ldquo;razões do tribunal de recurso para confirmar ou revogar a decisão&rdquo;</i>').replace('top:190px', 'top:90px') +
          col(580, 860, '0.26', 'LEGAL-ANCHOR PATTERNS', 'Phrases courts use to state facts, norms, proof and reasoning (examples):<br><br>' + arows).replace('top:190px', 'top:90px') +
          col(1450, 370, '0.22', 'CENTROID SIMILARITY', 'How close a sentence is to the <b>average embedding of the whole document</b>: keeps the case\'s overall context.').replace('top:190px', 'top:90px') +
          box({ x: 0, y: 654, w: 1840, h: 52, style: 'position:absolute;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700;color:#a8322a;gap:16px' },
            '<span class="tag" style="background:#a8322a;font-size:22px">hand-tuned</span><span class="tag" style="background:#a8322a;font-size:22px">not expert-validated</span><span style="color:#26221f;font-weight:600">a pragmatic engineering choice, made with AI help</span>'));
    }
  });

  /* ======================= 12  EVALUATION ======================= */
  slides.push({
    id: 'evaluation', number: 12, title: 'Evaluation: Future Cases Only', layout: 'custom',
    steps: [{ cue: 'Cutoff and counts' }, { cue: 'BERT: 85/15 temporal split' }, { cue: 'LLM few-shot examples' }, { cue: 'Nothing leaks from the future' }, { cue: 'Gold test set' }, { cue: 'Metrics and checks' }],
    notes: '', sources: [],
    html: function () {
      function ln(a, t, extra) { return '<p data-at="' + a + '" style="margin:0 0 14px;' + (extra || '') + '">' + t + '</p>'; }
      function card(x, a, t, body) { return box({ x: x, y: 175, w: 568, h: 640, at: a, cls: 'card' }, '<h3>' + t + '</h3><div class="card-body" style="font-size:31px">' + body + '</div>'); }
      return '' +
        /* timeline, mirrors the temporal-split logic: older -> newer */
        box({ x: 80, y: 30, w: 1000, h: 90, at: 0, cls: 'dbox fill center', style: 'font-size:29px;font-weight:700;text-align:center;border-radius:45px 0 0 45px;border-right:0' }, 'OLDER CASES &nbsp;·&nbsp; training + few-shot examples') +
        box({ x: 1080, y: 30, w: 760, h: 90, at: 3, cls: 'dbox dark center', style: 'font-size:29px;font-weight:700;text-align:center;border-radius:0 45px 45px 0' }, 'NEWER CASES &nbsp;·&nbsp; gold test set') +
        svg(path([[1080, 14], [1080, 136]], { w: 5, noHead: true, dash: '10 8', color: '#c3542b', at: 3 })) +
        /* TEMPORAL SPLIT card: short lines, revealed one by one */
        box({ x: 80, y: 175, w: 568, h: 650, cls: 'card' }, '<h3>TEMPORAL SPLIT</h3><div class="card-body" style="font-size:29px">' +
          ln(0, 'Everything <b>before the cutoff</b> goes into training.') +
          ln(0, '<b>1,064 DV · 1,752 BoC</b> cases.') +
          ln(1, 'BERT fine-tuning: <b>85/15 temporal split</b>.') +
          ln(2, 'Also the pool of the LLM\'s <b>few-shot examples</b>.') +
          ln(3, '<b>Nothing from the future leaks into training.</b>', 'color:#6c2118') +
          ln(3, 'Every gold-test case is newer than every training/example case.', 'color:#6c2118') + '</div>') +
        card(676, 4, 'GOLD TEST SET', '50 newest cases per domain, frozen before any model work.<br><br><b>DV</b> (from 2025-05-01): 29 KEPT · 21 ALTERED<br><br><b>BoC</b> (from 2023-10-01): 35 UNFAVORABLE · 9 PARTIAL · 6 FAVORABLE') +
        card(1272, 5, 'METRICS &amp; CHECKS', 'MCC + Macro-F1, 95% stratified bootstrap CIs (1,000 resamples).<br><br>Label check: 5 judges × 5 gold cases per domain.');
    }
  });

  /* ======================= 13  HEADLINE RESULTS ======================= */
  slides.push({
    id: 'headline-results', number: 13, title: 'Headline Results', layout: 'results',
    cols: '510px 160px 210px 200px 150px 1fr',
    head: ['Model', 'Domain', 'Mode', 'Macro-F1', 'MCC', '95% CI'],
    rows: [
      { domain: 'DV', cells: ['Fine-tuned LegalBERTimbau', 'DV', 'Supervised', '0.367', '0.000', '[0.000, 0.000]'] },
      { domain: 'DV', cells: ['DeepSeek-R1-8B', 'DV', 'Zero-shot', { t: '0.783', cl: '2:bi' }, { t: '0.592', cl: '2:bi' }, '[0.372, 0.797]'] },
      { domain: 'DV', cells: ['DeepSeek-R1-8B', 'DV', 'Few-shot', '0.696', '0.486', '[0.257, 0.681]'] },
      { domain: 'BoC', at: 1, cells: ['Fine-tuned LegalBERTimbau', 'BoC', 'Supervised', '0.275', '0.000', '[0.000, 0.000]'] },
      { domain: 'BoC', at: 1, cells: ['DeepSeek-R1-8B', 'BoC', 'Zero-shot', { t: '0.663', cl: '2:b' }, '0.468', '[0.243, 0.690]'] },
      { domain: 'BoC', at: 1, cells: ['DeepSeek-R1-8B', 'BoC', 'Few-shot', '0.631', { t: '0.523', cl: '2:b' }, '[0.314, 0.736]'] }
    ],
    caption: { at: 2, html: 'Light rows: DV · Darker rows: BoC · <b>Bold: best in each domain</b> · <b><i>Bold italic: best overall</i></b>' },
    steps: [{ cue: 'Domestic violence rows' }, { cue: 'Breach of contract rows' }, { cue: 'Best results emphasised' }],
    notes: '', sources: []
  });

  /* ======================= 14  GENERALIZATION? ======================= */
  slides.push({
    id: 'generalization', number: 14, title: 'Generalization?', layout: 'cards', cols: [
      { title: 'THE TEST IS AS FAIR AS THIS METHOD ALLOWS', html: '<p>Every test case is <b>newer</b> than anything the models learned from.</p><p>Cutoffs: <b>DV 2025-05-01</b> · <b>BoC 2023-10-01</b>.</p><p><i>Caveat:</i> retrospective classification of appeals already written. A real system would act before the final ruling exists.</p>' },
      { title: 'WHAT IT SHOWS', at: 1, html: '<p><b>BERT does not generalise:</b> DV MCC 0.000 on train, validation and test; BoC fades from 0.303 to 0.184 to 0.000.</p><p><b>DeepSeek zero-shot</b> shows some generalisation to future cases (MCC 0.592 DV / 0.468 BoC), up to a point and with wide intervals.</p>' }
    ],
    columns: '1fr 1fr', cardsH: 470, fs: 33,
    caption: { at: 2, html: 'Under the protocol tested, DeepSeek generalised better than BERT, to future cases of the same courts and domains only; nothing beyond that is claimed.' },
    steps: [{ cue: 'How fair the test is' }, { cue: 'What it shows' }, { cue: 'Claim, bounded' }],
    notes: [
      "BERT does not generalise: MCC 0.000 on train, validation, test.",
      "DeepSeek zero-shot: some generalisation, wide intervals.",
      "Test = newest cases; cutoffs DV 2025-05-01, BoC 2023-10-01.",
      "Caveat: retrospective classification of appeals already written.",
      "Claim bounded: same courts and domains only.",
      "If pushed: DeepSeek may have seen public DGSI texts; check its cutoff."
    ],
    sources: []
  });

  /* ======================= 15  WHY BERT COLLAPSED ======================= */
  slides.push({
    id: 'bert-collapse', number: 15, title: 'Why BERT Collapsed', layout: 'cards', cols: [
      { title: 'UNDERFITTING, NOT OVERFITTING', html: '<p><b>DV:</b> MCC 0.000 on train, validation AND test. It said KEPT for everything, even the 435 ALTERED cases it trained on (train set 51.9% / 48.1%).</p><p><b>BoC:</b> MCC 0.303 train → 0.184 validation → 0.000 test. PARTIAL never predicted, despite 328 training examples.</p>' },
      { title: 'CONSISTENT WITH', at: 1, html: '<p>Heavy regularisation for a small dataset: dropout 0.3, weight decay 0.05, label smoothing 0.1, learning rate 5e-6, early-stopping patience 7.</p><p>No hyperparameter search (hardware budget) and a domain shift: from legal text similarity to outcome classification.</p><p>Long documents cut to 16 windows and averaged, with the ruling itself removed from the input.</p>' }
    ],
    columns: '1fr 1fr', cardsH: 500, fs: 30,
    caption: { at: 2, html: 'This fine-tuning setup could not learn the task from ~1-2k cases.<br>However BERT stayed useful as the sentence encoder of the LLM summarization pipeline.' },
    steps: [{ cue: 'Underfitting, not overfitting' }, { cue: 'Consistent with (not tested)' }, { cue: 'Take-away' }],
    notes: [
      "Supervisor: testing is valid; the collapse is a result.",
      "DV: MCC 0.000 on all splits; KEPT for everything.",
      "BoC: MCC 0.303 train, 0.184 validation, 0.000 test.",
      "Underfitting, not overfitting: no train/test gap.",
      "Consistent with: heavy regularisation, low learning rate, no search, domain shift, 16 windows.",
      "None tested one by one; untuned baseline, the slide says so.",
      "Encoder stayed useful, frozen, for summarisation."
    ],
    sources: []
  });

  /* ======================= 16  SCALE AND METHODS, NOT CAPABILITY ======================= */
  slides.push({
    id: 'scale-methods', number: 16, title: 'Scale and methods, Not Capability', layout: 'custom',
    steps: [{ cue: 'Long 2018' }, { cue: 'Menezes-Neto 2022' }, { cue: 'Bertalan 2024' }, { cue: 'This thesis' }, { cue: 'Two differences: method and scale' }, { cue: 'Not BERT per se' }],
    notes: [
      "Older numbers look better: different input, different split.",
      "Long: facts + pleas + law articles; random split.",
      "Bertalan: outcome terms kept; random 10-fold; plain GRU matches.",
      "Menezes-Neto: temporal, no leakage; 700k appeals, best MCC 0.37.",
      "Here: 1-2k cases and the ruling removed.",
      "Real prediction should precede judgment; that data is sensitive."
    ],
    sources: [{ label: 'Long et al. 2018 (arXiv:1809.06537)', url: URL.long }, { label: 'Menezes-Neto 2022 (PLOS ONE)', url: URL.menezes }, { label: 'Bertalan 2024 (arXiv:2207.08823)', url: URL.bertalan }],
    html: function () {
      var COLS = '260px 190px 290px 250px 170px 1fr';
      function hd(t, cls) { return '<div class="sth ' + (cls || '') + '">' + t + '</div>'; }
      function row(i, a, cls, study, size, inn, out, split, res) {
        return box({ x: 80, y: 74 + i * 112, w: 1760, h: 104, at: a, cls: 'st-row ' + cls, style: 'grid-template-columns:' + COLS },
          '<div class="st-name">' + study + '</div><div class="st-scale">' + size + '</div><div>' + inn + '</div><div>' + out + '</div><div class="st-split">' + split + '</div><div>' + res + '</div>');
      }
      return '' +
        box({ x: 80, y: 8, w: 1760, h: 56, cls: 'st-head', style: 'grid-template-columns:' + COLS },
          hd('Study') + hd('Corpus size', 'scale') + hd('In', 'meth') + hd('Out', 'meth') + hd('Split', 'meth') + hd('Result')) +
        row(0, 0, '', link(URL.long, 'Long et al. 2018 (China)'), '100,000', 'facts + plaintiff\'s pleas + law articles', 'is each plea granted', 'Random', 'AutoJudge (Bi-GRU + attention): 82.2% acc / 83.4% F1') +
        row(1, 1, '', link(URL.menezes, 'Menezes-Neto 2022 (Brazil)'), '700,000+', 'first-instance decision', 'appeal court reverses or confirms it', 'Temporal', 'BERT+LSTM best, MCC 0.369 (ULMFiT and BigBird lower); legal experts 0.125') +
        row(2, 2, '', link(URL.bertalan, 'Bertalan 2024 (Brazil)'), '2,467', 'whole decision, chosen for explicit outcome terms', 'condemnation or absolution', 'Random 10-fold CV', 'HAN 98.7–99.8% acc (10-fold CV); a plain GRU matches it (99.2–99.7%)') +
        row(3, 3, 'thesis', 'This thesis', '1,126 DV / 1,802 BoC', 'appeal decision <b>without its ruling</b>', 'that ruling (2 or 3 classes)', 'Temporal hold-out', 'Fine-tuned LegalBERTimbau → MCC 0.000, both domains') +
        box({ x: 80, y: 536, w: 868, h: 116, at: 4, cls: 'dbox center', style: 'font-size:29px;background:#fbd9cc;border-color:#c3542b;text-align:center' },
          '<b style="color:#c3542b;letter-spacing:1px">METHODOLOGY DIFFERS</b><br>what the input contains and how the split is made') +
        box({ x: 972, y: 536, w: 868, h: 116, at: 4, cls: 'dbox center', style: 'font-size:29px;background:#e5eaf0;border-color:#394c60;text-align:center' },
          '<b style="color:#394c60;letter-spacing:1px">SCALE DIFFERS</b><br>100,000 / 700,000+ / 2,467 vs 1,126 DV / 1,802 BoC') +
        box({ x: 80, y: 672, w: 1760, h: 130, at: 5, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:35px' },
          '<div>BERT&rsquo;s collapse is <u>not</u> a problem of BERT or deep learning per se.</div>');
    }
  });

  /* ======================= 17  THE FEW-SHOT PARADOX ======================= */
  slides.push({
    id: 'few-shot-paradox', number: 17, title: 'The few-shot paradox', layout: 'custom',
    steps: [{ cue: 'Zero-shot values (DV, BoC)' }, { cue: 'Few-shot values appear' }, { cue: 'Take-away' }],
    notes: [
      "A genuine negative result, not a bug.",
      "DV: MCC 0.592 to 0.486; minority recall 0.619 to 0.429.",
      "BoC: MCC up, Macro-F1 down, FAVORABLE recall 0.833 to 0.333.",
      "k=1 per class: thin, high-variance anchor; similarity is only a proxy.",
      "Caution: n=50, CIs overlap; I do not claim few-shot is worse.",
      "Land it: small local model, no training, already beats BERT.",
      "If asked: no like-for-like LLM studies in related work."
    ],
    sources: [{ label: 'Alfieri et al. 2024 (CEUR-WS Vol-4089, paper 2)', url: URL.alfieri }, { label: 'Hannan 2026 (arXiv:2510.27675)', url: URL.hannan }, { label: 'Min et al. 2022 (arXiv:2202.12837)', url: URL.min }],
    html: function () {
      function row(label, z, f, up) {
        return '<div style="display:grid;grid-template-columns:300px 190px 1fr;align-items:center;font-size:46px;height:96px;border-bottom:2px solid rgba(108,33,24,.15)">' +
          '<div style="font-size:31px;font-weight:600;color:#394c60">' + label + '</div>' +
          '<div style="font-weight:700;color:#6c2118">' + z + '</div>' +
          '<div data-at="1" style="font-weight:700;color:' + (up ? '#1b5e30' : '#a8322a') + '">→ ' + f + '</div></div>';
      }
      function card(x, tag, rows) {
        return box({ x: x, y: 20, w: 860, h: 450, cls: 'card', style: 'padding:26px 40px' },
          '<div style="display:grid;grid-template-columns:300px 190px 1fr;align-items:center"><div><span class="tag" style="font-size:34px;padding:2px 26px">' + tag + '</span></div>' +
          '<div style="font-size:24px;letter-spacing:1px;font-weight:700;color:#6c2118">ZERO-SHOT</div><div data-at="1" style="font-size:24px;letter-spacing:1px;font-weight:700;color:#6c2118">FEW-SHOT</div></div>' +
          '<div style="margin-top:14px">' + rows + '</div>');
      }
      return '' +
        card(80, 'DV', row('MCC', '0.592', '0.486', false) + row('Macro-F1', '0.783', '0.696', false) + row('Minority recall', '0.619', '0.429', false)) +
        card(980, 'BoC', row('MCC', '0.468', '0.523', true) + row('Macro-F1', '0.663', '0.631', false) + row('Minority recall', '0.833', '0.333', false)) +
        box({ x: 80, y: 510, w: 1760, h: 290, at: 2, cls: 'caption', style: 'display:flex;flex-direction:column;justify-content:center;gap:16px' },
          '<div style="font-size:42px">Few-shot did not help the way I hypothesised.</div>' +
          '<div style="font-size:29px;font-style:normal;font-weight:600;color:#394c60">Consistent with one example per class being a thin, high-variance anchor, and with good examples being hard to pick: how many, and which ones, matter.</div>');
    }
  });

  /* ======================= 18  WHAT COULD IMPROVE THE LLMs ======================= */
  slides.push({
    id: 'improve-llms', number: 18, title: 'What Could Improve the LLMs', layout: 'pillRows', tone: 'orange', rowH: 104, gap: 12, badgeW: 230, fs: 27,
    rows: [
      { badge: 'FEW-SHOT', html: 'One retrieved example per class pulled predictions toward the majority class → better example selection (more, diverse, balanced)' },
      { badge: 'SUMMARY', html: 'Hand-set anchors and weights, never evaluated on its own → measure what is lost, make it data-driven', at: 1 },
      { badge: 'UNTESTED', html: 'Retrieval of statutes and precedents (RAG, H1b) and a chain-of-thought ablation — not tested', at: 2 },
      { badge: 'MODELS', html: 'One 8B model family → run other open-weight models on the same prompts', at: 3 },
      { badge: 'EVAL', html: 'n = 50 per domain, wide CIs → a larger, judge-validated gold set', at: 4 }
    ],
    caption: { at: 5, html: 'With no training and no examples, one prompt already reached MCC 0.59 / 0.47: the pipeline around the model is where the headroom is.' },
    steps: [{ cue: 'Few-shot' }, { cue: 'Summary' }, { cue: 'Untested' }, { cue: 'Models' }, { cue: 'Evaluation' }, { cue: 'Headroom' }],
    notes: [
      "Few-shot: k=1 pulled toward majority; better selection.",
      "Summary: hand-set weights; measure what is lost.",
      "Untested: RAG over statutes and precedents, CoT ablation.",
      "Models: only one 8B family run.",
      "Evaluation: n=50, wide CIs; larger judge-validated set."
    ],
    sources: []
  });

  /* ======================= 19  SHAP ON A COLLAPSED BERT ======================= */
  slides.push({
    id: 'shap', number: 19, title: 'SHAP on a Collapsed BERT', layout: 'custom',
    steps: [{ cue: 'What SHAP is (football analogy)' }, { cue: 'The model learned nothing' }, { cue: 'The scale: 48.9% → 83.6%' }, { cue: 'Top sentences, in percentage points' }, { cue: 'Nothing to validate' }],
    notes: [
      "Baseline = what the model outputs when ALL text is masked.",
      "Masked text is fed as the literal string \"...\" (explanation.py L92-103), not an empty string.",
      "Baseline output: 48.9% KEPT / 51.1% ALTERED, about a coin flip: the model's prior with no information.",
      "Train split (DV, before cutoff, n=1,064): 51.9% KEPT / 48.1% ALTERED; baseline is close but not literally that.",
      "Training used balanced class weights + label smoothing, pushing uninformative inputs to about 50/50.",
      "Then: baseline 48.9 + all 367 sentences (+34.8) = 83.6% KEPT.",
      "Which class is pushed? KEPT (decisão mantida) = predicted class; 0-100 = probability of KEPT in %.",
      "Analogy: scoreboard says HOME WIN every match; KEPT probabilities 77.5-87.8% in 5 of 5 DV cases.",
      "Case 5052 was truly ALTERED yet predicted KEPT (86.9%).",
      "Strongest sentence +0.57 points; median 0.08; 356 up, 11 down.",
      "Analogy limit: SHAP explains the model's behaviour, not legal reality; intuition only.",
      "If asked BoC: max about 2.9 points; repeated monetary figures (backup A4)."
    ],
    sources: [{ label: 'SHAP: Lundberg & Lee 2017 (arXiv:1705.07874)', url: URL.shap }, { label: 'Case 65/24.0GEBRG.G1 (jurisprudencia.csm.org.pt)', url: URL.shapCase }],
    html: function () {
      /* real values: sentence-level SHAP of bert_v3 on DV case 65/24.0GEBRG.G1, class KEPT (decisao_mantida), in percentage points */
      var rows = [
        ['489.º n.º 2 do CP, que refere que «A indemnização será, todavia calculada…»', 0.57],
        ['ENCONTRANDO-SE ASSIM PREENCHIDOS DE FORMA SUFICIENTE, ADEQUADA E PROPORCIONAL…', 0.36],
        ['Acordam em conferência na Secção Penal do Tribunal da Relação de Guimarães:', 0.33],
        ['I – Relatório · No processo comum singular com o NUIPC 65/24.0GEBRG…', 0.33],
        ['a) do Código Penal;', 0.30],
        ['EXAS., VENERANDOS JUÍZES DESEMBARGADORES, A COSTUMADA E SÃ JUSTIÇA!', 0.26],
        ['O Ministério Público respondeu ao recurso interposto, que rematou com…', 0.26],
        ['b) Declarar a mesma arguida inimputável perigoso em razão de anomalia psíquica…', 0.24]
      ];
      var h = '';
      var casehd = box({ x: 80, y: 6, w: 1760, h: 50, at: 1, style: 'font-size:32px;font-weight:700;color:#394c60' },
        'DV case&nbsp;' + link(URL.shapCase, '65/24.0GEBRG.G1') + '&nbsp;· true label KEPT · predicted KEPT (83.6%)');

      /* ---- step 0: the football analogy (visible only at step 0) ---- */
      var pl = [[0.9, 1], [-0.5, 0], [0.3, 1], [1.4, 1], [-1.0, 0], [0.5, 1], [-0.3, 0], [1.0, 1], [0.2, 1], [-0.7, 0], [0.6, 1]];
      var vis = '<line x1="120" y1="400" x2="1800" y2="400" stroke="#394c60" stroke-width="4" stroke-dasharray="14 8"/>' +
        '<text x="960" y="700" font-size="26" fill="#394c60" text-anchor="middle" font-family="Open Sans,Arial,sans-serif" font-weight="700">- - -  the average player (the baseline)</text>';
      pl.forEach(function (p, i) {
        var x = 190 + i * 150, hgt = Math.abs(p[0]) * 90, y = p[0] > 0 ? 400 - hgt : 400;
        vis += '<rect x="' + (x - 22) + '" y="' + y + '" width="44" height="' + hgt + '" rx="6" fill="' + (p[1] ? '#2f8a4c' : '#a8322a') + '"/>' +
          '<circle cx="' + x + '" cy="' + (p[0] > 0 ? 400 - hgt - 26 : 400 + hgt + 26) + '" r="18" fill="#fff" stroke="#394c60" stroke-width="4"/>';
      });
      vis += '<text x="140" y="262" font-size="27" fill="#2f8a4c" font-weight="700" font-family="Open Sans,Arial,sans-serif">above average</text>' +
        '<text x="140" y="560" font-size="27" fill="#a8322a" font-weight="700" font-family="Open Sans,Arial,sans-serif">below average</text>';
      h += box({ x: 0, y: 0, w: 1920, h: 840, at: 0, cl: '0-0:on', cls: 'only' },
        box({ x: 80, y: 8, w: 280, h: 44, cls: 'tag', style: 'font-size:28px;text-align:center;background:#394c60' }, 'AN ANALOGY') +
        box({ x: 80, y: 60, w: 1760, h: 150, style: 'font-size:37px;line-height:1.3;font-weight:600;color:#26221f' },
          'A football match ends with a result: win or lose.<br>After the game, SHAP tells <b>how much each player contributed</b> to it,<br>compared with the <b>average player</b>: some above, some below.') +
        box({ x: 1000, y: 150, w: 840, h: 44, style: 'font-size:30px;font-weight:700;color:#a8322a;text-align:right' }, 'SHAP explains the model, not the world.') +
        svg(vis) +
        box({ x: 80, y: 612, w: 560, h: 190, cls: 'tile', style: 'display:flex;flex-direction:column;justify-content:center' }, '<div class="big" style="font-size:40px">PLAYERS</div><div class="small" style="font-size:28px">= the sentences of the judgment</div>') +
        box({ x: 680, y: 612, w: 560, h: 190, cls: 'tile', style: 'display:flex;flex-direction:column;justify-content:center' }, '<div class="big" style="font-size:40px">RESULT</div><div class="small" style="font-size:28px">= the model\'s prediction</div>') +
        box({ x: 1280, y: 612, w: 560, h: 190, cls: 'tile', style: 'display:flex;flex-direction:column;justify-content:center' }, '<div class="big" style="font-size:40px">AVERAGE PLAYER</div><div class="small" style="font-size:28px">= the model\'s output when the text is masked</div>'));

      /* ---- step 1: the model learned nothing (+ continuation of the analogy) ---- */
      h += casehd;
      h += box({ x: 0, y: 60, w: 1920, h: 780, at: 1, cl: '1-1:on', cls: 'only' },
        box({ x: 80, y: 10, w: 460, h: 44, cls: 'tag', style: 'font-size:26px;text-align:center;background:#394c60;white-space:nowrap' }, 'THE ANALOGY, CONTINUED') +
        box({ x: 80, y: 66, w: 860, h: 330, cls: 'card', style: 'padding:20px 30px' },
          '<h3 style="margin:0 0 10px">FOOTBALL</h3><div class="card-body" style="font-size:30px;line-height:1.3">The scoreboard says <b>HOME WIN</b> in every match, <b>whoever plays</b>.<div style="margin-top:14px;font-size:50px;letter-spacing:6px">&#9917; 1–0 &nbsp;&#9917; 1–0 &nbsp;&#9917; 1–0</div></div>') +
        box({ x: 980, y: 66, w: 860, h: 330, cls: 'card', style: 'padding:20px 30px;background:#f5d9d5' },
          '<h3 style="margin:0 0 10px;color:#a8322a;border-color:#a8322a">BERT</h3><div class="card-body" style="font-size:30px;line-height:1.3"><b>KEPT</b> in <b>5 of 5</b> DV cases<br><b>UNFAVORABLE</b> in <b>5 of 5</b> BoC cases<br><span style="color:#a8322a;font-weight:700">(one DV case was actually ALTERED)</span><div style="margin-top:10px">MCC 0.000 on train, validation and test (DV)</div></div>') +
        box({ x: 80, y: 430, w: 1760, h: 110, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:36px' },
          '<div>If the result does not depend on the players, asking who scored it tells us nothing.</div>') +
        box({ x: 80, y: 570, w: 1760, h: 60, style: 'font-size:30px;font-weight:600;color:#394c60;text-align:center' }, 'The model learned nothing: by construction, SHAP has little to explain.') +
        box({ x: 80, y: 650, w: 1760, h: 40, style: 'font-size:24px;color:#6b6560;text-align:center;font-style:italic' }, 'An analogy, for intuition only.'));

      /* ---- step 2: the scale, on the real 0-100 axis (model probability of KEPT) ---- */
      var X0 = 160, PX = 16;            /* 100 points = 1600 px */
      var scale = '';
      for (var t = 0; t <= 100; t += 25) scale += '<line x1="' + (X0 + t * PX) + '" y1="196" x2="' + (X0 + t * PX) + '" y2="360" stroke="#d6dbe1" stroke-width="2"/><text x="' + (X0 + t * PX) + '" y="394" font-size="26" fill="#394c60" text-anchor="middle" font-family="Open Sans,Arial,sans-serif">' + t + '%</text>';
      scale += '<text x="' + (X0 + 50 * PX) + '" y="428" font-size="25" fill="#394c60" text-anchor="middle" font-family="Open Sans,Arial,sans-serif">percentage points of the model\'s probability for KEPT</text>';
      h += box({ x: 0, y: 60, w: 1920, h: 780, at: 2, cl: '2-2:on', cls: 'only' },
        box({ x: 80, y: 10, w: 1760, h: 50, style: 'font-size:30px;font-weight:600;color:#394c60' }, 'The scale: 0–100 is the model\'s probability (%) of the class <b>KEPT</b>') +
        box({ x: X0, y: 140, w: Math.round(48.9 * PX), h: 70, style: 'background:#9aa0a6;border-radius:12px 0 0 12px;color:#fff;font-size:28px;font-weight:700;display:flex;align-items:center;justify-content:center' }, 'whole text masked → 48.9% KEPT') +
        box({ x: X0 + Math.round(48.9 * PX), y: 140, w: Math.round(34.8 * PX), h: 70, style: 'background:#c3542b;color:#fff;font-size:28px;font-weight:700;display:flex;align-items:center;justify-content:center' }, '+34.8 (367 sentences)') +
        box({ x: X0 + Math.round(83.6 * PX), y: 140, w: Math.round(16.4 * PX), h: 70, style: 'background:#eceeef;border-radius:0 12px 12px 0;border:3px dashed #9aa0a6;color:#7b8187;font-size:26px;display:flex;align-items:center;justify-content:center' }, '') +
        svg(scale, 3) +
        box({ x: 80, y: 450, w: 880, h: 130, cls: 'dbox center', style: 'font-size:29px;text-align:center;border-color:#c3542b;background:#fbd9cc' },
          '<b>Full text → 83.6% KEPT</b><br>all 367 sentences together: 48.9 + 34.8 = 83.6<br>356 push towards KEPT, 11 slightly against') +
        box({ x: 1000, y: 450, w: 840, h: 130, cls: 'dbox center', style: 'font-size:29px;text-align:center;border-color:#a8322a;background:#f5d9d5' },
          '<b>Best single sentence: +0.57 points</b><br>(a thin sliver on this axis)<br>median sentence: +0.08') +
        box({ x: X0 + Math.round(48.9 * PX) - 3, y: 128, w: 6, h: 94, style: 'background:#26221f;z-index:5' }, '') +
        box({ x: 80, y: 620, w: 1760, h: 150, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:34px' },
          '<div>The confidence is spread thinly over the whole text:<br>no sentence steers the model in a meaningful way.</div>'));

      /* ---- step 3: top sentences on a fixed 0-5 axis ---- */
      var AX0 = 1160, PXS = 130;
      var bars = box({ x: 80, y: 8, w: 1760, h: 46, style: 'font-size:27px;font-weight:600;color:#394c60' }, 'Top 8 of 367 sentences: how many <b>percentage points of P(KEPT)</b> each one adds (fixed axis 0–5)');
      rows.forEach(function (r, i) {
        var y = 62 + i * 58, w = Math.max(3, Math.round(r[1] * PXS));
        bars += box({ x: 80, y: y, w: 1050, h: 52, style: 'font-size:24px;line-height:1.1;display:flex;align-items:center;white-space:nowrap;overflow:hidden;color:#26221f' }, '<div style="overflow:hidden;text-overflow:ellipsis;width:100%">' + r[0] + '</div>') +
          box({ x: AX0, y: y + 8, w: w, h: 36, style: 'background:#c3542b;border-radius:4px' }, '') +
          box({ x: AX0 + w + 12, y: y + 6, w: 120, h: 40, style: 'font-size:27px;font-weight:700;color:#6c2118;display:flex;align-items:center' }, '+' + r[1].toFixed(2));
      });
      var ax = '<line x1="' + AX0 + '" y1="58" x2="' + AX0 + '" y2="530" stroke="#394c60" stroke-width="3"/>';
      for (var k = 0; k <= 5; k++) ax += '<line x1="' + (AX0 + k * PXS) + '" y1="58" x2="' + (AX0 + k * PXS) + '" y2="530" stroke="#d6dbe1" stroke-width="2"' + (k ? ' stroke-dasharray="6 6"' : '') + '/>' +
        '<text x="' + (AX0 + k * PXS) + '" y="566" font-size="26" fill="#394c60" text-anchor="middle" font-family="Open Sans,Arial,sans-serif">' + k + '</text>';
      ax += '<text x="' + (AX0 + 2.5 * PXS) + '" y="600" font-size="24" fill="#394c60" text-anchor="middle" font-family="Open Sans,Arial,sans-serif">percentage points of the model\'s probability for KEPT</text>';
      h += box({ x: 0, y: 60, w: 1920, h: 780, at: 3, cl: '3-4:on', cls: 'only' }, bars + svg(ax, 4)) +
        box({ x: 80, y: 700, w: 1760, h: 110, at: 3, cl: '3-3:on', cls: 'caption only', style: 'display:flex;align-items:center;justify-content:center;font-size:33px' },
          '<div>The strongest sentence adds only <b>0.57 points</b> to P(KEPT) (median sentence: 0.08).<br>Mostly formulaic, procedural sentences, not legal reasoning.</div>');

      /* ---- step 4: takeaway ---- */
      h += box({ x: 80, y: 700, w: 1760, h: 110, at: 4, cls: 'dbox center', style: 'z-index:6;font-size:35px;font-weight:700;color:#6c2118;text-align:center;border-radius:30px;border-color:#6c2118;background:#f6e0d6' },
        'SHAP is faithful to a model that learned nothing.<br><span style="font-weight:600;color:#26221f;font-size:27px">Nothing to validate with the judges: that is why I did not ask them to rate it.</span>');
      return h;
    }
  });

  /* ======================= 20  READING THE LLM'S REASONING ======================= */
  slides.push({
    id: 'llm-reasoning', number: 20, title: 'Reading the LLM\'s Reasoning', layout: 'custom',
    steps: [{ cue: 'Case + reasoning trace' }, { cue: 'True label vs prediction' }, { cue: 'Fluent, legally framed and wrong' }],
    notes: [
      "Verbatim think-trace from thesis appendix; cuts marked (...).",
      "True label ALTERED; model said KEPT.",
      "Fluent and legal-sounding, yet wrong.",
      "Turpin et al. 2023: convincing explanations for wrong answers.",
      "Versus SHAP: faithful to nothing learned; trace legible, not guaranteed faithful.",
      "Neither validated by experts, so RQ2 stays partial."
    ],
    sources: [{ label: 'Case 236/23.6PBHRT.L1-9 (jurisprudencia.csm.org.pt)', url: URL.case236 }, { label: 'Turpin et al. 2023 (arXiv:2305.04388)', url: URL.turpin }],
    html: function () {
      return '' +
        box({ x: 80, y: 24, w: 900, h: 70, style: 'font-size:38px;font-weight:700;color:#394c60;display:flex;align-items:center' },
          'DV case&nbsp;' + link(URL.case236, '236/23.6PBHRT.L1-9')) +
        box({ x: 1000, y: 24, w: 840, h: 70, at: 1, style: 'display:flex;gap:22px;justify-content:flex-end;align-items:center' },
          '<span class="pill-label ok">True label: ALTERED</span><span class="pill-label no">DeepSeek: KEPT</span>') +
        box({ x: 80, y: 116, w: 1760, h: 500, cls: 'dbox fill', style: 'padding:40px 60px;border-color:#c3542b;border-left-width:14px;font-size:37px;line-height:1.4;font-style:italic;border-radius:26px' },
          '&ldquo;The defendant was initially acquitted by the lower court, and the prosecution appealed. (…)<br>' +
          'The appellate decision states that the lower court\'s sentence was upheld. (…)<br>' +
          'There\'s a mention of the appeal being dismissed without any criticism of the lower court\'s decision.<br><br>' +
          'The predicted_label should be &quot;decisao_mantida&quot; because the appellate court confirmed the lower court\'s acquittal.&rdquo;') +
        box({ x: 80, y: 650, w: 1760, h: 120, at: 2, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:35px' },
          '<div>Fluent, legally framed and wrong: a readable explanation is not a correct one (' + link(URL.turpin, 'Turpin et al., 2023') + ').</div>');
    }
  });

  /* ======================= 21  JUDGE VALIDATION ======================= */
  slides.push({
    id: 'judge-validation', number: 21, title: 'Judge Validation', layout: 'custom',
    steps: [{ cue: 'App demo video (press V to play)' }, { cue: 'Agreement counts' }, { cue: 'Footnote' }],
    notes: [
      "Press V to play or pause the demo video (audience window).",
      "FIRST: click once inside the audience window, so sound is allowed.",
      "Without a click the video starts muted; a click turns sound on.",
      "Status chip in the presenter bar shows playing / blocked.",
      "Video uses test accounts only; counts are small, comments matter more.",
      "Footnote: BoC 20 / 20 excluding one judge, chosen a posteriori."
    ],
    sources: [],
    html: function () {
      function tile(y, tag, num) {
        return box({ x: 1150, y: y, w: 690, h: 150, at: 1, cls: 'tile', style: 'display:flex;align-items:center;justify-content:space-between;padding:0 50px' },
          '<span class="tag" style="font-size:34px;padding:2px 26px">' + tag + '</span><span style="font-size:68px;font-weight:700;color:#6c2118">' + num + '</span>');
      }
      return '' +
        box({ x: 80, y: 30, w: 1000, h: 562, cls: 'video-panel', attrs: 'data-video="assets/video/app_demo.mp4" data-poster="assets/video/app_demo_poster.jpg"' },
          '<img class="vposter" src="assets/video/app_demo_poster.jpg" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000"><div class="video-ph" style="background:none"><div class="play" style="margin:0"></div><div class="vblocked">Click here to start the video</div></div>') +
        box({ x: 1150, y: 40, w: 690, h: 100, at: 1, style: 'font-size:38px;font-weight:700;color:#394c60;display:flex;align-items:center' }, 'Judges agreed with my labels in:') +
        tile(160, 'DV', '24 / 25') + tile(340, 'BoC', '21 / 25*') +
        box({ x: 80, y: 630, w: 1760, h: 190, at: 2, style: 'font-size:27px;line-height:1.4;color:#394c60' },
          '*BoC: 20 / 20 excluding one judge (chosen a posteriori) whose answers diverged from the other four. With 25 cases per domain the counts are too small for statistics — the comments (next slide) matter more in my opinion.');
    }
  });

  /* ======================= 22  JUDGES' COMMENTS ======================= */
  slides.push({
    id: 'judges-comments', number: 22, title: 'Judges\' Comments', layout: 'judges',
    intro: { html: 'Judges agreed with my labels, not that the labels are realistic: the rulings were still open to dispute.' },
    cardsH: 570,
    cards: [
      { at: 1, badge: 'J3 · DV', quotes: ['This first-instance sentence is the paradigmatic example of an excessive, systemic recourse to non-custodial measures, contrary to seeing domestic violence as a serious crime.'],
        meta: link(URL.j123, 'Case 123/22.5PACTX.E1 · agreed with the label, not confident') },
      { at: 2, badge: 'J2 + J3 · DV', quotes: ['I would have granted the prosecutor\'s appeal and revoked the decision.', 'The court erred in assessing the facts and applying the law.'],
        meta: link(URL.j87, 'Case 87/25.3PGSXL-A.L1-5 · both labelled it KEPT (correct), both not confident') },
      { at: 3, badge: 'J1 · BoC', quotes: ['Both parties may appeal, and the court can rule favourably on one appeal and unfavourably on the other.'],
        meta: link(URL.j1627, 'Case 1627/20.0T8PVZ.P1 · critic of the favourable / unfavourable label scheme') }
    ],
    steps: [{ cue: 'Framing' }, { cue: 'J3 · DV' }, { cue: 'J2 + J3 · DV' }, { cue: 'J1 · BoC' }],
    notes: '',
    sources: [{ label: 'Case 123/22.5PACTX.E1 (TRE, 2025)', url: URL.j123 }, { label: 'Case 87/25.3PGSXL-A.L1-5 (TRL, 2025)', url: URL.j87 }, { label: 'Case 1627/20.0T8PVZ.P1 (TRP, 2025)', url: URL.j1627 }]
  });

  /* ======================= 23  RQs AND OBJECTIVES ======================= */
  slides.push({
    id: 'rqs-objectives', number: 23, title: 'RQs and Objectives', layout: 'statusRows', badgeW: 380, fs: 30,
    rows: [
      { status: 'green', badge: 'RQ1: Answered', html: 'DeepSeek zero-shot outperforms fine-tuned LegalBERTimbau (MCC 0.592/0.468 vs 0.000) within this protocol — H1a supported.' },
      { status: 'amber', at: 1, badge: 'RQ2: Partially', html: 'Both models produce explanations (SHAP / reasoning traces), but neither validated against expert legal judgment — H2 unresolved.' },
      { status: 'grey', at: 2, badge: 'RQ3: Unanswered', html: 'DV vs BoC confidence intervals overlap; insufficient statistical power at n=50/domain — H3 open, not resolved.' },
      { status: 'green', at: 3, badge: 'OBJECTIVES', html: '<span style="white-space:pre-wrap">O1 ✓     O2 ✓     O3 — XAI explored, but not validated by a legal expert (same gap as RQ2).</span>' }
    ],
    steps: [{ cue: 'RQ1 answered' }, { cue: 'RQ2 partially' }, { cue: 'RQ3 unanswered' }, { cue: 'Objectives' }],
    notes: '', sources: []
  });

  /* ======================= 24  LIMITATIONS ======================= */
  slides.push({
    id: 'limitations', number: 24, title: 'Limitations', layout: 'pillRows', tone: 'red', rowH: 128, gap: 18, badgeW: 380, fs: 33,
    rows: [
      { badge: 'HEURISTIC LABELS', html: 'Class labels come from regex heuristics on the ruling text, not from a legal ground truth.' },
      { badge: 'SMALL &amp; NARROW', cls: 'alt', at: 1, html: 'Small dataset, single language and single jurisdiction (Portugal).' },
      { badge: 'HARDWARE', at: 2, html: 'Constrained by a 6 GB consumer GPU throughout — no large-scale tuning.' },
      { badge: 'ONE MODEL FAMILY', cls: 'alt', at: 3, html: 'Only one LLM family was run — no cross-model comparison.' },
      { badge: 'EXPLANATIONS', at: 4, html: 'SHAP and reasoning traces were never validated by legal experts at scale.' }
    ],
    steps: [{ cue: 'Heuristic labels' }, { cue: 'Small and narrow' }, { cue: 'Hardware' }, { cue: 'One model family' }, { cue: 'Explanations' }],
    notes: '', sources: []
  });

  /* ======================= 25 / 26  BIASES (DV, then BoC) ======================= */
  /* one builder for both domains: three EDA charts (class balance, outcome mix by year, judge gender) */
  function biasChartsHtml(files, caps, topIntro) {
    function chart(i, x) {
      return box({ x: x, y: 150, w: 540, h: 290, at: i, style: 'background:#fff;border-radius:12px;padding:6px' }, '<img src="assets/img/' + files[i] + '" alt="' + caps[i] + '" style="width:100%;height:100%;object-fit:contain">') +
        box({ x: x - 10, y: 446, w: 560, h: 62, at: i, style: 'font-size:26px;font-weight:600;color:#394c60;text-align:center;line-height:1.2' }, caps[i]);
    }
    return box({ x: 80, y: 4, w: 1760, h: 46, style: 'font-size:28px;font-weight:700;color:#394c60' }, topIntro) +
      box({ x: 80, y: 56, w: 1760, h: 470, cls: 'card', style: 'padding:20px 30px' }, '<h3 style="margin:0">DATA</h3>') +
      chart(0, 110) + chart(1, 690) + chart(2, 1270);
  }

  slides.push({
    id: 'biases-dv', number: 25, title: 'Biases: Domestic Violence', layout: 'custom',
    steps: [{ cue: 'DATA: class balance (DV)' }, { cue: 'DATA: outcome mix by year (DV)' }, { cue: 'DATA: judge gender (DV)' }, { cue: 'Reading' }],
    notes: [
      "Backup for bias questions (DV half).",
      "Classes 52.2% KEPT, 47.8% ALTERED.",
      "Outcome mix drifts by year; temporal split answers it.",
      "Judge gender via gender-guesser; no clear skew.",
      "Exploratory check only; AI360 and Responsible AI Toolbox not used.",
      "BERT: inverse-frequency weights; report MCC and Macro-F1."
    ],
    sources: [],
    html: function () {
      return biasChartsHtml(['bias_class_balance_dv.png', 'bias_outcome_by_year_dv.png', 'bias_judge_gender_dv.png'],
        ['Class balance (DV): 52 / 48%', 'Outcome mix drifts by year (DV)', 'Judge gender (DV): no clear skew'],
        'Three checks from the exploratory analysis — one per type, not an exhaustive fairness audit:') +
        box({ x: 80, y: 560, w: 1760, h: 150, at: 3, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:33px' },
          '<div>Classes are close to balanced, the outcome mix drifts over the years (the temporal split is the answer), and no clear skew by judge gender.</div>');
    }
  });

  slides.push({
    id: 'biases-boc', number: 26, title: 'Biases: Breach of Contract', layout: 'custom',
    steps: [{ cue: 'DATA: class balance (BoC)' }, { cue: 'DATA: outcome mix by year (BoC)' }, { cue: 'DATA: judge gender (BoC)' }, { cue: 'LABELS and MODELS' }, { cue: 'Closing line' }],
    notes: [
      "BoC half, plus LABELS and MODELS for both domains.",
      "Classes 50.3 / 28.3 / 21.4%; few cases before about 2004.",
      "Gender: small differences; not a fairness test.",
      "Labels: regex on the Dispositivo; judges criticised the scheme.",
      "Edge case: parcialmente improcedente.",
      "Models: no fairness analysis; prerequisite for deployment.",
      "Databases are published for lawyers; reuse is secondary."
    ],
    sources: [],
    html: function () {
      return biasChartsHtml(['bias_class_balance_boc.png', 'bias_outcome_by_year_boc.png', 'bias_judge_gender_boc.png'],
        ['Class balance (BoC): 50 / 28 / 21%', 'Outcome mix drifts by year (BoC)', 'Judge gender (BoC): similar mix, small differences'],
        'Same three checks for breach of contract; then labels and models:') +
        box({ x: 80, y: 536, w: 868, h: 200, at: 3, cls: 'card' }, '<h3>LABELS</h3><div class="card-body" style="font-size:25px"><b>Labels are also a bias:</b> regex over the court\'s own wording; judges criticised the scheme. Known edge case: &ldquo;parcialmente improcedente&rdquo;.</div>') +
        box({ x: 972, y: 536, w: 868, h: 200, at: 3, cls: 'card' }, '<h3>MODELS</h3><div class="card-body" style="font-size:25px">No formal fairness analysis of labels or predictions — required before any real use.</div>') +
        box({ x: 80, y: 748, w: 1760, h: 72, at: 4, cls: 'caption', style: 'display:flex;align-items:center;justify-content:center;font-size:31px' },
          'I checked what the data allowed; the missing fairness audit is stated as a limitation, not hidden.');
    }
  });

  /* ======================= 26  IMPACT ======================= */
  slides.push({
    id: 'impact', number: 26, title: 'Impact', layout: 'pillRows', tone: 'green', rowH: 150, gap: 24, badgeW: 400, fs: 35,
    rows: [
      { badge: 'FIRST BASELINE', html: 'The first Portugal-specific LJP baseline.' },
      { badge: 'OPEN DATA', at: 1, html: 'Labelled DV/BoC dataset plus a public, reproducible repo (' + link(URL.zenodo, 'Zenodo DOI') + ').' },
      { badge: 'JUDGE-CHECKED', at: 2, html: 'Five-judge validation with a purpose-built annotation app.' },
      { badge: 'COLLABORATION', at: 3, html: 'A judiciary collaboration opened from a cold contact.' }
    ],
    steps: [{ cue: 'First baseline' }, { cue: 'Open data' }, { cue: 'Judge-checked' }, { cue: 'Collaboration' }],
    notes: '',
    sources: [{ label: 'Zenodo record (DOI 10.5281/zenodo.22116675)', url: URL.zenodo }]
  });

  /* ======================= 27  FUTURE WORK IN PORTUGUESE LJP ======================= */
  slides.push({
    id: 'future-work', number: 27, title: 'Future Work in Portuguese LJP', layout: 'stat', gap: 60,
    lines: [
      { html: 'Depth over breadth is the higher-value design at this scale.', fs: 66 },
      { html: 'Go deeper on the same comparison, not wider across model families.', fs: 54, cls: 'ink', at: 1 },
      { html: 'For the field: better-documented case law, not bigger models.', fs: 54, cls: 'ink', at: 2 }
    ],
    steps: [{ cue: 'Depth over breadth' }, { cue: 'Go deeper' }, { cue: 'For the field' }], notes: '', sources: []
  });

  /* ======================= 28  WHERE AI COULD HELP ======================= */
  slides.push({
    id: 'where-ai', number: 28, title: 'Where AI Could Help', layout: 'compare', cardsH: 600, fs: 37,
    cols: [
      { title: 'CAN HELP', cls: 'good', html: '<p>Low-risk, low-interpretation steps in a well-designed system: drafting help, document search, summarisation, etc.</p><p><i>Autonomy is reasonable here, depending on context.</i></p>' },
      { title: 'CANNOT (YET)', cls: 'cannot', at: 1, html: '<p>Steps where risk and interpretation are high, like the decision itself: <b>a human decides.</b></p>' }
    ],
    steps: [{ cue: 'Can help' }, { cue: 'Cannot (yet)' }],
    notes: [
      "Thesis: neither model suitable for autonomous decision-aiding.",
      "Low-risk uses: routing, prioritisation, process automation, ranking.",
      "My nuance: the line is risk and interpretation, not AI vs human.",
      "Low risk, little to interpret: some autonomy, depending on context.",
      "The decision itself: a human decides.",
      "Next: The Answer; the agentic slide is an outside view."
    ],
    sources: []
  });

  /* ======================= 29  THE ANSWER ======================= */
  slides.push({
    id: 'answer', number: 29, title: 'The Answer', layout: 'stat', gap: 44,
    lines: [
      { html: 'In January: &ldquo;Should LLMs be the priority for our Justice System?&rdquo;', fs: 44, cls: 'q' },
      { html: 'Not by themselves. Domain knowledge comes first.', fs: 78, at: 1 },
      { html: 'LLMs work best inside a well-designed system, with autonomy only where little needs interpreting. High-stakes decisions stay human.', fs: 40, cls: 'ink', at: 2 },
      { html: 'What if research teams spent real time inside the justice system, side by side with the people who use it every day, and public institutions invested in that partnership instead?', fs: 38, cls: 'pill q', at: 3 }
    ],
    steps: [{ cue: 'The January question' }, { cue: 'Not by themselves' }, { cue: 'Inside a well-designed system' }, { cue: 'Closing question' }],
    notes: [
      "Answers my own January question; deliberately soft.",
      "Domain knowledge first; LLMs need a system around them.",
      "Autonomy only where risk and interpretation are low.",
      "Closing question: teams inside the system, institutions investing.",
      "Personal view, not from the manuscript; I may be wrong.",
      "Do NOT name an institution on the slide.",
      "Next slides: an outsider's illustration, not the solution."
    ],
    sources: []
  });

  /* ======================= 30  A SINGLE CASE, STEP BY STEP ======================= */
  slides.push({
    id: 'single-case', number: 30, title: 'A Single Case, Step by Step', layout: 'custom',
    steps: [{ cue: 'Why walk through one case?' }, { cue: 'Stages 1-4' }, { cue: 'Stages 5-8' }, { cue: 'The Acórdão: input of this thesis' }],
    notes: [
      "Say out loud: an extra, an outsider's example; an expert would correct it.",
      "Petição Inicial (Art. 552º): the plaintiff states facts, grounds, claim.",
      "Contestação (Art. 572º): the defendant answers and raises exceptions.",
      "Audiência Prévia (Arts. 591º-595º): parties and judge delimit the dispute.",
      "Instrução (Art. 410º): proof on the themes of proof.",
      "Audiência Final (Arts. 603º-606º): testimony and oral arguments.",
      "Sentença (Art. 607º): the judge decides: facts, norms, decision.",
      "Apelação (Art. 644º): appeal to the higher court.",
      "Acórdão: the appeal ruling; the input of this thesis.",
      "Not domestic violence: that is criminal procedure (CPP)."
    ],
    sources: [
      { label: '1. Petição Inicial: CPC Art. 552º', url: cpc(552) }, { label: '2. Contestação: CPC Art. 572º', url: cpc(572) },
      { label: '3. Audiência Prévia: CPC Arts. 591º-595º (opens 591º)', url: cpc(591) }, { label: '4. Fase de Instrução: CPC Art. 410º', url: cpc(410) },
      { label: '5. Audiência Final: CPC Arts. 603º-606º (opens 603º)', url: cpc(603) }, { label: '6. Sentença: CPC Art. 607º', url: cpc(607) },
      { label: '7. Recurso de Apelação: CPC Art. 644º', url: cpc(644) }
    ],
    html: function () {
      var X = function (i) { return 80 + i * 453; };
      function stage(i, row, a, name, artHtml, cl) {
        var y = row ? 500 : 225, x = X(i % 4);
        return box({ x: x, y: y, w: 400, h: 220, at: a, cl: cl, cls: 'dbox fill center', style: 'text-align:center;border-radius:26px' },
          '<div style="font-size:40px;font-weight:700;color:inherit;line-height:1.15;margin-top:14px">' + name + '</div><div style="font-size:29px;margin-top:12px">' + artHtml + '</div>') +
          box({ x: x + 16, y: y - 28, w: 64, h: 64, at: a, cls: 'stagenum' }, String(i + 1));
      }
      return '' +
        box({ x: 120, y: 270, w: 1680, h: 280, cl: '1:fr-top', cls: 'caption fr', style: 'display:flex;align-items:center;justify-content:center;font-size:44px' },
          '<div>An extra: one breach-of-contract case, stage by stage &mdash; an outsider&rsquo;s example.<br><span style="font-size:.78em;font-weight:600;font-style:normal;color:#394c60">Someone who knows the domain can take this example and find the real problems in each stage, and whether a system like this is needed at all.</span></div>') +
        box({ x: 80, y: 122, w: 1760, h: 84, at: 1, style: 'font-size:28px;font-weight:600;color:#394c60;text-align:center;display:flex;align-items:center;justify-content:center;line-height:1.3' },
          'The main stages only: a breach-of-contract case under the CPC. Domestic violence follows another path, in criminal procedure (CPP).') +
        stage(0, 0, 1, 'Petição Inicial', link(cpc(552), 'Art. 552º')) +
        stage(1, 0, 1, 'Contestação', link(cpc(572), 'Art. 572º')) +
        stage(2, 0, 1, 'Audiência Prévia', link(cpc(591), 'Arts. 591º-595º')) +
        stage(3, 0, 1, 'Fase de Instrução', link(cpc(410), 'Art. 410º')) +
        stage(4, 1, 2, 'Audiência Final', link(cpc(603), 'Arts. 603º-606º')) +
        stage(5, 1, 2, 'Sentença', link(cpc(607), 'Art. 607º')) +
        stage(6, 1, 2, 'Recurso de Apelação', link(cpc(644), 'Art. 644º')) +
        stage(7, 1, 2, 'Acórdão', '<i>The input of this thesis</i>', '3:hl') +
        svg(arrow(X(0) + 404, 335, X(1) - 4, 335, { at: 1, w: 5 }) + arrow(X(1) + 404, 335, X(2) - 4, 335, { at: 1, w: 5 }) + arrow(X(2) + 404, 335, X(3) - 4, 335, { at: 1, w: 5 }) +
          path([[X(3) + 200, 447], [X(3) + 200, 474], [X(0) + 200, 474], [X(0) + 200, 494]], { at: 2, w: 5 }) +
          arrow(X(0) + 404, 610, X(1) - 4, 610, { at: 2, w: 5 }) + arrow(X(1) + 404, 610, X(2) - 4, 610, { at: 2, w: 5 }) + arrow(X(2) + 404, 610, X(3) - 4, 610, { at: 2, w: 5 }), 2);
    }
  });

  /* ======================= A POSSIBLE AGENTIC SYSTEM (animated, driven by the next key) ======================= */
  /* One slide only (the old static version was removed). Each key press = one step; inside a stage the
     small specialised agents wake up one after the other. Sub-steps are named after what the Código de
     Processo Civil (Lei 41/2013) asks for in each article (read on pgdlisboa.pt); the agents themselves are
     an ILLUSTRATION seen from the outside, not something from the paper or from the courts. */
  slides.push({
    id: 'agentic', number: 32, title: 'A Possible Agentic System', layout: 'custom', live: 'agenticLive',
    steps: [{ cue: 'One case enters' }, { cue: '1 Petição Inicial' }, { cue: '2 Contestação' }, { cue: '3 Audiência Prévia' }, { cue: '4 Instrução' }, { cue: '5 Audiência Final' },
      { cue: '6 Sentença (human only)' }, { cue: '7 Apelação' }, { cue: '8 Acórdão (this thesis)' }, { cue: 'Two kinds of memory' }, { cue: 'The decision stays with the judge' }],
    notes: [
      "Say out loud: outsider's view; a domain expert would correct the stages.",
      "After LegalWorld (Fudan, 2026), adapted to the CPC. Not built, not evaluated.",
      "Several small agents per stage; sub-steps follow what each CPC article asks.",
      "Stage 4: the parties validate the evidence summaries.",
      "Stage 6 Sentença: no model; judge decides alone. Stage 8: input of this thesis.",
      "Simplified: CPC has loops (Art. 590º, 662º); not drawn.",
      "If asked: deployed? Only assistants; the human judge decides."
    ],
    sources: [{ label: 'LegalWorld (Fudan, 2026), arXiv:2606.18728', url: URL.legalworld }, { label: 'SimCourt (arXiv:2508.17322)', url: URL.simcourt },
      { label: '1. Petição Inicial: CPC Art. 552º', url: cpc(552) }, { label: '2. Contestação: CPC Art. 572º', url: cpc(572) }, { label: '3. Audiência Prévia: CPC Arts. 591º-595º', url: cpc(591) },
      { label: '3b. Art. 596º: objeto do litígio e temas da prova', url: cpc(596) }, { label: '4. Fase de Instrução: CPC Art. 410º', url: cpc(410) }, { label: '4b. Art. 598º: requerimento probatório', url: cpc(598) },
      { label: '5. Audiência Final: CPC Arts. 603º-606º', url: cpc(603) }, { label: '6. Sentença: CPC Art. 607º', url: cpc(607) }, { label: '7. Recurso de Apelação: CPC Art. 644º', url: cpc(644) }],
    liveData: {
      headings: { start: 'One breach-of-contract case', gating: 'Stage gating', memory: 'Two kinds of memory', recap: 'Outputs flow forward only', closing: 'The decision stays with the judge' },
      gate1: 'Stage gating: stage 1 sees nothing from later stages',
      gateN: 'Stage gating: stage {n} sees only the outputs of stages {m}',
      gateAll: 'Stage gating: a stage never sees the outputs of later stages',
      startInput: 'A client asks for help with a breach of a supply contract.',
      startCards: [{ name: 'Specialised agents', job: 'small and narrow; asleep until their stage', cls: 'idle', num: '·' }],
      emptyLocal: 'empty', startOut: '—', startNote: 'One case will flow through the 8 stages of the Portuguese civil process.',
      gatingInput: 'Example: stage 4 (Instrução).',
      gatingCards: [
        { name: 'Stages 1–3', job: 'outputs readable by stage 4', cls: 'ok', num: '✓' },
        { name: 'Stage 4', job: 'works with what stages 1–3 produced', cls: '', num: '4' },
        { name: 'Stages 5–8', job: 'LOCKED: the outcome never leaks backwards', cls: 'locked', num: '✗' }
      ],
      gatingLocal: 'this stage only', gatingOut: 'Nothing from stages 5–8.', gatingNote: 'A stage never sees the outputs of later stages.',
      memoryInput: 'The same case file, as the agents keep it.',
      memoryCards: [
        { name: 'LOCAL memory', job: 'one stage only: conversation and drafts of that step', cls: 'slate', num: 'L' },
        { name: 'GLOBAL memory', job: 'the whole case: facts, evidence, claims, deadlines, results', cls: 'slate', num: 'G' },
        { name: 'Memory tool', job: 'revise and expand the case fields', cls: 'slate', num: 'T' }
      ],
      memoryLocal: 'cleared when the stage ends', memoryOut: 'Case file grows stage by stage.', memoryNote: 'Local memory is cleared at each stage; global case memory grows.',
      recapInput: 'The case file after 8 stages.', recapCards: [{ name: 'Agents prepare', job: 'each stage hands a polished artefact to the next', cls: 'idle', num: '→' }],
      clearedLocal: 'cleared after each stage', recapOut: 'Nothing is delegated at the decision.', recapNote: 'A stage never sees the outputs of later stages.',
      closingInput: 'Everything the agents prepared.', closingCards: [{ name: 'Human judge', job: 'decides alone; the parties validate the evidence', cls: 'human', num: '⚖' }],
      closingOut: 'Sentence: written by the judge.', closingNote: 'Nothing is delegated. The decision stays with the judge.',
      stages: [
        { n: 1, name: 'Petição Inicial', art: 'Art. 552º', input: 'A client asks for the initial petition.',
          agents: [
            { name: 'Intake agent', job: 'court, parties, lawyer and form of process', tool: 'Retrieval' },
            { name: 'Facts agent', job: 'essential facts and legal grounds of the action', tool: 'Retrieval' },
            { name: 'Claim agent', job: 'the claim (pedido) and the value of the case', tool: 'Drafting' },
            { name: 'Review agent', job: 'completeness against Art. 552º; witness and proof list', tool: 'Verification' }],
          local: 'Sections of the petition, as drafted', glob: 'Facts · claims', out: 'Petição Inicial, ready to file', note: 'Several agents cooperate before anything is filed.' },
        { n: 2, name: 'Contestação', art: 'Art. 572º', input: 'The petition arrives from stage 1.',
          agents: [
            { name: 'Matching agent', job: 'individualise the action', tool: 'Retrieval' },
            { name: 'Defence agent', job: 'reasons of fact and law against the claim', tool: 'Drafting' },
            { name: 'Exceptions agent', job: 'essential facts of each exception, separately', tool: 'Drafting' },
            { name: 'Evidence agent', job: 'witness list and other proof requests', tool: 'Retrieval' }],
          local: 'Defence sections and exceptions', glob: 'Defence · exceptions', out: 'Contestação, ready to file', note: 'Sub-steps follow the elements asked by Art. 572º.' },
        { n: 3, name: 'Audiência Prévia', art: 'Arts. 591º-595º', input: 'Petition and defence arrive.',
          agents: [
            { name: 'Registry agent', job: 'prepare the convening and the schedule', tool: 'Verification' },
            { name: 'Dispute agent', job: 'summarise the positions; delimit the dispute', tool: 'Retrieval' },
            { name: 'Draft agent', job: 'draft the object of the dispute and proof themes, for the judge', tool: 'Drafting' },
            { name: 'Planning agent', job: 'plan the final hearing: sessions and dates', tool: 'Verification' }],
          local: 'Dispute summary and drafts for the judge', glob: 'Dispute · themes · dates', out: 'Dispute summary and drafts for the judge', note: 'Low-risk tasks; the judge issues the orders (Art. 596º).' },
        { n: 4, name: 'Instrução', art: 'Art. 410º', input: 'The parties offer their proof.',
          agents: [
            { name: 'Themes agent', job: 'list the themes of proof (Arts. 410º, 596º)', tool: 'Retrieval' },
            { name: 'Evidence agent', job: 'retrieve and summarise law, case law and proof', tool: 'Retrieval' },
            { name: 'Matching agent', job: 'link each piece of proof to its theme', tool: 'Verification' }],
          local: 'Evidence summaries per theme', glob: 'Evidence (parties validate)', out: 'Evidence summaries, validated by the parties', note: 'The parties validate what the evidence agent summarises.' },
        { n: 5, name: 'Audiência Final', art: 'Arts. 603º-606º', input: 'The final hearing takes place.',
          agents: [
            { name: 'Transcription agent', job: 'record testimony and experts\' clarifications' },
            { name: 'Summary agent', job: 'summarise each testimony', tool: 'Drafting' },
            { name: 'Arguments agent', job: 'note the oral arguments (alegações orais)', tool: 'Drafting' }],
          local: 'Transcript and summaries', glob: 'Hearing record', out: 'Hearing record', note: 'People lead the hearing; agents only transcribe and summarise.' },
        { n: 6, name: 'Sentença', art: 'Art. 607º', kind: 'human', input: 'The whole case file, read by a person.',
          agents: [{ name: 'Human judge', job: 'parties and object → questions → proven facts → legal norms → decision (Art. 607º)', cls: 'human tall', num: '⚖' }],
          local: 'Nothing delegated', glob: 'Sentence (judge alone)', out: 'Sentence, written by the judge alone', note: 'NO MODEL: the machinery stops. The human judge decides alone.' },
        { n: 7, name: 'Apelação', art: 'Art. 644º', input: 'The sentence is appealed.',
          agents: [
            { name: 'Appealability agent', job: 'is the decision appealable? (Art. 644º)', tool: 'Retrieval' },
            { name: 'Grounds agent', job: 'draft the grounds of the appeal, or the reply', tool: 'Drafting' },
            { name: 'Verification agent', job: 'citations, deadlines, completeness', tool: 'Verification' }],
          local: 'Appeal grounds or response', glob: 'Appeal grounds', out: 'Appeal or response, ready to file', note: 'Drafting agents prepare; people decide what is filed.' },
        { n: 8, name: 'Acórdão', art: 'The input of this thesis', kind: 'thesis', input: 'The appeal court rules; the ruling is written.',
          agents: [{ name: 'LJP classifier', job: 'this thesis: reads the written ruling', cls: 'thesis', num: '8' }],
          local: 'Ruling text (retrospective)', glob: 'Ruling to classify', out: 'KEPT or ALTERED', note: 'This thesis: classifies the ruling once written.' }
      ]
    },
    html: function () {
      var nodes = '', i, rows = '';
      for (i = 0; i < 8; i++) nodes += '<div class="lv-node locked" style="left:' + (120 + i * 240) + 'px">' + (i + 1) + '</div>';
      for (i = 0; i < 4; i++) rows += '<div class="lv-row off" style="left:350px;top:' + (172 + i * 110) + 'px;width:900px"><div class="lv-num"></div><div class="lv-rcol"><div class="lv-rname"></div><div class="lv-rtag"></div></div><div class="lv-rjob"></div></div>';
      return '<div class="lv">' +
        '<div class="lv-rail-bg"></div><div class="lv-rail-fill"></div>' + nodes + '<div class="lv-raildot"></div>' +
        box({ x: 60, y: 92, w: 1100, h: 66 }, '<span class="lv-head-t"></span><span class="lv-head-art"></span>') +
        box({ x: 1000, y: 92, w: 860, h: 66, cls: 'lv-gate', style: 'display:flex;align-items:center;justify-content:flex-end;font-size:24px' }, '') +
        box({ x: 30, y: 172, w: 300, h: 210, cls: 'lv-card lv-in' }, '<span class="lv-lab">INPUT</span><div class="lv-body lv-in-body" style="font-size:26px"></div>') +
        box({ x: 30, y: 398, w: 300, h: 218, cls: 'lv-card', style: 'border-color:#394c60;background:#fff' },
          '<span class="lv-lab" style="background:#394c60">PEOPLE</span><div class="lv-body" style="font-size:25px">Parties validate the evidence. The judge decides.</div>') +
        rows +
        box({ x: 350, y: 608, w: 900, h: 112, cls: 'lv-card lv-out' }, '<span class="lv-lab">OUTPUT → next stage</span><div class="lv-body lv-out-body" style="font-size:27px"></div>') +
        box({ x: 1280, y: 172, w: 610, h: 170, cls: 'lv-card lv-local' }, '<span class="lv-lab">LOCAL MEMORY</span> <span class="lv-sub" style="display:inline">this stage only</span><div class="lv-body lv-local-body" style="font-size:26px"></div>') +
        box({ x: 1280, y: 356, w: 610, h: 360, cls: 'lv-card lv-global' }, '<span class="lv-lab">GLOBAL CASE MEMORY</span> <span class="lv-sub" style="display:inline">grows</span><div class="lv-gbar"><div class="lv-gbar-fill"></div></div><div class="lv-glist"></div>') +
        '<div class="lv-dot lv-dot-a" style="left:1262px;top:300px"></div><div class="lv-dot lv-dot-b" style="left:1262px;top:330px"></div>' +
        box({ x: 30, y: 728, w: 1860, h: 56, cls: 'lv-note' }, '') +
        box({ x: 40, y: 792, w: 1840, h: 44, cls: 'lv-cap' }, 'Illustrative approximation after LegalWorld (Fudan, 2026) with the stages of the Portuguese civil process. Not built, not evaluated; possible only with the judiciary as partner. The decision stays with the judge.') +
        '</div>';
    }
  });

  /* ======================= 32  THANK YOU ======================= */
  slides.push({
    id: 'thanks', number: null, title: 'Thank you! Questions?', layout: 'cover',
    cover: { title: 'Thank you!<br>Questions?', by: 'Helton Mendonça &nbsp;·&nbsp; Nº 56870', supervisors: '' },
    steps: [{ cue: 'Thank you' }], notes: '', sources: []
  });

  /* ==========================================================================
     APPENDIX (backup slides for Q&A). Reachable ONLY from the presenter overview (O)
     or go-to (G, then A1..A4). Never part of the normal next/previous flow.
     Each one uses a figure that already exists in the manuscript.
     ========================================================================== */

  /* ---- A1  BERT training curves ---- */
  slides.push({
    id: 'appx-bert-curves', number: null, badge: 'A1', appendix: true, title: 'Backup: BERT training curves', layout: 'custom',
    steps: [{ cue: 'Loss curves, DV and BoC' }],
    notes: [
      "Backup: loss curves of the BERT fine-tuning.",
      "Up to 20 epochs; early stopping, patience 7.",
      "Use if asked how the collapse looked in training."
    ],
    sources: [],
    html: function () {
      return '' +
        box({ x: 60, y: 24, w: 90, h: 50, cls: 'tag', style: 'font-size:30px' }, 'DV') + box({ x: 990, y: 24, w: 110, h: 50, cls: 'tag', style: 'font-size:30px' }, 'BoC') +
        img('bk_bert_loss_dv.png', 60, 84, 870, 470, 0, '', 'BERT loss curves DV') + img('bk_bert_loss_boc.png', 990, 84, 870, 470, 0, '', 'BERT loss curves BoC') +
        box({ x: 80, y: 620, w: 1760, h: 150, cls: 'caption plain', style: 'display:flex;align-items:center;justify-content:center;font-size:31px' },
          '<div>Training and validation loss curves for LegalBERTimbau fine-tuning (DV left, BoC right): up to 20 epochs, early stopping (patience 7) on raw validation loss.</div>');
    }
  });

  /* ---- A2  UMAP after fine-tuning ---- */
  slides.push({
    id: 'appx-umap', number: null, badge: 'A2', appendix: true, title: 'Backup: UMAP after fine-tuning', layout: 'custom',
    steps: [{ cue: 'UMAP, DV and BoC' }],
    notes: [
      "Backup: UMAP of embeddings after fine-tuning.",
      "DV: one elongated, one small cluster; BoC: about two clusters.",
      "Classes stay intermixed: structure follows text style, not outcome."
    ],
    sources: [],
    html: function () {
      return '' +
        box({ x: 60, y: 20, w: 90, h: 50, cls: 'tag', style: 'font-size:30px' }, 'DV') + box({ x: 990, y: 20, w: 110, h: 50, cls: 'tag', style: 'font-size:30px' }, 'BoC') +
        img('bk_umap_dv.png', 60, 76, 870, 580, 0, '', 'UMAP DV') + img('bk_umap_boc.png', 990, 76, 870, 580, 0, '', 'UMAP BoC') +
        box({ x: 80, y: 680, w: 1760, h: 140, cls: 'caption plain', style: 'display:flex;align-items:center;justify-content:center;font-size:29px' },
          '<div>UMAP 3D projection of the embedding space after fine-tuning: the outcome classes stay intermixed inside the clusters, so the learned structure reflects linguistic or structural document features, not outcomes.</div>');
    }
  });

  /* ---- A3  Few-shot retrieval pipeline (mirrors manuscript TikZ fig:fewshot-retrieval) ---- */
  slides.push({
    id: 'appx-fewshot-retrieval', number: null, badge: 'A3', appendix: true, title: 'Backup: few-shot retrieval pipeline', layout: 'custom',
    steps: [{ cue: 'Query, candidates, similarity' }, { cue: 'Rank, select, examples' }],
    notes: [
      "Backup: few-shot retrieval pipeline.",
      "Query and candidates already summarised.",
      "Rank-weighted cosine similarity; top candidate per class (k=1).",
      "Diversity filter unused (see Limitations)."
    ],
    sources: [],
    html: function () {
      var BX = function (i) { return 60 + i * 366; };
      function st(i, a, cls, t) { return box({ x: BX(i), y: 150, w: 330, h: 300, at: a, cls: 'dbox ' + cls, style: 'font-size:25px;padding:18px 20px' }, t); }
      return '' +
        st(0, 0, 'fill', '<h4>Query and Candidate Sentences</h4>Both already ranked by the smart summarization pipeline') +
        st(1, 0, '', '<h4>Rank-Weighted Cosine Similarity</h4>score(Q,C) = w<sub>Q</sub><sup>T</sup> S w<sub>C</sub>,<br>higher-ranked sentences from the summarization weighted more') +
        st(2, 1, '', '<h4>Rank Candidates per Class</h4>Same case type, grouped by decision class') +
        st(3, 1, 'salmon', '<h4>Select Top Candidate per Class</h4>k=1 per class (diversity filter unused, see Limitations)') +
        st(4, 1, 'out', '<h4>Few-Shot Examples</h4>One labeled example per class, inserted into the prompt') +
        svg(arrow(BX(0) + 334, 300, BX(1) - 4, 300, { w: 5 }) + arrow(BX(1) + 334, 300, BX(2) - 4, 300, { at: 1, w: 5 }) +
          arrow(BX(2) + 334, 300, BX(3) - 4, 300, { at: 1, w: 5 }) + arrow(BX(3) + 334, 300, BX(4) - 4, 300, { at: 1, w: 5 })) +
        box({ x: 80, y: 560, w: 1760, h: 100, cls: 'caption plain', style: 'display:flex;align-items:center;justify-content:center;font-size:30px' },
          '<div>Embedding-based few-shot retrieval: from ranked query/candidate sentences to the final selected examples.</div>');
    }
  });

  /* ---- A4  SHAP on BoC ---- */
  slides.push({
    id: 'appx-shap-boc', number: null, badge: 'A4', appendix: true, title: 'Backup: SHAP on BoC', layout: 'custom',
    steps: [{ cue: 'Top positive / negative sentences, BoC' }],
    notes: [
      "Backup: SHAP on a BoC case, top sentences.",
      "Positive and negative comparable in size.",
      "Many refer to monetary amounts in the case report.",
      "Max attribution about 2.9 points; not legal reasoning."
    ],
    sources: [],
    html: function () {
      return '' +
        img('bk_shap_boc.png', 60, 16, 790, 800, 0, '', 'SHAP top sentences BoC') +
        box({ x: 900, y: 120, w: 940, h: 560, cls: 'card', style: 'padding:36px 44px' },
          '<h3>SHAP, BREACH OF CONTRACT</h3><div class="card-body" style="font-size:32px">Top 10 positive and top 10 negative SHAP sentences (truncated) for a BoC gold-test case, predicted class UNFAVORABLE.<br><br>Unlike the DV example, both positive and negative sentences appear with comparable magnitude, and several refer to monetary claim amounts from the case report rather than to legal reasoning.</div>');
    }
  });

  /* printed corner numbers always follow the position in the main flow (cover and last slide stay unnumbered) */
  slides.forEach(function (s, i) { if (!s.appendix && s.number != null) s.number = i; });

  root.DECK = {
    meta: {
      title: 'Legal Judgement Prediction and Explanation using LLMs in Portugal',
      author: 'Helton Mendonça',
      targetMinutes: 23,
      hardStopMinutes: 30,
      /* usable:false = the file is blank/unusable, a labelled placeholder is shown instead (tools/check-deck.js verifies this flag) */
      logos: { header: { src: 'logos/LASIGE.png', usable: true }, footer: { src: 'logos/U_lisboa-fcul-fct.png', usable: true } },
      canvaRevision: 'FINAL DEFENSE - DRAFT v1 (Canva DAHVK4OXwsY), text copied 2026-10-08 plus late edits'
    },
    slides: slides
  };
  if (typeof module !== 'undefined') module.exports = root.DECK;
})(typeof window !== 'undefined' ? window : globalThis);
