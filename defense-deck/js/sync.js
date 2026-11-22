/* ==========================================================================
   sync.js - keeps the presenter window and the audience window talking.

   THREE channels are used at the same time (the first message that arrives wins,
   duplicates are ignored by their unique id):
     1. window.postMessage between the two windows (works on file:// everywhere)
     2. BroadcastChannel (same browser, same "site")
     3. localStorage "storage" events (fallback)
   The PRESENTER owns the state. The AUDIENCE only renders what it receives.

   Message types
     {t:'state', ...}  presenter -> audience : which slide/step to show, blank, zoom
     {t:'video', a}    presenter -> audience : play/pause the demo video
     {t:'hello'}       audience -> presenter : "I just opened, please send the state"
     {t:'ping', seq}   audience -> presenter : "I'm alive" (every 2 s)
   ========================================================================== */
(function (root) {
  'use strict';
  var KEY_MSG = 'ljp-deck-msg';
  var MAGIC = 'ljp-deck-v1';
  var role = null, handler = null, peer = null, bc = null, seen = {}, seenOrder = [];
  var counter = 0;

  function uid() { counter++; return role + '-' + Date.now().toString(36) + '-' + counter + '-' + Math.random().toString(36).slice(2, 6); }
  function remember(id) {
    seen[id] = 1; seenOrder.push(id);
    if (seenOrder.length > 200) delete seen[seenOrder.shift()];
  }
  function receive(m, source) {
    if (!m || m.magic !== MAGIC || m.from === role || seen[m.id]) return;
    remember(m.id);
    if (source && role === 'presenter') peer = source;   // learn the audience window from its own message
    if (handler) handler(m);
  }

  var Sync = {
    /** role: 'presenter' | 'audience'. onMessage(msg) is called for every message from the other side. */
    init: function (r, onMessage) {
      role = r; handler = onMessage;
      root.addEventListener('message', function (e) {
        var m = e.data;
        if (typeof m === 'string') { try { m = JSON.parse(m); } catch (x) { return; } }
        receive(m, e.source);
      });
      try {
        bc = new root.BroadcastChannel('ljp-deck');
        bc.onmessage = function (e) { receive(e.data, null); };
      } catch (e) { bc = null; }
      root.addEventListener('storage', function (e) {
        if (e.key !== KEY_MSG || !e.newValue) return;
        try { receive(JSON.parse(e.newValue), null); } catch (x) {}
      });
      if (role === 'audience' && root.opener) peer = root.opener;
    },
    setPeer: function (w) { peer = w; },
    hasPeerWindow: function () { try { return !!peer && !peer.closed; } catch (e) { return false; } },
    send: function (msg) {
      msg.magic = MAGIC; msg.from = role; msg.id = uid();
      remember(msg.id);
      try { if (peer && !peer.closed) peer.postMessage(msg, '*'); } catch (e) {}
      try { if (bc) bc.postMessage(msg); } catch (e) {}
      try { root.localStorage.setItem(KEY_MSG, JSON.stringify(msg)); } catch (e) {}
    }
  };
  root.Sync = Sync;
})(window);
