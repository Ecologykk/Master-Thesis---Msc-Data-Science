# How to open and run the defense deck

Written for someone who has never built a web page. No installation, no internet, no server.

## 1. The 30-second version

1. Open the folder `defense-deck`.
2. Double-click **`index.html`**. It opens in your browser (use **Microsoft Edge** or **Google Chrome**).
   This window is the **PRESENTER window** (your laptop screen): notes, timer, previews.
3. Press the **`A`** key (or click "Open audience window"). A second, smaller window opens: the **AUDIENCE window**.
   If nothing opens, your browser blocked the pop-up: click the small "pop-up blocked" icon in the address bar,
   choose "Always allow", and press `A` again.
4. Drag the audience window to the projector screen, click once inside it and press **`F`** (or double-click it)
   to make it full screen. Press `F` again (or `Esc`) to leave full screen.
5. Click once on the presenter window (so it has the keyboard) and use the arrow keys / clicker.

Git Bash alternative (same result):

```bash
cd "/c/Users/helto/OneDrive/Documentos/Mestrado Ciência de Dados/Tese de Mestrado"
start defense-deck/index.html                                   # default browser
start msedge "$(cygpath -w "$PWD/defense-deck/index.html")"    # force Edge
```

## 2. Two screens (projector)

- Windows key + **P** -> choose **Extend** (not "Duplicate"). Do this BEFORE pressing `A`.
- If the room only allows **Duplicate** (same picture on both): use **solo mode**, open
  `defense-deck/index.html#solo` (add `#solo` at the end of the address). One window, full screen with `F11`,
  keys work in that same window; you lose the presenter panel (notes, timer). Keep your printed notes or the Canva PDF with notes.

## 3. Keyboard cheat sheet (presenter window)

| Key | What it does |
|---|---|
| `→`  `Space`  `PageDown` | next step (reveals the next piece; at the end of a slide, goes to the next slide) |
| `←`  `PageUp` | previous step (the piece disappears again from the audience screen) |
| `Shift + →` / `Shift + ←` | next / previous **slide** (skips the remaining steps) |
| `B` or `.` | **blank**: audience screen goes black (presenter keeps working). Press again to return |
| `F` | **freeze**: the audience keeps seeing what it sees now; you can browse privately. `F` again = audience catches up |
| `Esc` twice quickly | **panic**: unblank + unfreeze + audience goes back to the last position it was showing |
| `G` | go to slide: type a number (0 to 33), or `A1`..`A4` for the appendix, then Enter |
| `O` | overview of all slides + appendix (click a slide to jump). `Esc` closes it |
| `V` | play / pause the demo video on slide 21 (Judge Validation). A status chip in the presenter bar shows "Video: playing / paused / BLOCKED". Click once inside the audience window beforehand so the sound is allowed; you can also click the video itself in the audience window |
| `Z` | zoom the audience screen into the point under your mouse (hover the NOW preview first). `Z` again = zoom out |
| `N` | bigger / smaller speaker notes |
| `T` / `Shift+T` | start-pause / reset the timer (turns amber at 90% of 23 min, red after) |
| `A` | open (or re-connect) the audience window |
| `R` | return from an appendix slide to where you were |
| `Home` | back to the cover (also resets blank/freeze) |
| `?` | help |

Most USB clickers send PageUp/PageDown: both work. In the **audience** window the keyboard and mouse do nothing
(on purpose), except `F` / double-click = full screen.

Links (sources) are listed under the NOW preview in the presenter window; clicking one opens it in the
presenter's browser, never on the projector. Links on the slides themselves are inert in the audience window.

## 4. Reset / start again

- Reset the timer: `Shift+T`.
- Go back to the beginning: press `Home`. Blank/freeze/zoom are cleared.
- Full reset (forget position and timer): close both windows and open `index.html` again, then press `Home` and `Shift+T`.
  (The deck remembers the last position in the browser, so reloading a window never loses your place.)

## 5. If the two windows lose sync

The status chip at the top of the presenter window says **"Audience: connected"** (green) or **"not connected"**.

1. Press **`A`** in the presenter window: it re-connects to the existing audience window (or opens a new one).
2. If you closed the audience window by accident: press `A`.
3. If you reloaded the presenter window (`F5`): nothing is lost; the audience re-attaches within 2 seconds.
4. If it still does not work: press `B` (black screen), close everything, open `index.html` again, press `A`.
5. Emergency: `index.html#solo` (one window), or open the Canva PDF.

## 6. Export a PDF (backup)

Open `defense-deck/index.html?print` (Git Bash: `start "defense-deck/index.html?print"` may not accept the `?`, so
instead type the address in the browser: `file:///C:/Users/helto/OneDrive/Documentos/Mestrado Ciência de Dados/Tese de Mestrado/defense-deck/index.html?print`).
All steps are visible, one slide per page, the 4 appendix slides at the end. Press **Ctrl + P**, choose
**Save as PDF**, Layout **Landscape**, Margins **None**, tick **Background graphics**. This is a bonus; the official PDF
fallback is the Canva export.

## 7. Optional checker (needs Node, which is installed on your laptop)

```bash
cd "/c/Users/helto/OneDrive/Documentos/Mestrado Ciência de Dados/Tese de Mestrado"
node defense-deck/tools/check-deck.js          # expect: "38 slides, 135 reveal states, 0 errors, 0 warnings" (34 main + 4 appendix)
node defense-deck/tools/check-deck.js --text   # also prints the text of every slide (to compare with Canva)
```

## 8. Where things are

```
defense-deck/
  index.html            the only page (presenter by default, #audience, #solo, ?print)
  data/slides.js        ALL the slide content (text, notes, links, reveal steps): edit text here
  js/                   the engine (you do not need to read it to present)
  css/                  look (live.css = the animated slide): tokens.css = colours and font, deck.css = slides, presenter.css = laptop view
  assets/fonts          Open Sans (OFL licence), bundled so it works offline
  assets/img, logos, video   pictures; see ASSETS_TODO.md for what is still a placeholder
  tools/check-deck.js   optional checker
```

Changing a word on a slide: open `data/slides.js` in any text editor, find the sentence, change it, save, press F5
in both windows. Keep the surrounding quotes and `<b>`...`</b>` tags intact.

## 9. Before the defense (room checklist)

- Copy the whole `defense-deck` folder to the laptop Desktop AND to a USB stick; also put the Canva PDFs there.
- Test with Wi-Fi OFF. Test `Win+P` -> Extend, press `A`, allow pop-ups, `F` for full screen in the audience window.
- Click once inside the audience window before the video slide, so the browser allows sound.
- Know the panic key: `B` (black), then open the Canva PDF if needed.


## 10. Speaker notes and logos (changed)

- Notes are short bullet lists (3-7 per slide). They show in the presenter window, right column, under the NEXT preview. `N` makes them bigger/smaller.
- Logos: the deck looks for `logos/LASIGE.png` (header) and `logos/U_lisboa-fcul-fct.png` (footer) next to `index.html`. While a file is flagged `usable: false` in `data/slides.js` (`meta.logos`) a labelled placeholder is drawn. After replacing a file with a working logo (white artwork on transparent background, for the maroon bands) set `usable: true` there and run the checker.
