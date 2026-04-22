<script>
  import { onMount, untrack } from 'svelte';
  import * as d3 from 'd3';
  import { init, uploadPixels, render } from './renderer.js';
  import { TileCache } from './tilecache.js';

  let {
    ra = $bindable(), dec = $bindable(),
    wavemin, wavemax, maplvl,
    alpha, Q,
    loading = $bindable(false),
    statusText = $bindable(''),
    errorMsg = $bindable(null),
  } = $props();

  let canvasEl, svgEl;

  // View state — plain mutable vars, not reactive; keeps pan/zoom out of Svelte
  let viewRa    = $state(ra);
  let viewDec   = $state(dec);
  let viewScale = 50000;

  // 10 arcsec/pixel in px/rad: 1 / (10 * π / (3600 * 180)) ≈ 20626
  const MIN_SCALE = 206265 / 10;

  let rendCtx = null;

  // When ra/dec are changed externally (e.g. Go button), push into view state.
  // untrack prevents subscribing to viewRa/viewDec so this doesn't fire during pan.
  $effect(() => {
    const newRa = ra, newDec = dec;
    if (newRa !== untrack(() => viewRa) || newDec !== untrack(() => viewDec)) {
      viewRa = newRa;
      viewDec = newDec;
      redraw();
      sendRequest();
    }
  });

  // ── Tile cache ─────────────────────────────────────────────────────────────

  const cache = new TileCache();

  function configKey() {
    return `${maplvl}|${wavemin.toFixed(4)}|${wavemax.toFixed(4)}`;
  }

  // ── Rendering ──────────────────────────────────────────────────────────────

  function redraw() {
    if (!rendCtx) return;
    render(rendCtx, viewRa, viewDec, viewScale, maplvl, alpha, Q);
    updateGraticule();
    updateScaleBar();
  }

  // Assembles cached tiles into a merged pixel buffer and uploads to GPU.
  // chunksAll must be sorted ascending (guaranteed by /chunks response).
  // mappix is stored as interleaved Uint32Array [lo0,hi0, lo1,hi1, ...] (2 u32 per pixel).
  function compose(chunksAll) {
    const parts = [];
    let nPix = 0;
    for (const id of chunksAll) {
      const t = cache.get(id);
      if (t) { parts.push(t); nPix += t.flux.length; }
    }
    if (nPix === 0) {
      uploadPixels(rendCtx, new Uint32Array(0), new Float32Array(0));
      return 0;
    }
    const mappix = new Uint32Array(nPix * 2);
    const flux   = new Float32Array(nPix);
    let mOff = 0, fOff = 0;
    for (const t of parts) {
      mappix.set(t.mappix, mOff);
      flux.set(t.flux, fOff);
      mOff += t.mappix.length;
      fOff += t.flux.length;
    }
    uploadPixels(rendCtx, mappix, flux);
    return nPix;
  }

  // ── Graticule (D3 direct DOM manipulation, no Svelte reactivity) ───────────

  function adaptiveStep() {
    const fov = Math.max(canvasEl.width, canvasEl.height) / viewScale * (180 / Math.PI);
    if (fov < 1)  return 0.25;
    if (fov < 3)  return 0.5;
    if (fov < 10) return 1;
    if (fov < 30) return 5;
    return 15;
  }

  function updateGraticule() {
    if (!svgEl) return;
    const proj = d3.geoGnomonic()
      .rotate([-viewRa, -viewDec])
      .reflectX(true)
      .scale(viewScale)
      .translate([canvasEl.width / 2, canvasEl.height / 2])
      .clipAngle(60);
    const grat = d3.geoGraticule().step([adaptiveStep(), adaptiveStep()])();
    const pathD = d3.geoPath(proj)(grat) ?? '';
    let el = svgEl.querySelector('path');
    if (!el) {
      el = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      el.setAttribute('fill', 'none');
      el.setAttribute('stroke', 'rgba(255,255,255,0.15)');
      el.setAttribute('stroke-width', '0.5');
      svgEl.appendChild(el);
    }
    el.setAttribute('d', pathD);
  }

  function updateScaleBar() {
    if (!svgEl || !canvasEl) return;

    // Target ~90px per displayed unit; naturalUnitDeg is the ideal unit size in degrees.
    const naturalUnitDeg = 90 / (viewScale * Math.PI / 180);

    let val, unitLabel, unitDeg;
    if (naturalUnitDeg >= 0.5) {
      const nice = [1, 2, 5, 10, 20, 30, 45, 60, 90];
      val = nice.reduce((a, b) =>
        Math.abs(b - naturalUnitDeg) < Math.abs(a - naturalUnitDeg) ? b : a);
      unitLabel = '°'; unitDeg = val;
    } else if (naturalUnitDeg * 60 >= 1) {
      const nat = naturalUnitDeg * 60;
      const nice = [1, 2, 5, 10, 15, 20, 30];
      val = nice.reduce((a, b) => Math.abs(b - nat) < Math.abs(a - nat) ? b : a);
      unitLabel = '′'; unitDeg = val / 60;
    } else {
      const nat = naturalUnitDeg * 3600;
      const nice = [1, 2, 5, 10, 20, 30, 60];
      val = nice.reduce((a, b) => Math.abs(b - nat) < Math.abs(a - nat) ? b : a);
      unitLabel = '″'; unitDeg = val / 3600;
    }

    const pxPerUnit = unitDeg * (Math.PI / 180) * viewScale;
    const W = canvasEl.width;
    const H = canvasEl.height;
    const MX = 20, MY = 30;
    const barY  = H - MY;
    const xRight = W - MX;
    const xLeft  = xRight - 2 * pxPerUnit;
    const xMid   = xLeft + pxPerUnit;
    const TICK   = 5;
    const labelY = barY + TICK + 11;

    const ns = 'http://www.w3.org/2000/svg';
    let g = svgEl.querySelector('.scale-bar');
    if (!g) {
      g = document.createElementNS(ns, 'g');
      g.setAttribute('class', 'scale-bar');
      svgEl.appendChild(g);
    }
    g.innerHTML = '';

    const COLOR = 'rgba(255,255,255,0.7)';
    const textBase = `fill:${COLOR};font-size:10px;font-family:monospace;dominant-baseline:middle;`;

    function seg(x1, y1, x2, y2) {
      const el = document.createElementNS(ns, 'line');
      el.setAttribute('x1', x1); el.setAttribute('y1', y1);
      el.setAttribute('x2', x2); el.setAttribute('y2', y2);
      el.setAttribute('stroke', COLOR);
      el.setAttribute('stroke-width', '1.5');
      el.setAttribute('stroke-linecap', 'square');
      g.appendChild(el);
    }

    function label(x, str, anchor = 'middle') {
      const el = document.createElementNS(ns, 'text');
      el.setAttribute('x', x); el.setAttribute('y', labelY);
      el.setAttribute('style', textBase + `text-anchor:${anchor};`);
      el.textContent = str;
      g.appendChild(el);
    }

    // Horizontal bar
    seg(xLeft, barY, xRight, barY);
    // End caps (full height tick)
    seg(xLeft,  barY - TICK, xLeft,  barY + TICK);
    seg(xRight, barY - TICK, xRight, barY + TICK);
    // Mid tick (half height)
    seg(xMid, barY - TICK * 0.6, xMid, barY + TICK * 0.6);

    label(xMid,  `${val}${unitLabel}`);
    label(xRight, `${val * 2}${unitLabel}`);
  }

  // ── WebSocket ──────────────────────────────────────────────────────────────

  let ws, sendTimer, wsTimer;
  let reqSeq = 0;   // increments on every send; response is dropped if seq is stale
  let pendingSeq = 0;

  let chunksAbort = null;   // AbortController for in-flight /chunks fetch
  let pendingChunks = null; // chunksAll from the last /chunks response that triggered a WS send

  // ── Mid-pan cache renders ──────────────────────────────────────────────────

  const PAN_FPS = 20;
  const PAN_INTERVAL_MS = 1000 / PAN_FPS;

  let panTimer = null;
  let lastPanRender = 0;
  let fpsTimes = [];
  let fpsHideTimer = null;

  // Upload all cached tiles and redraw. No network request — the shader clips
  // to the viewport, so off-screen pixels are simply never drawn.
  function renderCached() {
    if (!rendCtx) return;
    const tiles = [...cache.tiles.values()];
    if (tiles.length === 0) return;
    let nPix = 0;
    for (const t of tiles) nPix += t.flux.length;
    const mappix = new Uint32Array(nPix * 2);
    const flux   = new Float32Array(nPix);
    let mOff = 0, fOff = 0;
    for (const t of tiles) {
      mappix.set(t.mappix, mOff); flux.set(t.flux, fOff);
      mOff += t.mappix.length; fOff += t.flux.length;
    }
    uploadPixels(rendCtx, mappix, flux);
    redraw();
    updateFpsDisplay();
  }

  // Fire at most once per PAN_INTERVAL_MS; if called again before the interval
  // is up, the scheduled call fires at the right time (not immediately).
  function scheduleRenderCached() {
    if (panTimer !== null) return;
    const elapsed = performance.now() - lastPanRender;
    const delay = Math.max(0, PAN_INTERVAL_MS - elapsed);
    panTimer = setTimeout(() => {
      panTimer = null;
      lastPanRender = performance.now();
      renderCached();
    }, delay);
  }

  function updateFpsDisplay() {
    if (!svgEl || !canvasEl) return;
    const now = performance.now();
    fpsTimes.push(now);
    if (fpsTimes.length > 10) fpsTimes.shift();
    if (fpsTimes.length < 2) return;
    const fps = Math.round(1000 * (fpsTimes.length - 1) / (fpsTimes.at(-1) - fpsTimes[0]));
    const ns = 'http://www.w3.org/2000/svg';
    let el = svgEl.querySelector('.fps');
    if (!el) {
      el = document.createElementNS(ns, 'text');
      el.setAttribute('class', 'fps');
      el.setAttribute('style', 'fill:rgba(255,255,255,0.35);font-size:10px;font-family:monospace;text-anchor:start;');
      svgEl.appendChild(el);
    }
    el.setAttribute('x', 12);
    el.setAttribute('y', canvasEl.height - 12);
    el.textContent = `${fps} fps`;

    clearTimeout(fpsHideTimer);
    fpsHideTimer = setTimeout(() => { fpsTimes = []; el.textContent = ''; }, 5000);
  }

  // ── Full request (on gesture end / config change) ──────────────────────────

  async function sendRequest() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (panTimer !== null) { clearTimeout(panTimer); panTimer = null; }
    chunksAbort?.abort();
    chunksAbort = new AbortController();
    const thisAbort = chunksAbort;
    clearTimeout(wsTimer);

    const fov = Math.max(canvasEl.width, canvasEl.height) / viewScale * (180 / Math.PI);
    loading = true;
    statusText = 'fetching chunks…';

    let chunksAll;
    try {
      const res = await fetch(
        `/chunks?ra=${viewRa}&dec=${viewDec}&width=${fov}&height=${fov}`,
        { signal: thisAbort.signal }
      );
      chunksAll = (await res.json()).chunks;
    } catch (e) {
      if (e.name === 'AbortError') return;
      errorMsg = 'chunks fetch error';
      loading = false;
      return;
    }

    cache.setConfig(configKey());
    pendingChunks = chunksAll;

    const missing = chunksAll.filter(id => !cache.has(id));

    // Render whatever is already cached — no waiting for missing data
    const n = compose(chunksAll);
    redraw();

    if (missing.length === 0) {
      loading = false;
      errorMsg = null;
      statusText = `${n} pixels (cached)`;
      return;
    }

    statusText = `loading… ${n} pixels`;

    // Debounce only the WS request so rapid panning doesn't spam the server
    wsTimer = setTimeout(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const seq = ++reqSeq;
      pendingSeq = seq;
      ws.send(JSON.stringify({ chunks: missing, maplvl, wavemin, wavemax, seq }));
      statusText = `fetching ${missing.length} chunks…`;
    }, 300);
  }

  function scheduleFetch() {
    // Used for config changes — debounce the whole pipeline including /chunks
    if (panTimer !== null) { clearTimeout(panTimer); panTimer = null; }
    clearTimeout(sendTimer);
    clearTimeout(wsTimer);
    sendTimer = setTimeout(sendRequest, 300);
  }

  function openSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/tiles`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => { errorMsg = null; sendRequest(); };

    ws.onmessage = ({ data }) => {
      const view    = new DataView(data);
      const seq     = view.getUint32(0, true);
      const flags   = view.getUint32(4, true);
      const isFinal = (flags & 1) !== 0;
      if (seq !== pendingSeq) return;

      const nTiles = view.getUint32(8, true);
      let offset = 12;

      for (let t = 0; t < nTiles; t++) {
        const chunkId = view.getUint32(offset, true); offset += 4;
        const nPix    = view.getUint32(offset, true); offset += 4;
        const mappix  = new Uint32Array(data.slice(offset, offset + nPix * 8));  offset += nPix * 8;
        const flux    = new Float32Array(data.slice(offset, offset + nPix * 4)); offset += nPix * 4;
        cache.put(chunkId, mappix, flux);
      }

      // Re-render on every message so tiles appear as they stream in.
      const n = compose(pendingChunks ?? []);
      redraw();

      if (isFinal) {
        loading = false;
        errorMsg = null;
        statusText = `${n} pixels`;
      } else {
        statusText = `loading… ${n} pixels`;
      }
    };

    ws.onerror = () => { errorMsg = 'ws error'; loading = false; statusText = 'ws error'; };
    ws.onclose = () => setTimeout(openSocket, 1000);
  }

  // Re-fetch when wavelength or level controls change
  $effect(() => {
    void wavemin; void wavemax; void maplvl;
    scheduleFetch();
  });

  // Stretch params only need a redraw — no new data required
  $effect(() => {
    void alpha; void Q;
    redraw();
  });

  // ── Mount ──────────────────────────────────────────────────────────────────

  onMount(() => {
    canvasEl.width  = window.innerWidth;
    canvasEl.height = window.innerHeight;

    // Initialise WebGL renderer
    try {
      rendCtx = init(canvasEl);
    } catch (e) {
      errorMsg = e.message;
      statusText = e.message;
      console.error(e);
      return;
    }

    openSocket();

    // ── D3 zoom / pan ──
    // The transform is reset to identity after every gesture so that t.x/t.y
    // always represent deltas from the start of the current gesture only.
    // Without this, accumulated t.x/t.y from past gestures corrupt the
    // baseProj.invert() call and cause jumps on each new gesture.
    let baseRa, baseDec, baseScale, baseProj;
    let resetting = false;

    const zoom = d3.zoom()
      .scaleExtent([0.001, 1e8])
      .on('start', () => {
        if (resetting) return;
        baseRa    = viewRa;
        baseDec   = viewDec;
        baseScale = viewScale;
        baseProj  = d3.geoGnomonic()
          .rotate([-baseRa, -baseDec])
          .reflectX(true)
          .scale(baseScale)
          .translate([canvasEl.width / 2, canvasEl.height / 2])
          .clipAngle(60);
      })
      .on('zoom', (event) => {
        if (resetting) return;
        scheduleRenderCached();
        const t = event.transform;
        viewScale = Math.max(MIN_SCALE, baseScale * t.k);
        const effectiveK = viewScale / baseScale;

        // For a zoom gesture D3 sets t.x = cx*(1 - t.k). If the scale was
        // clamped we only applied effectiveK, so scale the translations to
        // match. Pure pans have t.k === 1 and need no adjustment.
        let adjX = t.x, adjY = t.y;
        if (t.k !== 1) {
          const ratio = (1 - effectiveK) / (1 - t.k);
          adjX = t.x * ratio;
          adjY = t.y * ratio;
        }

        const inv = baseProj.invert([
          (canvasEl.width  / 2 - adjX) / effectiveK,
          (canvasEl.height / 2 - adjY) / effectiveK,
        ]);
        if (inv) { viewRa = inv[0]; viewDec = inv[1]; }
        redraw();
      })
      .on('end', () => {
        if (resetting) return;
        ra  = viewRa;
        dec = viewDec;
        sendRequest();
        resetting = true;
        d3.select(canvasEl).call(zoom.transform, d3.zoomIdentity);
        resetting = false;
      });

    d3.select(canvasEl).call(zoom);

    // ── Resize ──
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      canvasEl.width  = width;
      canvasEl.height = height;
      svgEl.setAttribute('width',  width);
      svgEl.setAttribute('height', height);
      redraw();
    });
    ro.observe(canvasEl);

    redraw();

    return () => { ro.disconnect(); ws?.close(); };
  });
</script>

<!-- WebGL canvas for healpix pixels -->
<canvas bind:this={canvasEl}></canvas>

<!-- SVG overlay for graticule (pointer-events: none so zoom still works) -->
<svg bind:this={svgEl} style="position:fixed;top:0;left:0;pointer-events:none" />

<style>
  canvas {
    display: block;
    width: 100vw;
    height: 100vh;
    cursor: grab;
  }
  canvas:active { cursor: grabbing; }
</style>
