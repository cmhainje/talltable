<script>
  let {
    ra = $bindable(0),
    dec = $bindable(0),
    wavemin = $bindable(1.85),
    wavemax = $bindable(1.9),
    maplvl = $bindable(6),
    alpha = $bindable(1.0),
    Q = $bindable(5.0),
    loading   = false,
    statusText = '',
    errorMsg  = null,
  } = $props();

  let raInput  = $state(ra.toFixed(4));
  let decInput = $state(dec.toFixed(4));

  // Keep inputs in sync when the map pans/zooms
  $effect(() => { raInput  = ra.toFixed(4); });
  $effect(() => { decInput = dec.toFixed(4); });

  let waveInput = $state(1.875);
  let waveRange = $state(0.05);

  $effect(() => {
    waveInput;
    waveRange;

    wavemin = Math.round(10000 * (waveInput - 0.5 * waveRange)) / 10000;
    wavemax = Math.round(10000 * (waveInput + 0.5 * waveRange)) / 10000;
  });

  $effect(() => {
    wavemin;
    wavemax;

    waveInput = Math.round(10000 * (0.5 * (wavemin + wavemax))) / 10000;
    waveRange = Math.round(10000 * (wavemax - wavemin)) / 10000;
  });

  function goTo() {
    const r = parseFloat(raInput);
    const d = parseFloat(decInput);
    if (!isNaN(r) && !isNaN(d)) {
      ra = ((r % 360) + 360) % 360;
      dec = Math.max(-90, Math.min(90, d));
    }
  }
</script>

<div class="panel">
  <div class="content">
    <h3>talltable</h3>

    <section>
      <span class="label">Coordinates</span>

      <div class="coord-grid">
        <input
          type="number"
          name="ra"
          bind:value={raInput}
          min="0"
          max="360"
          step="0.1"
        />
        <input
          type="number"
          name="dec"
          bind:value={decInput}
          min="-90"
          max="90"
          step="0.1"
        />
        <button onclick={goTo}>Go</button>
        <label class="hint" for="ra">RA [°]</label>
        <label class="hint" for="dec">Dec [°]</label>
      </div>
    </section>

    <section>
      <span class="label">
        Wavelength band ({wavemin.toFixed(4)}–{wavemax.toFixed(4)} µm)
      </span>

      <div class="row">
        <input
          type="range"
          min="0.7"
          max="5.0"
          step="0.005"
          bind:value={waveInput}
          name="wavelength"
        />
        <input type="number" name="waveInput" id="" bind:value={waveInput} />
      </div>
      <label for="wavelength" class="hint">Central wavelength [µm]</label>

      <div class="row">
        <input
          type="range"
          min="0.01"
          max="0.5"
          step="0.005"
          bind:value={waveRange}
          name="waverange"
        />
        <input type="number" name="waveRange" id="" bind:value={waveRange} />
      </div>
      <label for="waverange" class="hint">Range [µm]</label>
    </section>

    <section>
      <span class="label">Arcsinh stretch</span>
      <div class="row">
        <input type="range" min="0.1" max="10" step="0.1" bind:value={alpha} name="alpha" />
        <input type="number" bind:value={alpha} name="alphaNum" />
      </div>
      <label for="alpha" class="hint">α (brightness)</label>
      <div class="row">
        <input type="range" min="0.1" max="20" step="0.1" bind:value={Q} name="Q" />
        <input type="number" bind:value={Q} name="QNum" />
      </div>
      <label for="Q" class="hint">Q (nonlinearity; 0 → linear)</label>
    </section>

    <section>
      <label class="label" for="maplvl"
        >HEALPix level (Nside = 2<sup>{maplvl}</sup>)</label
      >
      <input
        type="range"
        min="12"
        max="15"
        step="1"
        bind:value={maplvl}
        name="maplvl"
      />
      <div class="hint">
        ~{((58.6 / Math.pow(2, maplvl)) * 60 * 60).toFixed(0)}&Prime; / pixel
      </div>
    </section>

    <section class="footer">
      <div class="help-hint">Drag to pan &middot; Scroll to zoom</div>
      <div class="status">
        {#if loading}
          <span class="dot loading"></span>
        {:else if errorMsg}
          <span class="dot error"></span>
        {:else}
          <span class="dot ok"></span>
        {/if}
        <span class="status-text">{statusText}</span>
      </div>
    </section>
  </div>
</div>

<style>
  .panel {
    position: fixed;
    top: 20px;
    right: 20px;
    background: rgba(10, 10, 20, 0.88);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    backdrop-filter: blur(8px);
    width: 260px;
    font-size: 12px;
    color: #ccc;
    overflow: hidden;
  }

  .content {
    padding: 12px 14px 14px;
  }

  h3 {
    font-size: 40px;
    line-height: 40px;
    font-style: italic;
    color: #fff;
    margin-bottom: 14px;
    letter-spacing: -0.05em;
    font-weight: 900;
  }

  section {
    margin-bottom: 14px;
  }

  .label {
    display: block;
    color: #888;
    margin-bottom: 5px;
    font-size: 11px;
  }

  input[type="number"] {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    color: #ddd;
    padding: 3px 6px;
    width: 72px;
    font-size: 11px;
    font-family: monospace;
  }

  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
  }
  input[type="number"] {
    appearance: inherit;
    -moz-appearance: textfield; /* Firefox */
  }

  input[type="range"] {
    width: 100%;
    accent-color: #f5a623;
    margin: 2px 0;
  }

  .coord-grid {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    column-gap: 12px;

    input {
      width: 100%;
    }
  }

  .row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  button {
    background: rgba(245, 166, 35, 0.2);
    border: 1px solid rgba(245, 166, 35, 0.4);
    border-radius: 3px;
    color: #f5a623;
    cursor: pointer;
    font-size: 11px;
    padding: 3px 8px;
    text-transform: uppercase;
  }
  button:hover {
    background: rgba(245, 166, 35, 0.35);
  }

  .hint {
    color: #666;
    font-size: 10px;
    margin-top: 2px;
  }

  .footer {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 10px;
    margin-bottom: 0;
  }

  .help-hint {
    color: #555;
    font-size: 10px;
    margin-bottom: 6px;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    font-family: monospace;
    color: #666;
  }

  .status-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot.loading { background: #f5a623; animation: pulse 1s infinite; }
  .dot.error   { background: #e74c3c; }
  .dot.ok      { background: #2ecc71; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.3; }
  }
</style>
