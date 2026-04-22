<script>
  import Map from './Map.svelte';
  import Controls from './Controls.svelte';

  let ra = $state(98.4375);
  let dec = $state(4.9983);
  let wavemin = $state(1.85);
  let wavemax = $state(1.90);
  let maplvl = $state(12);
  let alpha  = $state(1.0);
  let Q      = $state(5.0);

  let loading   = $state(false);
  let statusText = $state('');
  let errorMsg  = $state(null);
</script>

<div class="app">
  <Map bind:ra bind:dec {wavemin} {wavemax} {maplvl} {alpha} {Q}
       bind:loading bind:statusText bind:errorMsg />
  <Controls bind:ra bind:dec bind:wavemin bind:wavemax bind:maplvl bind:alpha bind:Q
            {loading} {statusText} {errorMsg} />
</div>

<style>
  .app {
    width: 100vw;
    height: 100vh;
    position: relative;
  }
</style>
