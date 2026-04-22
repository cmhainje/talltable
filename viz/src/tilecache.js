const BUDGET = 128 * 1024 * 1024; // 128 MB

export class TileCache {
  constructor(budget = BUDGET) {
    this.budget = budget;
    this.tiles = new Map();   // Map<chunkId, TileEntry>
    this.currentKey = '';
    this._totalBytes = 0;
    this._lru = 0;
  }

  // Returns the TileEntry for chunkId, or undefined. Bumps lastUsed.
  get(chunkId) {
    const entry = this.tiles.get(chunkId);
    if (entry) entry.lastUsed = ++this._lru;
    return entry;
  }

  // Returns true if chunkId is in cache (without updating lastUsed).
  has(chunkId) {
    return this.tiles.has(chunkId);
  }

  // Stores a tile. Bumps LRU, adds bytes, then evicts until under budget.
  put(chunkId, mappix, flux) {
    const existing = this.tiles.get(chunkId);
    if (existing) {
      this._totalBytes -= existing.bytes;
    }

    const bytes = 4 * mappix.length + 4 * flux.length; // mappix is 2×uint32 per pixel
    const entry = { chunkId, mappix, flux, bytes, lastUsed: ++this._lru };
    this.tiles.set(chunkId, entry);
    this._totalBytes += bytes;

    // Evict LRU entries until under budget
    while (this._totalBytes > this.budget && this.tiles.size > 0) {
      let oldest = null;
      for (const e of this.tiles.values()) {
        if (oldest === null || e.lastUsed < oldest.lastUsed) {
          oldest = e;
        }
      }
      if (oldest) {
        this._totalBytes -= oldest.bytes;
        this.tiles.delete(oldest.chunkId);
      }
    }
  }

  // Change config. If key differs from current, wipe all tiles.
  setConfig(key) {
    if (key !== this.currentKey) {
      this.tiles.clear();
      this._totalBytes = 0;
      this.currentKey = key;
    }
  }

  // Total bytes currently cached.
  get totalBytes() {
    return this._totalBytes;
  }
}
