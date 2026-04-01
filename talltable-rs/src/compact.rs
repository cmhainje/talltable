use arrow::array::{Float32Array, Int32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bytemuck;
use glob::glob;
use num_cpus;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::io::BufWriter;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

const HP_HIGH_LEVEL: u32 = 22;
const MAX_ROWS_PER_PART: usize = 200_000_000;
const PART_MAX_LEVEL: u32 = 10;
const PIXEL_DB_PATH: &str = "/mnt/sdceph/users/chainje/spxdb-test/pixels";
const BATCH_SIZE: usize = 1_048_576;

fn part_to_level_index(part: u32) -> (u32, u32) {
    let n = part.checked_ilog2().unwrap_or(0);
    let level = (n / 2) - 4;
    let index = part - (1 << n);
    (level, index)
}

fn level_index_to_part(level: u32, index: u32) -> u32 {
    (1 << (2 * (level + 4))) + index
}

fn int_env(key: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(key) {
        Ok(val) => Ok(val.parse::<usize>()?),
        Err(_) => Ok(default),
    }
}

// ---------------------------------------------------------------------------
// ColumnData: holds the 7 column Vecs together
// ---------------------------------------------------------------------------

struct ColumnData {
    flux: Vec<f32>,
    variance: Vec<f32>,
    zodi: Vec<f32>,
    flags: Vec<i32>,
    hphigh: Vec<i64>,
    waveid: Vec<i32>,
    imageid: Vec<i64>,
}

impl ColumnData {
    fn with_capacity(n: usize) -> Self {
        Self {
            flux: Vec::with_capacity(n),
            variance: Vec::with_capacity(n),
            zodi: Vec::with_capacity(n),
            flags: Vec::with_capacity(n),
            hphigh: Vec::with_capacity(n),
            waveid: Vec::with_capacity(n),
            imageid: Vec::with_capacity(n),
        }
    }

    fn len(&self) -> usize {
        self.hphigh.len()
    }

    fn extend_from_batch(&mut self, batch: &RecordBatch) {
        macro_rules! extend_col {
            ($field:ident, $name:literal, $arrow_ty:ty) => {
                self.$field.extend(
                    batch
                        .column_by_name($name)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<$arrow_ty>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied(),
                );
            };
        }
        extend_col!(flux, "flux", Float32Array);
        extend_col!(variance, "variance", Float32Array);
        extend_col!(zodi, "zodi", Float32Array);
        extend_col!(flags, "flags", Int32Array);
        extend_col!(hphigh, "hphigh", Int64Array);
        extend_col!(waveid, "waveid", Int32Array);
        extend_col!(imageid, "imageid", Int64Array);
    }

    fn read_from_bin(
        &mut self,
        f: &std::fs::File,
        header_size: usize,
        file_num_rows: usize,
        start: usize,
        end: usize,
    ) -> anyhow::Result<()> {
        let mut off = 0usize;
        read_bin_column_into::<f32>(f, header_size, off, start, end, &mut self.flux)?;
        off += file_num_rows * 4;
        read_bin_column_into::<f32>(f, header_size, off, start, end, &mut self.variance)?;
        off += file_num_rows * 4;
        read_bin_column_into::<f32>(f, header_size, off, start, end, &mut self.zodi)?;
        off += file_num_rows * 4;
        read_bin_column_into::<i32>(f, header_size, off, start, end, &mut self.flags)?;
        off += file_num_rows * 4;
        read_bin_column_into::<i64>(f, header_size, off, start, end, &mut self.hphigh)?;
        off += file_num_rows * 8;
        read_bin_column_into::<i32>(f, header_size, off, start, end, &mut self.waveid)?;
        off += file_num_rows * 4;
        read_bin_column_into::<i64>(f, header_size, off, start, end, &mut self.imageid)?;
        Ok(())
    }

    /// Push a single row from another ColumnData.
    #[inline]
    fn push_row(&mut self, other: &ColumnData, i: usize) {
        self.flux.push(other.flux[i]);
        self.variance.push(other.variance[i]);
        self.zodi.push(other.zodi[i]);
        self.flags.push(other.flags[i]);
        self.hphigh.push(other.hphigh[i]);
        self.waveid.push(other.waveid[i]);
        self.imageid.push(other.imageid[i]);
    }

    fn is_sorted_by_hphigh(&self) -> bool {
        self.hphigh.windows(2).all(|w| w[0] <= w[1])
    }

    fn sort_by_hphigh(&mut self) {
        let n = self.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| self.hphigh[i].cmp(&self.hphigh[j]));

        fn apply_permutation<T: Copy>(v: Vec<T>, indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&i| v[i]).collect()
        }
        self.flux = apply_permutation(std::mem::take(&mut self.flux), &indices);
        self.variance = apply_permutation(std::mem::take(&mut self.variance), &indices);
        self.zodi = apply_permutation(std::mem::take(&mut self.zodi), &indices);
        self.flags = apply_permutation(std::mem::take(&mut self.flags), &indices);
        self.hphigh = apply_permutation(std::mem::take(&mut self.hphigh), &indices);
        self.waveid = apply_permutation(std::mem::take(&mut self.waveid), &indices);
        self.imageid = apply_permutation(std::mem::take(&mut self.imageid), &indices);
    }

    fn into_record_batch(self, schema: &Arc<Schema>) -> anyhow::Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Float32Array::from(self.flux)),
                Arc::new(Float32Array::from(self.variance)),
                Arc::new(Float32Array::from(self.zodi)),
                Arc::new(Int32Array::from(self.flags)),
                Arc::new(Int64Array::from(self.hphigh)),
                Arc::new(Int32Array::from(self.waveid)),
                Arc::new(Int64Array::from(self.imageid)),
            ],
        )?)
    }
}

// ---------------------------------------------------------------------------
// Binary chunk I/O (unchanged)
// ---------------------------------------------------------------------------

/// Read the binary chunk header and return (num_partitions, part_ids, part_starts, part_ends, num_rows, header_size).
fn read_chunk_header(
    f: &mut std::fs::File,
) -> anyhow::Result<(usize, Vec<u32>, Vec<u64>, Vec<u64>, usize, usize)> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    let num_part = u32::from_le_bytes(buf4) as usize;

    let mut id_buf = vec![0u8; num_part * 4];
    f.read_exact(&mut id_buf)?;
    let part_ids: Vec<u32> = bytemuck::cast_slice(&id_buf).to_vec();

    let mut start_buf = vec![0u8; num_part * 8];
    f.read_exact(&mut start_buf)?;
    let part_starts: Vec<u64> = bytemuck::cast_slice(&start_buf).to_vec();

    let mut end_buf = vec![0u8; num_part * 8];
    f.read_exact(&mut end_buf)?;
    let part_ends: Vec<u64> = bytemuck::cast_slice(&end_buf).to_vec();

    f.read_exact(&mut buf8)?;
    let num_rows = u64::from_le_bytes(buf8) as usize;

    let header_size = 4 + num_part * 20 + 8;
    Ok((num_part, part_ids, part_starts, part_ends, num_rows, header_size))
}

/// Read a typed column slice from a binary chunk file directly into a Vec.
/// Uses pread (positional read) — no seeking, no file position state.
fn read_bin_column_into<T: bytemuck::Pod>(
    f: &std::fs::File,
    header_size: usize,
    col_byte_offset: usize,
    start: usize,
    end: usize,
    dest: &mut Vec<T>,
) -> anyhow::Result<()> {
    let elem_size = std::mem::size_of::<T>();
    let offset = (header_size + col_byte_offset + start * elem_size) as u64;
    let count = end - start;
    let old_len = dest.len();
    dest.reserve(count);
    // SAFETY: we're extending into the spare capacity, then reading bytes into it.
    // read_exact_at will fill exactly count * elem_size bytes or return an error.
    unsafe {
        let spare = std::slice::from_raw_parts_mut(
            dest.as_mut_ptr().add(old_len) as *mut u8,
            count * elem_size,
        );
        f.read_exact_at(spare, offset)?;
        dest.set_len(old_len + count);
    }
    Ok(())
}

fn scan_chunk_files() -> anyhow::Result<HashMap<u32, Vec<(PathBuf, usize, usize)>>> {
    let bin_files = glob(&format!("{}/chunk_*.bin", PIXEL_DB_PATH))?;
    let mut part_index: HashMap<u32, Vec<(PathBuf, usize, usize)>> = HashMap::new();

    for fpath in bin_files {
        let fpath = match fpath {
            Ok(p) => p,
            Err(e) => {
                eprintln!("warning: skipping glob entry: {}", e);
                continue;
            }
        };

        let result = (|| -> anyhow::Result<()> {
            let mut f = std::fs::File::open(&fpath)?;
            let (_, part_ids, part_starts, part_ends, _, _) = read_chunk_header(&mut f)?;

            for ((&pid, &start), &end) in part_ids
                .iter()
                .zip(part_starts.iter())
                .zip(part_ends.iter())
            {
                part_index
                    .entry(pid)
                    .or_default()
                    .push((fpath.clone(), start as usize, end as usize));
            }
            Ok(())
        })();

        if let Err(e) = result {
            eprintln!("warning: skipping {}: {}", fpath.display(), e);
        }
    }

    Ok(part_index)
}

fn stride_partitions(
    mut part_index: HashMap<u32, Vec<(PathBuf, usize, usize)>>,
    task_id: usize,
    num_tasks: usize,
) -> HashMap<u32, Vec<(PathBuf, usize, usize)>> {
    let mut keys: Vec<u32> = part_index.keys().copied().collect();
    keys.sort();

    let keys_oi: HashSet<u32> = (task_id..keys.len())
        .step_by(num_tasks)
        .map(|i| keys[i])
        .collect();

    part_index.extract_if(|k, _v| keys_oi.contains(k)).collect()
}

// ---------------------------------------------------------------------------
// Schema / writer helpers
// ---------------------------------------------------------------------------

fn pixel_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("flux", DataType::Float32, false),
        Field::new("variance", DataType::Float32, false),
        Field::new("zodi", DataType::Float32, false),
        Field::new("flags", DataType::Int32, false),
        Field::new("hphigh", DataType::Int64, false),
        Field::new("waveid", DataType::Int32, false),
        Field::new("imageid", DataType::Int64, false),
    ]))
}

fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(1_048_576)) // 1M rows (default)
        .build()
}

fn open_writer(path: &str, schema: &Arc<Schema>) -> anyhow::Result<ArrowWriter<BufWriter<std::fs::File>>> {
    let file = BufWriter::with_capacity(4 * 1024 * 1024, std::fs::File::create(path)?);
    Ok(ArrowWriter::try_new(file, Arc::clone(schema), Some(writer_props()))?)
}

// ---------------------------------------------------------------------------
// K-way block merge
// ---------------------------------------------------------------------------

/// Fused k-way merge: merges sorted runs directly into output, element-wise.
///
/// Uses a min-heap to always pick the run with the smallest current hphigh,
/// then copies that single row to the output. All heap operations are O(log K)
/// where K is the number of runs — the heap is tiny and stays in L1 cache.
/// All writes to `out` are sequential (push to pre-allocated Vecs).
fn merge_runs(runs: &[ColumnData], total_rows: usize) -> ColumnData {
    let mut out = ColumnData::with_capacity(total_rows);
    if total_rows == 0 {
        return out;
    }

    // min-heap of (hphigh_value, run_index)
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut cursors: Vec<usize> = vec![0; runs.len()];

    for (i, run) in runs.iter().enumerate() {
        if !run.hphigh.is_empty() {
            heap.push(Reverse((run.hphigh[0], i)));
        }
    }

    while let Some(Reverse((_, winner))) = heap.pop() {
        let cur = cursors[winner];
        out.push_row(&runs[winner], cur);
        cursors[winner] = cur + 1;

        if cursors[winner] < runs[winner].len() {
            heap.push(Reverse((runs[winner].hphigh[cursors[winner]], winner)));
        }
    }

    out
}

/// Fused k-way merge with streaming output for the partition-split case.
/// Routes rows to the correct child writer based on hphigh boundaries.
fn merge_runs_split(
    runs: &[ColumnData],
    schema: &Arc<Schema>,
    writers: &mut [(i64, ArrowWriter<BufWriter<std::fs::File>>)],
) -> anyhow::Result<()> {
    let mut buf = ColumnData::with_capacity(BATCH_SIZE);
    let mut writer_idx = 0;

    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut cursors: Vec<usize> = vec![0; runs.len()];

    for (i, run) in runs.iter().enumerate() {
        if !run.hphigh.is_empty() {
            heap.push(Reverse((run.hphigh[0], i)));
        }
    }

    while let Some(Reverse((hp, winner))) = heap.pop() {
        // Switch to next writer if we've crossed a partition boundary
        while writer_idx < writers.len() - 1 && hp >= writers[writer_idx].0 {
            if buf.len() > 0 {
                let batch = std::mem::replace(&mut buf, ColumnData::with_capacity(BATCH_SIZE))
                    .into_record_batch(schema)?;
                writers[writer_idx].1.write(&batch)?;
            }
            writer_idx += 1;
        }

        let cur = cursors[winner];
        buf.push_row(&runs[winner], cur);
        cursors[winner] = cur + 1;

        if buf.len() >= BATCH_SIZE {
            let batch = std::mem::replace(&mut buf, ColumnData::with_capacity(BATCH_SIZE))
                .into_record_batch(schema)?;
            writers[writer_idx].1.write(&batch)?;
        }

        if cursors[winner] < runs[winner].len() {
            heap.push(Reverse((runs[winner].hphigh[cursors[winner]], winner)));
        }
    }

    if buf.len() > 0 {
        writers[writer_idx].1.write(&buf.into_record_batch(schema)?)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

fn compact_partition(part: &u32, sources: &Vec<(PathBuf, usize, usize)>) -> anyhow::Result<()> {
    if sources.is_empty() {
        return Ok(());
    }

    let schema = pixel_schema();

    // set up paths
    let part_dir = format!("{}/part={}", PIXEL_DB_PATH, part);
    std::fs::create_dir_all(&part_dir)?;
    let pq_path = format!("{}/compacted.parquet", &part_dir);
    let staging_path = format!("{}/compacted_new.parquet", &part_dir);

    let t0 = std::time::Instant::now();

    // Collect sorted runs for the k-way merge.
    let mut runs: Vec<ColumnData> = Vec::new();
    let mut total_rows: usize = 0;

    // Run 0: existing compacted parquet (already sorted by hphigh)
    if let Ok(f) = std::fs::File::open(&pq_path) {
        if let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(f) {
            let num_rows = builder.metadata().file_metadata().num_rows() as usize;
            let mut run = ColumnData::with_capacity(num_rows);
            for batch in builder.build()? {
                run.extend_from_batch(&batch?);
            }
            total_rows += run.len();
            runs.push(run);
        }
    }

    // Runs 1..K: binary chunk sources (sorted by hphigh if batch.py sorts by hphigh)
    // Read in parallel — each source is an independent file slice.
    let chunk_runs: Vec<anyhow::Result<ColumnData>> = sources
        .par_iter()
        .map(|(fpath, start, end)| {
            let mut f = std::fs::File::open(fpath)?;
            let (_, _, _, _, file_num_rows, header_size) = read_chunk_header(&mut f)?;

            let mut run = ColumnData::with_capacity(end - start);
            run.read_from_bin(&f, header_size, file_num_rows, *start, *end)?;

            if !run.is_sorted_by_hphigh() {
                run.sort_by_hphigh();
            }

            Ok(run)
        })
        .collect();

    for result in chunk_runs {
        let run = result?;
        total_rows += run.len();
        runs.push(run);
    }

    let t_read = t0.elapsed();

    // Fast path: single run, no merge needed — single write call
    if runs.len() == 1 && total_rows <= MAX_ROWS_PER_PART {
        let run = runs.pop().unwrap();
        let mut writer = open_writer(&staging_path, &schema)?;
        writer.write(&run.into_record_batch(&schema)?)?;
        writer.close()?;
        return Ok(());
    }

    let num_runs = runs.len();

    // Check if partition needs splitting
    let (level, index) = part_to_level_index(*part);
    if total_rows > MAX_ROWS_PER_PART && level < PART_MAX_LEVEL {
        // Split into 4 child partitions — use streaming merge+write
        let child_level = level + 1;
        let mut writers: Vec<(i64, ArrowWriter<BufWriter<std::fs::File>>)> = Vec::new();

        for k in 0u32..4 {
            let child_index = (index << 2) + k;
            let child_part = level_index_to_part(child_level, child_index);
            let child_dir = format!("{}/part={}", PIXEL_DB_PATH, child_part);
            std::fs::create_dir_all(&child_dir)?;
            let child_pq = format!("{}/compacted_new.parquet", child_dir);

            let hi = (((child_index + 1) as u64) << (2 * (HP_HIGH_LEVEL - child_level))) as i64;
            let writer = open_writer(&child_pq, &schema)?;
            writers.push((hi, writer));
        }

        if let Some(last) = writers.last_mut() {
            last.0 = i64::MAX;
        }

        merge_runs_split(&runs, &schema, &mut writers)?;

        for (_, writer) in writers {
            writer.close()?;
        }

        std::fs::File::create(format!("{}/.split", part_dir))?;
    } else {
        // Non-split: fused merge into one big ColumnData, single write call
        let merged = merge_runs(&runs, total_rows);
        drop(runs); // free input memory before writing

        let file = BufWriter::with_capacity(4 * 1024 * 1024, std::fs::File::create(&staging_path)?);
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(writer_props()))?;
        writer.write(&merged.into_record_batch(&schema)?)?;
        writer.close()?;
    }

    let t_total = t0.elapsed();
    let t_merge_write = t_total - t_read;
    eprintln!(
        "part {}: {} rows ({} runs), read {:.1}s, merge+write {:.1}s, total {:.1}s",
        part,
        total_rows,
        num_runs,
        t_read.as_secs_f64(),
        t_merge_write.as_secs_f64(),
        t_total.as_secs_f64(),
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let task_id = int_env("SLURM_PROCID", 0)?;
    let num_tasks = int_env("SLURM_NTASKS", 1)?;
    let job_id = std::env::var("SLURM_JOB_ID").unwrap_or(0.to_string());
    let _out_file =
        std::env::var("SLURM_JOB_STDOUT").unwrap_or(format!("./slurm-{}.out", job_id));
    let cpus = int_env("SLURM_CPUS_PER_TASK", num_cpus::get())?;
    let num_threads = int_env("COMPACT_THREADS", cpus * 4)?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;

    let part_index = scan_chunk_files()?;
    let part_index = stride_partitions(part_index, task_id.into(), num_tasks.into());

    part_index
        .par_iter()
        .for_each(|(part, sources)| compact_partition(part, sources).unwrap());

    Ok(())
}
