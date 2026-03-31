use arrow::array::{Float32Array, Int32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bytemuck;
use glob::glob;
use num_cpus;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

const HP_HIGH_LEVEL: u32 = 22;
const MAX_ROWS_PER_PART: usize = 200_000_000;
const PART_MAX_LEVEL: u32 = 10;
const PIXEL_DB_PATH: &str = "/mnt/sdceph/users/chainje/spxdb-test/pixels";

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
        .build()
}

fn write_record_batch(path: &str, schema: &Arc<Schema>, batch: &RecordBatch) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(schema), Some(writer_props()))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// Extract a typed column from a RecordBatch and extend a Vec.
macro_rules! extend_from_batch {
    ($batch:expr, $( $col:ident : $name:literal : $arrow_ty:ty ),+ $(,)?) => {
        $( $col.extend(
            $batch.column_by_name($name).unwrap()
                .as_any().downcast_ref::<$arrow_ty>().unwrap()
                .values().iter().copied(),
        ); )+
    };
}

/// Read a typed column slice from a binary chunk file.
fn read_bin_column<T: bytemuck::Pod>(
    f: &mut std::fs::File,
    header_size: usize,
    col_byte_offset: usize,
    start: usize,
    end: usize,
) -> anyhow::Result<Vec<T>> {
    let elem_size = std::mem::size_of::<T>();
    let offset = header_size + col_byte_offset + start * elem_size;
    f.seek(SeekFrom::Start(offset as u64))?;
    let count = end - start;
    let mut buf = vec![0u8; count * elem_size];
    f.read_exact(&mut buf)?;
    Ok(bytemuck::cast_slice(&buf).to_vec())
}

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

    // look for existing table
    let builder = std::fs::File::open(&pq_path)
        .ok()
        .map(|f| ParquetRecordBatchReaderBuilder::try_new(f))
        .transpose()?;

    let num_rows = builder
        .as_ref()
        .map_or(0, |b| b.metadata().file_metadata().num_rows()) as usize;
    let total_len = num_rows
        + sources
            .iter()
            .map(|(_, start, end)| end - start)
            .sum::<usize>();

    // pre-allocate column vectors
    let mut col_flux: Vec<f32> = Vec::with_capacity(total_len);
    let mut col_variance: Vec<f32> = Vec::with_capacity(total_len);
    let mut col_zodi: Vec<f32> = Vec::with_capacity(total_len);
    let mut col_flags: Vec<i32> = Vec::with_capacity(total_len);
    let mut col_hphigh: Vec<i64> = Vec::with_capacity(total_len);
    let mut col_waveid: Vec<i32> = Vec::with_capacity(total_len);
    let mut col_imageid: Vec<i64> = Vec::with_capacity(total_len);

    // read existing compacted parquet
    if let Some(b) = builder {
        for batch in b.build()? {
            let batch = batch?;
            extend_from_batch!(batch,
                col_flux:     "flux":     Float32Array,
                col_variance: "variance": Float32Array,
                col_zodi:     "zodi":     Float32Array,
                col_flags:    "flags":    Int32Array,
                col_hphigh:   "hphigh":   Int64Array,
                col_waveid:   "waveid":   Int32Array,
                col_imageid:  "imageid":  Int64Array,
            );
        }
    }

    // read binary chunk sources
    for (fpath, start, end) in sources.iter() {
        let mut f = std::fs::File::open(fpath)?;
        let (_, _, _, _, file_num_rows, header_size) = read_chunk_header(&mut f)?;
        let (start, end) = (*start, *end);

        // Column byte offsets within the data section (after header).
        // Must match the write order in batch.py CHUNK_COLUMNS.
        let mut off = 0usize;
        col_flux.extend(read_bin_column::<f32>(&mut f, header_size, off, start, end)?);
        off += file_num_rows * 4;
        col_variance.extend(read_bin_column::<f32>(&mut f, header_size, off, start, end)?);
        off += file_num_rows * 4;
        col_zodi.extend(read_bin_column::<f32>(&mut f, header_size, off, start, end)?);
        off += file_num_rows * 4;
        col_flags.extend(read_bin_column::<i32>(&mut f, header_size, off, start, end)?);
        off += file_num_rows * 4;
        col_hphigh.extend(read_bin_column::<i64>(&mut f, header_size, off, start, end)?);
        off += file_num_rows * 8;
        col_waveid.extend(read_bin_column::<i32>(&mut f, header_size, off, start, end)?);
        off += file_num_rows * 4;
        col_imageid.extend(read_bin_column::<i64>(&mut f, header_size, off, start, end)?);
    }

    // sort by hphigh — one column at a time to avoid 2x total memory
    let mut sort_indices: Vec<usize> = (0..total_len).collect();
    sort_indices.sort_by(|&i, &j| col_hphigh[i].cmp(&col_hphigh[j]));

    fn apply_permutation<T: Copy>(v: Vec<T>, indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&i| v[i]).collect()
    }
    let col_flux = apply_permutation(col_flux, &sort_indices);
    let col_variance = apply_permutation(col_variance, &sort_indices);
    let col_zodi = apply_permutation(col_zodi, &sort_indices);
    let col_flags = apply_permutation(col_flags, &sort_indices);
    let col_hphigh = apply_permutation(col_hphigh, &sort_indices);
    let col_waveid = apply_permutation(col_waveid, &sort_indices);
    let col_imageid = apply_permutation(col_imageid, &sort_indices);
    drop(sort_indices);

    // build sorted RecordBatch (consumes the Vecs, zero-copy slicing below)
    let sorted = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Float32Array::from(col_flux)),
            Arc::new(Float32Array::from(col_variance)),
            Arc::new(Float32Array::from(col_zodi)),
            Arc::new(Int32Array::from(col_flags)),
            Arc::new(Int64Array::from(col_hphigh)),
            Arc::new(Int32Array::from(col_waveid)),
            Arc::new(Int64Array::from(col_imageid)),
        ],
    )?;

    // check if partition needs splitting
    let (level, index) = part_to_level_index(*part);
    if sorted.num_rows() > MAX_ROWS_PER_PART && level < PART_MAX_LEVEL {
        let hphigh = sorted
            .column_by_name("hphigh")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // split into 4 subpartitions at the next level
        let child_level = level + 1;
        for k in 0u32..4 {
            let child_index = (index << 2) + k;
            let child_part = level_index_to_part(child_level, child_index);
            let child_dir = format!("{}/part={}", PIXEL_DB_PATH, child_part);
            std::fs::create_dir_all(&child_dir)?;

            let lo = ((child_index as u64) << (2 * (HP_HIGH_LEVEL - child_level))) as i64;
            let hi = (((child_index + 1) as u64) << (2 * (HP_HIGH_LEVEL - child_level))) as i64;

            // binary search on sorted hphigh for the range [lo, hi)
            let start = hphigh.values().partition_point(|&v| v < lo);
            let end = hphigh.values().partition_point(|&v| v < hi);

            // zero-copy slice of the sorted batch
            let child_batch = sorted.slice(start, end - start);
            let child_pq = format!("{}/compacted_new.parquet", child_dir);
            write_record_batch(&child_pq, &schema, &child_batch)?;
        }

        // write split marker so post_compact knows to clean up the parent
        std::fs::File::create(format!("{}/.split", part_dir))?;
    } else {
        write_record_batch(&staging_path, &schema, &sorted)?;
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    // read environment, do setup
    let task_id = int_env("SLURM_PROCID", 0)?;
    let num_tasks = int_env("SLURM_NTASKS", 1)?;
    let job_id = std::env::var("SLURM_JOB_ID").unwrap_or(0.to_string());
    let _out_file = std::env::var("SLURM_JOB_STDOUT").unwrap_or(format!("./slurm-{}.out", job_id));
    let cpus = int_env("SLURM_CPUS_PER_TASK", num_cpus::get())?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(cpus)
        .build_global()?;

    let part_index = scan_chunk_files()?;
    let part_index = stride_partitions(part_index, task_id.into(), num_tasks.into());

    // do work
    part_index
        .par_iter()
        .for_each(|(part, sources)| compact_partition(part, sources).unwrap());

    Ok(())
}
