use arrow::array::{Float32Array, RecordBatch, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use glob::glob;
use itertools::izip;
use ndarray::s;
use num_cpus;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
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

fn scan_chunk_files() -> anyhow::Result<HashMap<u32, Vec<(PathBuf, usize, usize)>>> {
    let h5_files = glob(&format!("{}/chunk_*.hdf5", PIXEL_DB_PATH))?;
    let mut part_index: HashMap<u32, Vec<(PathBuf, usize, usize)>> = HashMap::new();

    for fpath in h5_files {
        let fpath = match fpath {
            Ok(p) => p,
            Err(e) => {
                eprintln!("warning: skipping glob entry: {}", e);
                continue;
            }
        };

        let f = match hdf5::File::open(&fpath) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("warning: skipping {}: {}", fpath.display(), e);
                continue;
            }
        };

        let result = (|| -> anyhow::Result<()> {
            let part_ids = f.attr("part_ids")?.read_1d::<u32>()?;
            let part_starts = f.attr("part_starts")?.read_1d::<u64>()?;
            let part_ends = f.attr("part_ends")?.read_1d::<u64>()?;

            for (pid, start, end) in izip!(part_ids.iter(), part_starts.iter(), part_ends.iter()) {
                part_index.entry(*pid).or_insert(Vec::new()).push((
                    fpath.clone(),
                    *start as usize,
                    *end as usize,
                ));
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
        Field::new("flags", DataType::UInt32, false),
        Field::new("hphigh", DataType::UInt64, false),
        Field::new("hppart", DataType::UInt32, false),
        Field::new("waveid", DataType::UInt32, false),
        Field::new("imageid", DataType::UInt64, false),
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

/// Read columns from an HDF5 file and extend the pre-allocated Vecs.
macro_rules! extend_from_h5 {
    ($file:expr, $start:expr, $end:expr, $( $col:ident : $ds:literal : $ty:ty ),+ $(,)?) => {
        $( $col.extend(
            $file.dataset($ds)?
                .read_slice_1d::<$ty, _>(s![$start..$end])?
                .iter().copied(),
        ); )+
    };
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
    let mut col_flags: Vec<u32> = Vec::with_capacity(total_len);
    let mut col_hphigh: Vec<u64> = Vec::with_capacity(total_len);
    let mut col_hppart: Vec<u32> = Vec::with_capacity(total_len);
    let mut col_waveid: Vec<u32> = Vec::with_capacity(total_len);
    let mut col_imageid: Vec<u64> = Vec::with_capacity(total_len);

    // read existing compacted parquet
    if let Some(b) = builder {
        for batch in b.build()? {
            let batch = batch?;
            extend_from_batch!(batch,
                col_flux:     "flux":     Float32Array,
                col_variance: "variance": Float32Array,
                col_zodi:     "zodi":     Float32Array,
                col_flags:    "flags":    UInt32Array,
                col_hphigh:   "hphigh":   UInt64Array,
                col_hppart:   "hppart":   UInt32Array,
                col_waveid:   "waveid":   UInt32Array,
                col_imageid:  "imageid":  UInt64Array,
            );
        }
    }

    // read HDF5 chunk sources
    for (fpath, start, end) in sources.iter() {
        let file = hdf5::File::open(fpath)?;
        let (start, end) = (*start, *end);
        extend_from_h5!(file, start, end,
            col_flux:     "flux":     f32,
            col_variance: "variance": f32,
            col_zodi:     "zodi":     f32,
            col_flags:    "flags":    u32,
            col_hphigh:   "hphigh":   u64,
            col_hppart:   "hppart":   u32,
            col_waveid:   "waveid":   u32,
            col_imageid:  "imageid":  u64,
        );
    }

    // sort by hphigh — one column at a time to avoid 2x total memory
    let mut sort_indices: Vec<usize> = (0..total_len).collect();
    sort_indices.sort_by(|&i, &j| col_hphigh[i].cmp(&col_hphigh[j]));

    // apply_permutation takes ownership, so the old Vec is dropped before the next
    fn apply_permutation<T: Copy>(v: Vec<T>, indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&i| v[i]).collect()
    }
    let col_flux = apply_permutation(col_flux, &sort_indices);
    let col_variance = apply_permutation(col_variance, &sort_indices);
    let col_zodi = apply_permutation(col_zodi, &sort_indices);
    let col_flags = apply_permutation(col_flags, &sort_indices);
    let col_hphigh = apply_permutation(col_hphigh, &sort_indices);
    let col_hppart = apply_permutation(col_hppart, &sort_indices);
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
            Arc::new(UInt32Array::from(col_flags)),
            Arc::new(UInt64Array::from(col_hphigh)),
            Arc::new(UInt32Array::from(col_hppart)),
            Arc::new(UInt32Array::from(col_waveid)),
            Arc::new(UInt64Array::from(col_imageid)),
        ],
    )?;

    // check if partition needs splitting
    let (level, index) = part_to_level_index(*part);
    if sorted.num_rows() > MAX_ROWS_PER_PART && level < PART_MAX_LEVEL {
        let hphigh = sorted
            .column_by_name("hphigh")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // split into 4 subpartitions at the next level
        let child_level = level + 1;
        for k in 0u32..4 {
            let child_index = (index << 2) + k;
            let child_part = level_index_to_part(child_level, child_index);
            let child_dir = format!("{}/part={}", PIXEL_DB_PATH, child_part);
            std::fs::create_dir_all(&child_dir)?;

            let lo = (child_index as u64) << (2 * (HP_HIGH_LEVEL - child_level));
            let hi = ((child_index + 1) as u64) << (2 * (HP_HIGH_LEVEL - child_level));

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
