use std::env;
use std::process;
use std::time::Instant;

use chrono::Local;
use rand::RngCore;
use wgpu::util::DeviceExt;

use enclose_horse_solver::flood_fill;
use enclose_horse_solver::grid::{Grid, PuzzleData};

const MAX_BUDGET: usize = 64;
const NUM_THREADS: u32 = 1024;
// Must match @workgroup_size(N) in the WGSL shader. If these diverge,
// the dispatch math will be wrong and threads will be skipped or duplicated.
const WORKGROUP_SIZE: u32 = 64;

// SA parameters for GPU: we run shorter annealing schedules than CPU, but
// compensate with massively more restarts (1024+ simultaneous threads).
// The idea is that many short random walks are better than one long walk
// when you have thousands of cores available.
const INITIAL_TEMP: f32 = 30.0;
const COOLING_RATE: f32 = 0.997;
const MIN_TEMP: f32 = 0.1;
const ITERS_PER_TEMP: u32 = 20;

// This struct is shared between Rust and WGSL. The layout must match EXACTLY.
//
// #[repr(C)] ensures Rust uses C-style struct layout (no field reordering).
// Without it, the Rust compiler may rearrange fields for alignment, breaking
// the GPU's interpretation of the buffer contents.
//
// bytemuck::Pod + Zeroable: these traits from the `bytemuck` crate enable
// safe zero-copy casting between this struct and raw byte slices (&[u8]).
// Pod ("Plain Old Data") means the type has no padding bytes that could leak
// uninitialized memory, and can be safely transmuted from any bit pattern.
// This avoids the unsafety of std::mem::transmute while being zero-cost.
//
// ALIGNMENT: WGSL uniform buffers require 16-byte alignment for the overall
// struct. With 13 fields (52 bytes), we pad to 64 bytes (4 x 16). The _pad
// fields ensure the struct size is a multiple of 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    horse_row: u32,
    horse_col: u32,
    num_grass: u32,
    num_portals: u32,
    num_bonus: u32,
    budget: u32,
    num_threads: u32,
    iters_per_temp: u32,
    initial_temp: f32,
    cooling_rate: f32,
    min_temp: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().skip(1).collect();
    let num_threads = args
        .iter()
        .find_map(|a| a.strip_prefix("--threads=").and_then(|v| v.parse().ok()))
        .unwrap_or(NUM_THREADS);
    let date = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| Local::now().format("%Y-%m-%d").to_string());

    // Fetch puzzle
    let url = format!("https://enclose.horse/api/daily/{}", date);
    eprintln!("Fetching puzzle from {}...", url);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|e| {
            eprintln!("Error creating HTTP client: {}", e);
            process::exit(1);
        });

    let response = client.get(&url).send().unwrap_or_else(|e| {
        eprintln!("Error fetching puzzle: {}", e);
        process::exit(1);
    });

    if !response.status().is_success() {
        eprintln!("API returned status {}", response.status());
        process::exit(1);
    }

    let body = response.text().unwrap_or_else(|e| {
        eprintln!("Error reading response body: {}", e);
        process::exit(1);
    });

    let puzzle: PuzzleData = serde_json::from_str(&body).unwrap_or_else(|e| {
        eprintln!("Error parsing puzzle JSON: {}", e);
        process::exit(1);
    });

    let grid = Grid::from_puzzle(&puzzle);
    let (horse_row, horse_col) = grid.pos_to_rc(grid.horse_pos);

    eprintln!(
        "Grid: {}x{}, horse at ({},{}), budget: {}, candidates: {}",
        grid.width,
        grid.height,
        horse_col,
        horse_row,
        puzzle.budget,
        grid.grass_indices.len()
    );

    // Handle degenerate case: no grass tiles to place walls on
    if grid.grass_indices.is_empty() || puzzle.budget == 0 {
        let result = flood_fill::evaluate(&grid, &[]);
        println!("Score: {} (no walls to place)", result.score);
        return;
    }

    // Run GPU solver
    let (best_score, best_walls) =
        pollster::block_on(run_gpu(&grid, puzzle.budget, num_threads));

    // VERIFICATION: re-evaluate the GPU's best solution using the CPU flood fill.
    // GPU floating point and random behavior can occasionally produce unexpected
    // results, so we always cross-check. This is cheap (one flood fill) compared
    // to the GPU work (thousands of flood fills).
    let cpu_result = flood_fill::evaluate(&grid, &best_walls);
    if cpu_result.score != best_score {
        eprintln!(
            "WARNING: GPU score ({}) doesn't match CPU verification ({}), using CPU score",
            best_score, cpu_result.score
        );
    }

    // Print results
    let mut wall_positions: Vec<(usize, usize)> =
        best_walls.iter().map(|&idx| grid.pos_to_rc(idx)).collect();
    wall_positions.sort();

    println!(
        "Puzzle: \"{}\" (Day {}, {})",
        puzzle.name,
        puzzle.day_number.unwrap_or(0),
        puzzle.daily_date.as_deref().unwrap_or("unknown"),
    );
    println!("Budget: {} walls", puzzle.budget);
    println!(
        "Optimal: {}",
        puzzle
            .optimal_score
            .map_or("unknown".to_string(), |s| s.to_string())
    );
    println!();
    println!(
        "Solution found: score = {} (GPU), verified = {} (CPU)",
        best_score, cpu_result.score
    );

    let wall_strs: Vec<String> = wall_positions
        .iter()
        .map(|(r, c)| format!("({},{})", r, c))
        .collect();
    println!("Walls at: {}", wall_strs.join(" "));
    println!();

    // Print the map with W markers for placed walls
    println!("Map:");
    let wall_set: std::collections::HashSet<usize> = best_walls.iter().copied().collect();
    let rows: Vec<&str> = puzzle.map.lines().collect();
    for (r, row_str) in rows.iter().enumerate() {
        let line: String = row_str
            .chars()
            .enumerate()
            .map(|(c, ch)| {
                let idx = r * grid.width + c;
                if wall_set.contains(&idx) {
                    'W'
                } else {
                    ch
                }
            })
            .collect();
        println!("{}", line);
    }
}

// =============================================================================
// GPU PIPELINE OVERVIEW (wgpu)
//
// The wgpu pipeline follows a strict sequence to get work onto the GPU:
//
//   1. Instance   — entry point to the wgpu API, enumerates available backends
//   2. Adapter    — represents a physical GPU; we request high-performance
//   3. Device     — a logical connection to the GPU; all resource creation goes through it
//   4. Shader     — the WGSL compute program, compiled to GPU machine code
//   5. Buffers    — GPU memory allocations for input data and output results
//   6. Bind Group — tells the shader which buffers map to which @binding slots
//   7. Pipeline   — combines the shader + bind group layout into a runnable unit
//   8. Dispatch   — records GPU commands (set pipeline, bind data, launch N workgroups)
//   9. Submit     — sends the command buffer to the GPU for execution
//  10. Readback   — copy results from GPU memory to CPU-readable staging buffers
//
// =============================================================================
async fn run_gpu(grid: &Grid, budget: usize, num_threads: u32) -> (i32, Vec<usize>) {
    let start = Instant::now();

    // --- wgpu setup ---
    // Instance: the root object that discovers available GPU backends (Vulkan, Metal, DX12, etc.)
    let instance = wgpu::Instance::default();

    // Adapter: selects which physical GPU to use. HighPerformance prefers
    // discrete GPUs over integrated ones (important on laptops with both).
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No GPU adapter found");

    eprintln!("Using GPU: {}", adapter.get_info().name);

    // Device + Queue: the logical GPU handle. The Device creates resources
    // (buffers, shaders, pipelines). The Queue accepts command buffers for
    // execution. A single Device can have multiple Queues on some backends.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("GPU Solver"),
            ..Default::default()
        })
        .await
        .expect("Failed to create GPU device");

    // --- Shader ---
    // include_str! embeds the WGSL source at compile time, so the binary is
    // self-contained (no need to ship the .wgsl file separately). The GPU
    // driver compiles WGSL to native GPU machine code at runtime.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SA Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    // --- Prepare data ---
    // Transform the Grid's Rust data structures into flat arrays that match
    // the WGSL shader's expected buffer layouts.
    let (horse_row, horse_col) = grid.pos_to_rc(grid.horse_pos);

    let params = GpuParams {
        width: grid.width as u32,
        height: grid.height as u32,
        horse_row: horse_row as u32,
        horse_col: horse_col as u32,
        num_grass: grid.grass_indices.len() as u32,
        num_portals: grid.portal_pairs.len() as u32,
        num_bonus: grid.bonus_scores.len() as u32,
        budget: budget as u32,
        num_threads,
        iters_per_temp: ITERS_PER_TEMP,
        initial_temp: INITIAL_TEMP,
        cooling_rate: COOLING_RATE,
        min_temp: MIN_TEMP,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };

    // Pad bitboards to 32 rows (the shader uses fixed-size arrays)
    let mut passable_data = [0u32; 32];
    let mut border_data = [0u32; 32];
    for (i, &v) in grid.passable_bitboard.iter().enumerate() {
        passable_data[i] = v;
    }
    for (i, &v) in grid.border_bitboard.iter().enumerate() {
        border_data[i] = v;
    }

    // Grass indices: flat list of grid positions where walls can be placed
    let grass_data: Vec<u32> = grid.grass_indices.iter().map(|&i| i as u32).collect();

    // Portal data: flatten each (pos_a, pos_b) pair into (row_a, col_a, row_b, col_b)
    let mut portal_data_vec: Vec<u32> = Vec::new();
    for &(a, b) in &grid.portal_pairs {
        let (ra, ca) = grid.pos_to_rc(a);
        let (rb, cb) = grid.pos_to_rc(b);
        portal_data_vec.extend_from_slice(&[ra as u32, ca as u32, rb as u32, cb as u32]);
    }

    // Bonus data: flatten each (pos, adjustment) into (row, col, adjustment)
    let mut bonus_data_vec: Vec<i32> = Vec::new();
    for &(pos, adj) in &grid.bonus_scores {
        let (r, c) = grid.pos_to_rc(pos);
        bonus_data_vec.extend_from_slice(&[r as i32, c as i32, adj]);
    }

    // RNG seeds: 2 u32s per thread. The PCG RNG needs (state, increment) where
    // increment must be odd to ensure a full-period LCG. We force oddness with |1.
    // Each thread gets a unique random seed so their SA runs explore different
    // parts of the solution space.
    let mut rng = rand::rng();
    let rng_data: Vec<u32> = (0..num_threads * 2)
        .map(|i| {
            let val = rng.next_u32();
            if i % 2 == 1 { val | 1 } else { val } // odd increment
        })
        .collect();

    // --- Create GPU buffers ---
    //
    // bytemuck::bytes_of / cast_slice: zero-copy conversion from typed Rust data
    // to &[u8] byte slices. The GPU API needs raw bytes; bytemuck does this safely
    // by verifying at compile time (via the Pod trait) that the types have no
    // padding holes or invalid bit patterns. This replaces what would otherwise
    // require unsafe transmute calls.
    //
    // Buffer usage flags tell wgpu how each buffer will be used, enabling the
    // driver to place it in optimal memory:
    //   UNIFORM  — small, read-only, broadcast to all threads (fast constant cache)
    //   STORAGE  — larger read-only or read-write arrays (global GPU memory)
    //   COPY_SRC — this buffer's contents will be copied elsewhere (to staging)
    //   COPY_DST — this buffer receives copied data (staging buffer)
    //   MAP_READ — CPU can memory-map this buffer to read results back

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let passable_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Passable BB"),
        contents: bytemuck::cast_slice(&passable_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let border_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Border BB"),
        contents: bytemuck::cast_slice(&border_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // wgpu requires all buffers to have non-zero size. For empty arrays (e.g.,
    // no grass tiles), we provide a dummy single-element array. The shader won't
    // read beyond num_grass/num_portals/num_bonus anyway.
    let grass_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grass Indices"),
        contents: bytemuck::cast_slice(if grass_data.is_empty() {
            &[0u32]
        } else {
            &grass_data
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let portal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Portal Data"),
        contents: bytemuck::cast_slice(if portal_data_vec.is_empty() {
            &[0u32]
        } else {
            &portal_data_vec
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bonus_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bonus Data"),
        contents: bytemuck::cast_slice(if bonus_data_vec.is_empty() {
            &[0i32]
        } else {
            &bonus_data_vec
        }),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let rng_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RNG State"),
        contents: bytemuck::cast_slice(&rng_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let result_scores_size = (num_threads as u64) * 4; // 4 bytes per i32
    let result_walls_size = (num_threads as u64) * (MAX_BUDGET as u64) * 4;

    // Output buffers need COPY_SRC because GPU-side storage buffers can't be
    // directly read by the CPU. We'll copy their contents to staging buffers.
    let result_scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Scores"),
        size: result_scores_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let result_walls_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Walls"),
        size: result_walls_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // STAGING BUFFERS:
    // GPU memory is typically not directly accessible by the CPU (it lives in
    // VRAM on discrete GPUs). To read results back, we need an intermediate
    // "staging" buffer that is CPU-mappable (MAP_READ) and can receive GPU
    // copies (COPY_DST). The flow is:
    //   GPU storage buffer --[copy_buffer_to_buffer]--> staging buffer --[map_async]--> CPU
    let staging_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Scores"),
        size: result_scores_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_walls = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Walls"),
        size: result_walls_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Bind group layout ---
    // A bind group defines how shader @binding(N) declarations map to actual
    // GPU buffers. The layout declares the *shape* (what type of buffer at each
    // binding), and the bind group provides the *actual* buffers.
    //
    // This is similar to a function signature (layout) vs. calling with actual
    // arguments (bind group). The GPU uses this to set up descriptor tables
    // that the shader references at runtime.
    //
    // Bindings 0: uniform (params), 1-5: read-only storage, 6-8: read-write storage.
    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..9u32)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    // Bindings 1-5 are read-only input data; 6-8 are mutable (RNG, outputs)
                    wgpu::BufferBindingType::Storage { read_only: i <= 5 }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SA Layout"),
        entries: &bind_group_layout_entries,
    });

    // Create the actual bind group: connect each @binding(N) to a real buffer
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SA Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: passable_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: border_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grass_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: portal_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: bonus_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: rng_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: result_scores_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: result_walls_buf.as_entire_binding(),
            },
        ],
    });

    // --- Pipeline ---
    // A compute pipeline bundles the compiled shader + bind group layout into
    // a single object the GPU can execute. Creating the pipeline triggers
    // shader compilation on the GPU driver, which can take a moment.
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SA Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SA Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("sa_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- Dispatch ---
    // WORKGROUP DISPATCHING MATH:
    // We want `num_threads` total threads, each workgroup has WORKGROUP_SIZE threads.
    // ceil(num_threads / WORKGROUP_SIZE) gives the number of workgroups needed.
    // The integer ceil trick: (a + b - 1) / b avoids floating point.
    // Extra threads in the last workgroup (if num_threads isn't a multiple of 64)
    // are handled by the guard check at the top of sa_main.
    let num_workgroups = (num_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    eprintln!(
        "Dispatching {} threads ({} workgroups of {})...",
        num_threads, num_workgroups, WORKGROUP_SIZE
    );

    let dispatch_start = Instant::now();

    // Command encoder: records a sequence of GPU commands into a command buffer.
    // Commands are not executed immediately — they're batched and submitted at once.
    // This lets the GPU driver optimize the command stream (reorder, batch, etc.).
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SA Encoder"),
    });

    {
        // Compute pass: a scope where we can dispatch compute shaders.
        // The braces limit the borrow on `encoder` so we can use it again below.
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SA Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Launch num_workgroups in the x dimension (y=1, z=1 since our problem is 1D)
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // After compute finishes, copy results from GPU storage to staging buffers.
    // This is recorded as part of the same command buffer, so it executes
    // after the compute pass completes (GPU commands execute in order).
    encoder.copy_buffer_to_buffer(&result_scores_buf, 0, &staging_scores, 0, result_scores_size);
    encoder.copy_buffer_to_buffer(&result_walls_buf, 0, &staging_walls, 0, result_walls_size);

    // Submit the command buffer to the GPU queue for execution.
    // This is non-blocking on the CPU — the GPU works asynchronously.
    queue.submit(Some(encoder.finish()));

    // --- Read back results ---
    // ASYNC MAP PATTERN:
    // map_async requests that the staging buffer be made CPU-accessible.
    // It's async because the GPU may still be computing — the mapping only
    // completes after the GPU finishes all submitted work.
    //
    // The flow:
    //   1. Call map_async with a callback that signals completion via a channel
    //   2. Call device.poll(Wait) to block until the GPU finishes all work
    //   3. Receive on the channel to confirm the mapping succeeded
    //   4. get_mapped_range() returns a &[u8] view of the buffer contents
    //
    // We use std::sync::mpsc channels to bridge the callback-based API into
    // synchronous code. The callback fires on the wgpu internal thread.
    let scores_slice = staging_scores.slice(..);
    let walls_slice = staging_walls.slice(..);

    let (tx1, rx1) = std::sync::mpsc::sync_channel(1);
    let (tx2, rx2) = std::sync::mpsc::sync_channel(1);
    scores_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx1.send(r);
    });
    walls_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx2.send(r);
    });
    device.poll(wgpu::PollType::Wait).unwrap();
    rx1.recv().unwrap().unwrap();
    rx2.recv().unwrap().unwrap();

    let gpu_time = dispatch_start.elapsed();

    // get_mapped_range returns raw bytes; bytemuck::cast_slice reinterprets
    // them as typed slices without copying (zero-cost)
    let scores_view = scores_slice.get_mapped_range();
    let scores: &[i32] = bytemuck::cast_slice(&scores_view);

    let walls_view = walls_slice.get_mapped_range();
    let all_walls: &[u32] = bytemuck::cast_slice(&walls_view);

    // Find the thread that achieved the highest score.
    // We only look at active threads (not excess threads from workgroup padding,
    // which wrote i32::MIN as their score).
    let best_tid = scores
        .iter()
        .take(num_threads as usize)
        .enumerate()
        .max_by_key(|(_, &s)| s)
        .map(|(i, _)| i)
        .unwrap();

    let best_score = scores[best_tid];
    // Each thread's walls occupy a contiguous MAX_BUDGET-sized slot in the flat array
    let best_walls: Vec<usize> = all_walls
        [best_tid * MAX_BUDGET..best_tid * MAX_BUDGET + budget]
        .iter()
        .map(|&w| w as usize)
        .collect();

    // Score distribution stats for diagnostics
    let nonzero: Vec<i32> = scores.iter().copied().filter(|&s| s > 0).collect();
    let avg = if nonzero.is_empty() {
        0.0
    } else {
        nonzero.iter().sum::<i32>() as f64 / nonzero.len() as f64
    };

    eprintln!(
        "GPU compute: {:?} ({} threads, {} found enclosure, best={}, avg={:.1})",
        gpu_time,
        num_threads,
        nonzero.len(),
        best_score,
        avg
    );
    eprintln!("Total time: {:?}", start.elapsed());

    // Clean up: drop the mapped views before unmapping the buffers
    drop(scores_view);
    drop(walls_view);
    staging_scores.unmap();
    staging_walls.unmap();

    (best_score, best_walls)
}
