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
const WORKGROUP_SIZE: u32 = 64;

// SA parameters for GPU: shorter runs than CPU, but many more restarts
const INITIAL_TEMP: f32 = 30.0;
const COOLING_RATE: f32 = 0.997;
const MIN_TEMP: f32 = 0.1;
const ITERS_PER_TEMP: u32 = 20;

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

    // Verify GPU result with CPU flood fill
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

async fn run_gpu(grid: &Grid, budget: usize, num_threads: u32) -> (i32, Vec<usize>) {
    let start = Instant::now();

    // --- wgpu setup ---
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No GPU adapter found");

    eprintln!("Using GPU: {}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("GPU Solver"),
            ..Default::default()
        })
        .await
        .expect("Failed to create GPU device");

    // --- Shader ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SA Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    // --- Prepare data ---
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

    // Passable and border bitboards (padded to 32 rows)
    let mut passable_data = [0u32; 32];
    let mut border_data = [0u32; 32];
    for (i, &v) in grid.passable_bitboard.iter().enumerate() {
        passable_data[i] = v;
    }
    for (i, &v) in grid.border_bitboard.iter().enumerate() {
        border_data[i] = v;
    }

    // Grass indices
    let grass_data: Vec<u32> = grid.grass_indices.iter().map(|&i| i as u32).collect();

    // Portal data: flat (row_a, col_a, row_b, col_b) per portal
    let mut portal_data_vec: Vec<u32> = Vec::new();
    for &(a, b) in &grid.portal_pairs {
        let (ra, ca) = grid.pos_to_rc(a);
        let (rb, cb) = grid.pos_to_rc(b);
        portal_data_vec.extend_from_slice(&[ra as u32, ca as u32, rb as u32, cb as u32]);
    }

    // Bonus data: flat (row, col, adjustment) per bonus tile
    let mut bonus_data_vec: Vec<i32> = Vec::new();
    for &(pos, adj) in &grid.bonus_scores {
        let (r, c) = grid.pos_to_rc(pos);
        bonus_data_vec.extend_from_slice(&[r as i32, c as i32, adj]);
    }

    // RNG seeds: 2 u32s per thread (state + increment, increment must be odd)
    let mut rng = rand::rng();
    let rng_data: Vec<u32> = (0..num_threads * 2)
        .map(|i| {
            let val = rng.next_u32();
            if i % 2 == 1 { val | 1 } else { val }
        })
        .collect();

    // --- Create buffers ---
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

    // For empty buffers, provide at least 4 bytes (wgpu requires non-zero size)
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

    let result_scores_size = (num_threads as u64) * 4;
    let result_walls_size = (num_threads as u64) * (MAX_BUDGET as u64) * 4;

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
    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..9u32)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
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
    let num_workgroups = (num_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    eprintln!(
        "Dispatching {} threads ({} workgroups of {})...",
        num_threads, num_workgroups, WORKGROUP_SIZE
    );

    let dispatch_start = Instant::now();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SA Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SA Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Copy results to staging buffers
    encoder.copy_buffer_to_buffer(&result_scores_buf, 0, &staging_scores, 0, result_scores_size);
    encoder.copy_buffer_to_buffer(&result_walls_buf, 0, &staging_walls, 0, result_walls_size);

    queue.submit(Some(encoder.finish()));

    // --- Read back results ---
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

    let scores_view = scores_slice.get_mapped_range();
    let scores: &[i32] = bytemuck::cast_slice(&scores_view);

    let walls_view = walls_slice.get_mapped_range();
    let all_walls: &[u32] = bytemuck::cast_slice(&walls_view);

    // Find best result across active threads only
    let best_tid = scores
        .iter()
        .take(num_threads as usize)
        .enumerate()
        .max_by_key(|(_, &s)| s)
        .map(|(i, _)| i)
        .unwrap();

    let best_score = scores[best_tid];
    let best_walls: Vec<usize> = all_walls
        [best_tid * MAX_BUDGET..best_tid * MAX_BUDGET + budget]
        .iter()
        .map(|&w| w as usize)
        .collect();

    // Score distribution stats
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

    drop(scores_view);
    drop(walls_view);
    staging_scores.unmap();
    staging_walls.unmap();

    (best_score, best_walls)
}
