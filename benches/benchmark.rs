use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tensorlab::Tensor;
use std::time::Instant;

fn bench_transpose_methods(c:&mut Criterion) {
    let size = 4096; // 4096x4096 matrix
    let data = vec![1.0; size * size];
    let tensor = Tensor::new(&data, &vec![size, size]).unwrap();

    println!("Encontrando el block_size optimo para {}x{} matrix", size, size);
    let optimal_block_size = tensor.find_optimal_block();
    println!("Block size optimo: {}", optimal_block_size);
    
    let mut group = c.benchmark_group("transpose_4096x4096");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    // Benchmark naive
    let mut time_naive = 0.0;
    group.bench_function("transpose_naive", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(tensor.transpose_naive().unwrap());
            }
            let elapsed = start.elapsed();
            time_naive = elapsed.as_secs_f64() / iters as f64;
            elapsed
        })
    });
 
    // Benchmark blocked
    let mut time_blocked = 0.0;
    group.bench_function(
        &format!("transpose_blocked_{}", optimal_block_size),
        |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(tensor.transpose_blocked(optimal_block_size).unwrap());
                }
                let elapsed = start.elapsed();
                time_blocked = elapsed.as_secs_f64() / iters as f64;
                elapsed
            })
        },
    );
 
    // Benchmark inplace
    let mut time_inplace = 0.0;
    group.bench_function("transpose_inplace", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut t = tensor.clone();
                black_box(t.transpose_inplace().unwrap());
            }
            let elapsed = start.elapsed();
            time_inplace = elapsed.as_secs_f64() / iters as f64;
            elapsed
        })
    });
 
    group.finish();
    
    println!("Times:");
    println!("   transpose_naive:           {:.4} ms (baseline)", time_naive * 1000.0);
    println!(
        "   transpose_blocked_{}:      {:.4} ms",
        optimal_block_size,
        time_blocked * 1000.0
    );
    println!("   transpose_inplace:         {:.4} ms", time_inplace * 1000.0);
 
    println!("Speedups:");
    let speedup_blocked = time_naive / time_blocked;
    let speedup_inplace = time_naive / time_inplace;
    println!("   blocked vs naive:          {:.2}× faster", speedup_blocked);
    println!("   inplace vs naive:          {:.2}× faster", speedup_inplace);
 
    println!("% improvement:");
    let percent_blocked = ((time_naive - time_blocked) / time_naive) * 100.0;
    let percent_inplace = ((time_naive - time_inplace) / time_naive) * 100.0;
    println!("   blocked vs naive:          {:.1}% faster", percent_blocked);
    println!("   inplace vs naive:          {:.1}% faster", percent_inplace);
 
    println!("Difference:");
    let diff_blocked = (time_naive - time_blocked) * 1000.0;
    let diff_inplace = (time_naive - time_inplace) * 1000.0;
    println!("   blocked saves:            {:.4} ms per transposition", diff_blocked);
    println!("   inplace saves:            {:.4} ms por transposition", diff_inplace);
 
    println!("Comparison between methods:");
    let speedup_blocked_vs_inplace = time_inplace / time_blocked;
    let percent_blocked_vs_inplace = ((time_inplace - time_blocked) / time_inplace) * 100.0;
    println!(
        "   inplace vs blocked:        {:.2}× faster ({}% improvement)",
        speedup_blocked_vs_inplace, percent_blocked_vs_inplace
    );
}
 
criterion_group!(benchmark, bench_transpose_methods);
criterion_main!(benchmark);