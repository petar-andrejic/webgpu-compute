use pollster::FutureExt;
use std::error::Error;

use webgpu_compute::{Engine, SyncError};

type EmptyResult = Result<(), Box<dyn Error>>;

async fn run(engine: &Engine) -> EmptyResult {
    let shader_desc = wgpu::include_wgsl!("../shaders/hello_world.wgsl");
    let module = engine.device.create_shader_module(shader_desc);
    let op = engine.create_operation(&module, "hello world".to_string());
    let data_in = engine.bind_vec(vec![0u32, 1u32, 2u32, 3u32]);
    let tmp = engine.create_buffer::<u32>(4);
    let mut data_out = engine.create_gpu_vec::<u32>(4);

    let view1 = [data_in.buf(), &tmp];
    let view2 = [&tmp, data_out.buf()];
    
    engine.to_gpu(&data_in).await;
    let fut = async {
        engine.dispatch([(&op, view1), (&op, view2)]).await; 
        println!("Gpu work done");
        Ok::<(),SyncError>(())
    };
    println!("Doing some CPU work while waiting for GPU");
    fut.await?; // Lifetimes ensure we have to await for this before calling to_cpu!
    engine.to_cpu(&mut data_out).await?;
    print!("[ ");
    for i in data_out.data {
        print!("{} ", i);
    }
    println!("]");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let engine = Engine::new().block_on()?;
    run(&engine).block_on()?;
    Ok(())
}
