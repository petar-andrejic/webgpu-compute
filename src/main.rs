use std::{error::Error, time::Duration};
use tokio::time::sleep;

use webgpu_compute::Engine;

async fn gpu_work(engine: Engine) {
    println!("Begin GPU work");
    let shader_desc = wgpu::include_wgsl!("../shaders/hello_world.wgsl");
    let module = engine.device.create_shader_module(shader_desc);
    let op = engine.create_operation(&module, "hello world".to_string());
    let data_in = engine.bind_vec(vec![0u32, 1u32, 2u32, 3u32]);
    let tmp = engine.create_buffer::<u32>(4);
    let mut data_out = engine.create_gpu_vec::<u32>(4);

    let view1 = [data_in.buf(), &tmp];
    let view2 = [&tmp, data_out.buf()];

    engine.to_gpu(&data_in).await;
    engine.dispatch([(&op, view1), (&op, view2)]).await;
    engine.to_cpu(&mut data_out).await;
    print!("[ ");
    for i in data_out.data {
        print!("{} ", i);
    }
    println!("]");
    println!("Finish GPU work");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let engine = Engine::new().await?;
    let handle1 = tokio::task::spawn(async move { gpu_work(engine).await });
    let handle2 = tokio::task::spawn(async {
        sleep(Duration::from_millis(4)).await;
        println!("Doing some CPU work while waiting for GPU");
    });
    tokio::try_join!(handle1, handle2)?;
    Ok(())
}
