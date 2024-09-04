use std::error::Error;


use webgpu_compute::Engine;

fn main() -> Result<(), Box<dyn Error>> {
    let engine = Engine::new()?;
    let shader_desc = wgpu::include_wgsl!("../shaders/hello_world.wgsl");
    let module = engine.device.create_shader_module(shader_desc);
    let op = engine.create_operation(&module, "hello world".to_string());
    let data_in = engine.bind_vec(vec![0u32, 1u32, 2u32, 3u32]);
    let tmp = engine.create_gpu_vec::<u32>(4);
    let mut data_out = engine.create_gpu_vec::<u32>(4);

    engine.to_gpu(&data_in);
    
    let view1 = [data_in.buf(), tmp.buf()];
    let view2 = [tmp.buf(), data_out.buf()];

    engine.dispatch([
        (&op, view1),
        (&op, view2)
    ]);

    engine.to_cpu(&mut data_out)?;
    print!("[ ");
    for i in data_out.data {
        print!("{} ", i);
    }
    println!("]");
    Ok(())
}
