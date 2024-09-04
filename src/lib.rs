use log::error;
use std::iter::once;

use bytemuck::{cast_slice, NoUninit, Pod};
use futures::channel::oneshot;
use pollster::FutureExt;
#[derive(Debug)]
pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[derive(Debug)]
pub struct Operation {
    pub label: String,
    pub pipeline: wgpu::ComputePipeline,
    pub workgroup_size: [u32; 3],
}

#[repr(C, align(16))]
#[derive(Debug)]
pub struct GPUArray<T> {
    pub data: Vec<T>,
    buffer: wgpu::Buffer,
}

#[derive(thiserror::Error, Debug)]
pub enum EngineCreationError {
    #[error("Failed to request adapter")]
    RequestAdapter,
    #[error("Failed to request device")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
}

#[derive(thiserror::Error, Debug)]
pub enum SyncError {
    #[error("Failed to map buffer")]
    MapBuffer(#[from] wgpu::BufferAsyncError),
    #[error("Failed to sync channel")]
    Channel(#[from] futures::channel::oneshot::Canceled),
}

#[derive(thiserror::Error, Debug)]
pub enum DispatchError {
    #[error("Invalid buffer type")]
    BufferType,
}

trait Run {
    fn run(&self, task :wgpu::SubmissionIndex);

    fn wait(&self);
}

impl Run for wgpu::Device {
    fn run(&self, task :wgpu::SubmissionIndex) {
        self.poll(wgpu::Maintain::wait_for(task));
    }

    fn wait(&self) {
        self.poll(wgpu::Maintain::Wait);
    }
}
impl<T> GPUArray<T> {
    pub fn buf(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Engine {
    pub fn new() -> Result<Self, EngineCreationError> {
        let instance = wgpu::Instance::new(Default::default());

        let adapter = instance
            .request_adapter(&Default::default())
            .block_on()
            .ok_or(EngineCreationError::RequestAdapter)?;

        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .block_on()?;

        Ok(Engine { device, queue })
    }

    pub fn create_operation(&self, module: &wgpu::ShaderModule, label: String) -> Operation {
        let pipeline_label = label.clone() + "_pipeline";
        let pipeline_desc = wgpu::ComputePipelineDescriptor {
            label: Some(pipeline_label.as_str()),
            layout: None,
            module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        };
        let pipeline = self.device.create_compute_pipeline(&pipeline_desc);

        Operation {
            label,
            pipeline,
            workgroup_size: [64, 64, 1],
        }
    }

    pub fn bind_vec<T>(&self, data: Vec<T>) -> GPUArray<T> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            size: (size_of::<T>() * data.len()) as u64,
        });
        GPUArray { buffer, data }
    }

    pub fn create_gpu_vec<T: Pod>(&self, size: usize) -> GPUArray<T> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            size: (size_of::<T>() * size) as u64,
        });
        let data = vec![T::zeroed(); size];
        GPUArray { buffer, data }
    }

    pub fn bind<'item, I>(&self, op: &'item Operation, buffers: I) -> wgpu::BindGroup
    where
        I: IntoIterator<Item = &'item wgpu::Buffer>,
    {
        let layout = &op.pipeline.get_bind_group_layout(0);
        let bindings: Vec<_> = buffers
            .into_iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group_desc = wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &bindings,
        };

        self.device.create_bind_group(&bind_group_desc)
    }

    pub fn dispatch<'item, I1, I2>(&self, args: I1)
    where
        I1: IntoIterator<Item = (&'item Operation, I2)>,
        I2: IntoIterator<Item = &'item wgpu::Buffer>,
    {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for (op, buffers) in args {
            let binding = self.bind(op, buffers);
            pass.set_bind_group(0, &binding, &[]);
            pass.set_pipeline(&op.pipeline);
            pass.dispatch_workgroups(
                op.workgroup_size[0],
                op.workgroup_size[1],
                op.workgroup_size[2],
            );
        }
        drop(pass);
        let task = self.queue.submit(Some(encoder.finish()));
        self.device.run(task);
    }

    pub fn to_gpu<T: NoUninit + Pod>(&self, x: &GPUArray<T>) {
        // Create staging buffer
        let staging_desc = wgpu::BufferDescriptor {
            label: None,
            size: x.buffer.size(),
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        };
        let staging = self.device.create_buffer(&staging_desc);

        // Copy CPU -> staging
        let mut view = staging.slice(..).get_mapped_range_mut();
        view.copy_from_slice(cast_slice(&x.data[..]));
        drop(view);
        staging.unmap();

        // Copy staging -> buffer
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &x.buffer, 0, x.buffer.size());
        let task = self.queue.submit(once(encoder.finish()));
        self.device.run(task);
    }

    pub fn to_cpu<T: NoUninit + Pod>(&self, x: &mut GPUArray<T>) -> Result<(), SyncError> {
        // Create staging buffer
        let staging_desc = wgpu::BufferDescriptor {
            label: None,
            size: x.buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let staging = self.device.create_buffer(&staging_desc);

        // Copy buffer -> staging
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&x.buffer, 0, &staging, 0, x.buffer.size());
        let task = self.queue.submit(once(encoder.finish()));
        self.device.run(task);

        // Copy staging -> CPU
        let slice = staging.slice(..);
        let (sx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| match sx.send(result) {
            Ok(()) => (),
            Err(err) => error!("Failed to send map result: {:?}", err),
        });
        self.device.wait();
        rx.block_on()??;
        let data_raw = slice.get_mapped_range();
        x.data.copy_from_slice(cast_slice(&data_raw[..]));

        Ok(())
    }
}
