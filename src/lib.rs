pub mod channel;

use std::{
    iter::once,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, sleep},
    time::Duration,
};

use channel::{once_signal, oneshot};
use bytemuck::{cast_slice, NoUninit, Pod};
use log::error;

// use pollster::FutureExt;
#[derive(Debug)]
pub struct Engine {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    _close_handle: CloseHandle,
}
#[derive(Debug)]
pub struct Operation {
    pub label: String,
    pub pipeline: wgpu::ComputePipeline,
    pub workgroup_size: [u32; 3],
}

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
    #[error("Channel failed")]
    Channel(#[from] channel::ChannelError),
}

trait AsyncQueue {
    async fn submit_async<I>(&self, command_buffers: I)
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>;
}

impl AsyncQueue for wgpu::Queue {
    async fn submit_async<I>(&self, command_buffers: I)
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let (sx, rx) = once_signal::channel();
        self.submit(command_buffers);
        self.on_submitted_work_done(move || sx.send());
        rx.await;
    }
}

trait AsyncBufferSlice {
    async fn map_read(&self) -> Result<(), SyncError>;
    async fn map_write(&self) -> Result<(), SyncError>;
    async fn map_future(&self, mode: wgpu::MapMode) -> Result<(), SyncError>;
}

impl AsyncBufferSlice for wgpu::BufferSlice<'_> {
    async fn map_read(&self) -> Result<(), SyncError> {
        self.map_future(wgpu::MapMode::Read).await
    }

    async fn map_write(&self) -> Result<(), SyncError> {
        self.map_future(wgpu::MapMode::Write).await
    }

    async fn map_future(&self, mode: wgpu::MapMode) -> Result<(), SyncError> {
        let (sx, rx) = oneshot::channel();
        self.map_async(mode, move |result| {
            sx.send(result);
        });
        rx.await??;
        Ok(())
    }
}

#[derive(Debug)]
struct CloseHandle {
    alive: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for CloseHandle {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
        self.handle.take().map(|h| match h.join() {
            Ok(()) => (),
            Err(err) => {
                error!("Thread panicked while closing handle: {:?}", err)
            }
        });
    }
}

impl<T> GPUArray<T> {
    pub fn buf(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

trait AsyncDevice {
    fn create_poll_loop(self: Arc<Self>, wait_duration: Duration) -> CloseHandle;
}

impl AsyncDevice for wgpu::Device {
    fn create_poll_loop(self: Arc<Self>, wait_duration: Duration) -> CloseHandle {
        let task_alive = Arc::<AtomicBool>::new(true.into());
        let handle_alive = task_alive.clone();
        let join_handle = thread::spawn(move || {
            while task_alive.load(Ordering::Acquire) {
                self.poll(wgpu::Maintain::Poll);
                sleep(wait_duration);
            }
        });

        CloseHandle {
            alive: handle_alive,
            handle: Some(join_handle),
        }
    }
}

impl Engine {
    pub async fn new() -> Result<Self, EngineCreationError> {
        let instance = wgpu::Instance::new(Default::default());

        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .ok_or(EngineCreationError::RequestAdapter)?;

        let (raw_device, raw_queue) = adapter.request_device(&Default::default(), None).await?;
        let device = Arc::new(raw_device);
        let queue = Arc::new(raw_queue);
        let close_handle = device.clone().create_poll_loop(Duration::from_millis(100));
        Ok(Engine {
            device,
            queue,
            _close_handle: close_handle,
        })
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

    pub fn create_buffer<T>(&self, length: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            size: (size_of::<T>() * length) as u64,
        })
    }

    pub fn bind_vec<T>(&self, data: Vec<T>) -> GPUArray<T> {
        let buffer = self.create_buffer::<T>(data.len());
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

    pub async fn dispatch<'item, I1, I2>(&self, args: I1)
    where
        I1: IntoIterator<Item = (&'item Operation, I2)>,
        I2: IntoIterator<Item = &'item wgpu::Buffer>,
    {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
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
        }
        self.queue.submit_async(Some(encoder.finish())).await;
    }

    pub async fn to_gpu<T: NoUninit + Pod>(&self, x: &GPUArray<T>) {
        // Create staging buffer
        let staging_desc = wgpu::BufferDescriptor {
            label: None,
            size: x.buffer.size(),
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        };
        let staging = self.device.create_buffer(&staging_desc);

        // Copy CPU -> staging
        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.copy_from_slice(cast_slice(&x.data[..]));
        }
        staging.unmap();

        // Copy staging -> buffer
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &x.buffer, 0, x.buffer.size());
        self.queue.submit_async(once(encoder.finish())).await;
    }

    pub async fn to_cpu<T: NoUninit + Pod>(&self, x: &mut GPUArray<T>) -> Result<(), SyncError> {
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
        self.queue.submit_async(once(encoder.finish())).await;

        // Copy staging -> CPU
        let slice = staging.slice(..);
        slice.map_read().await?;
        let data_raw = slice.get_mapped_range();
        x.data.copy_from_slice(cast_slice(&data_raw[..]));

        Ok(())
    }
}
