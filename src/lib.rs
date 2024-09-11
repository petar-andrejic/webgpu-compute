pub mod channel;

use std::{
    iter::once,
    marker::PhantomData,
    sync::Arc,
    thread::{park, spawn, JoinHandle},
};

use bytemuck::{cast_slice, NoUninit, Pod};
use channel::{once_signal, oneshot, spsc, ChannelError};

#[derive(Debug)]
pub struct Task {
    idx: wgpu::SubmissionIndex,
    finished: once_signal::Sender,
}
#[derive(Debug)]
pub struct Engine {
    pub device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    send_handle: Option<spsc::Sender<Task>>,
    poll_handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
pub struct Operation {
    pub label: String,
    pub pipeline: wgpu::ComputePipeline,
    pub workgroup_size: [u32; 3],
}

#[derive(Debug)]
pub struct Buffer<T> {
    _phantom: PhantomData<T>,
    length: usize,
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

impl<T> Buffer<T> {
    pub fn view(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn size(&self) -> u64 {
        self.buffer.size()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(send_handle) = self.send_handle.take() {
            drop(send_handle);
        } else {
            log::error!("Send handle was missing");
        }
        if let Some(poll_handle) = self.poll_handle.take() {
            poll_handle.thread().unpark();
            if let Err(err) = poll_handle.join() {
                log::error!("Poll loop panicked with error {:?}", err);
            };
        } else {
            log::error!("Poll handle was missing");
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

        let task_device = device.clone();
        let (sx, rx) = spsc::channel::<Task>();
        let poll_handle = spawn(move || loop {
            match rx.try_receive() {
                Ok(task) => {
                    task_device.poll(wgpu::Maintain::wait_for(task.idx));
                    task.finished.send();
                }
                Err(ChannelError::Empty) => park(),
                Err(ChannelError::SenderClosed) => {
                    return;
                }
                Err(_) => unreachable!(),
            }
        });
        Ok(Engine {
            device,
            queue,
            poll_handle: Some(poll_handle),
            send_handle: Some(sx),
        })
    }

    pub async fn submit<I>(&self, command_buffers: I)
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let idx = self.queue.submit(command_buffers);
        let (sx, rx) = once_signal::channel();
        let task = Task { idx, finished: sx };
        self.send_handle
            .as_ref()
            .expect("Missing send handle")
            .send(task);
        self.poll_handle
            .as_ref()
            .expect("Missing poll handle")
            .thread()
            .unpark();
        rx.await;
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

    pub fn create_buffer<T>(&self, length: usize) -> Buffer<T> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            size: (size_of::<T>() * length) as u64,
        });

        Buffer {
            buffer,
            length,
            _phantom: PhantomData {},
        }
    }

    pub async fn load<T: NoUninit + Pod>(&self, data: &[T]) -> Buffer<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: true,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            size: (size_of::<T>() * data.len()) as u64,
        });
        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.copy_from_slice(cast_slice(data));
        }
        staging.unmap();

        let buffer = self.create_buffer(data.len());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, buffer.view(), 0, buffer.size());
        self.submit(once(encoder.finish())).await;
        buffer
    }

    pub async fn save<T: NoUninit + Pod + Clone>(&self, buffer: &Buffer<T>) -> Result<Vec<T>, SyncError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: buffer.size(),
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer.view(), 0, &staging, 0, buffer.size());
        self.submit(once(encoder.finish())).await;
        let (sx, rx) = oneshot::channel();
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            sx.send(result);
        });
        self.device.poll(wgpu::Maintain::Poll);
        rx.await??;
        let view = slice.get_mapped_range();
        Ok(cast_slice(&view[..]).to_vec())
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
        self.submit(Some(encoder.finish())).await;
    }
}
