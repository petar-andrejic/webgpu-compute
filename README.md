# webgpu-compute

While quite nice and ergonomic compared to say Vulkan, webGPU still has a decent amount of boilerplate involved compared with something like
CUDA. In this repo I've sketched out a fairly minimal wrapper for wgpu in Rust around common tasks such as synchronising a GPU to CPU and vice versa, and store
and dispatch pipelines on buffers. This isn't intended to be serious library code, more a helpful resource for getting started on your own
custom wgpu compute code
