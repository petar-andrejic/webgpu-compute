@group(0) @binding(0) var<storage, read> data_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> data_out: array<u32>;

@compute @workgroup_size(1) fn main(
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x;
  data_out[i] = data_in[i] * 2;
}