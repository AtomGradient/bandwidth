#include <metal_stdlib>
using namespace metal;

// STREAM Copy: dst = src  (2 arrays, read+write)
kernel void stream_copy(
    device const float4* src [[buffer(0)]],
    device       float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src[gid];
}

// STREAM Triad: dst = src_a + scalar * src_b  (3 arrays, 2 reads + 1 write)
kernel void stream_triad(
    device const float4* src_a  [[buffer(0)]],
    device const float4* src_b  [[buffer(1)]],
    device       float4* dst    [[buffer(2)]],
    constant     float&  scalar [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src_a[gid] + scalar * src_b[gid];
}

// Vectorized read-only (max read BW)
kernel void stream_read(
    device const float4* src [[buffer(0)]],
    device       float4* sink [[buffer(1)]],  // write 1 element to prevent optimization
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float4* tg [[threadgroup(0)]]
) {
    tg[tid] = src[gid];
    // Reduction to prevent dead-code elimination
    if (tid == 0) sink[0] = tg[0];
}

// Write-only (max write BW)
kernel void stream_write(
    device float4* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = float4(1.0f, 2.0f, 3.0f, 4.0f);
}
