import Metal
import Foundation

// ─── Config ───────────────────────────────────────────────────────────────────
let BUFFER_GB: Double = 1.0          // GB per buffer (increase if you have RAM)
let ITERATIONS: Int  = 20            // benchmark repetitions per kernel
let THREADGROUP_SIZE: Int = 1024     // optimal for Apple Silicon

// ─── Setup ────────────────────────────────────────────────────────────────────
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device found")
}

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Metal Memory Bandwidth Benchmark")
print("  Device : \(device.name)")
print("  Buffer : \(BUFFER_GB) GB per array")
print("  Passes : \(ITERATIONS)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

// Compile shaders from the .metal file alongside this script
let metalSrc = try! String(contentsOfFile: "bandwidth.metal", encoding: .utf8)
let library = try! device.makeLibrary(source: metalSrc, options: nil)

func makePipeline(_ name: String) -> MTLComputePipelineState {
    let fn = library.makeFunction(name: name)!
    return try! device.makeComputePipelineState(function: fn)
}

let pipeCopy   = makePipeline("stream_copy")
let pipeTriad  = makePipeline("stream_triad")
let pipeRead   = makePipeline("stream_read")
let pipeWrite  = makePipeline("stream_write")

let queue = device.makeCommandQueue()!

// ─── Allocate buffers (shared / unified memory) ───────────────────────────────
let elementSize = MemoryLayout<SIMD4<Float>>.stride  // float4 = 16 bytes
let bufferBytes = Int(BUFFER_GB * 1_000_000_000)
let elementCount = bufferBytes / elementSize

print("Allocating \(String(format: "%.0f", BUFFER_GB * 3)) GB of shared memory...")

let bufA = device.makeBuffer(length: bufferBytes, options: .storageModeShared)!
let bufB = device.makeBuffer(length: bufferBytes, options: .storageModeShared)!
let bufC = device.makeBuffer(length: bufferBytes, options: .storageModeShared)!

// Init buffers
let pA = bufA.contents().bindMemory(to: SIMD4<Float>.self, capacity: elementCount)
let pB = bufB.contents().bindMemory(to: SIMD4<Float>.self, capacity: elementCount)
for i in 0..<elementCount {
    pA[i] = SIMD4<Float>(1, 2, 3, 4)
    pB[i] = SIMD4<Float>(0.5, 0.5, 0.5, 0.5)
}

var scalar: Float = 2.0
let scalarBuf = device.makeBuffer(bytes: &scalar, length: 4, options: .storageModeShared)!

print("Buffers ready. Warming up GPU...\n")

// ─── Benchmark helper ─────────────────────────────────────────────────────────
func benchmark(
    label: String,
    bytesPerIter: Int,
    setup: (MTLComputeCommandEncoder) -> Void
) {
    // Warmup
    for _ in 0..<3 {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        setup(enc)
        enc.dispatchThreads(
            MTLSize(width: elementCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: THREADGROUP_SIZE, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // Timed runs
    var best: Double = 0
    var total: Double = 0

    for _ in 0..<ITERATIONS {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        setup(enc)
        enc.dispatchThreads(
            MTLSize(width: elementCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: THREADGROUP_SIZE, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmd.commit()

        let t0 = Date()
        cmd.waitUntilCompleted()
        let elapsed = Date().timeIntervalSince(t0)

        let gbps = Double(bytesPerIter) / elapsed / 1e9
        best  = max(best, gbps)
        total += gbps
    }

    let avg = total / Double(ITERATIONS)
    print(String(format: "  %-20s  best: %6.1f GB/s   avg: %6.1f GB/s", label, best, avg))
}

print("Running benchmarks:")
print(String(repeating: "─", count: 56))

// Copy: read A, write C  → 2 × buffer bytes
benchmark(label: "COPY  (R+W)", bytesPerIter: bufferBytes * 2) { enc in
    enc.setComputePipelineState(pipeCopy)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufC, offset: 0, index: 1)
}

// Triad: read A+B, write C  → 3 × buffer bytes (most demanding)
benchmark(label: "TRIAD (2R+W)", bytesPerIter: bufferBytes * 3) { enc in
    enc.setComputePipelineState(pipeTriad)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    enc.setBuffer(scalarBuf, offset: 0, index: 3)
}

// Read-only: read A  → 1 × buffer bytes
benchmark(label: "READ  (R)", bytesPerIter: bufferBytes) { enc in
    enc.setComputePipelineState(pipeRead)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufC, offset: 0, index: 1)
    enc.setThreadgroupMemoryLength(THREADGROUP_SIZE * elementSize, index: 0)
}

// Write-only: write C  → 1 × buffer bytes
benchmark(label: "WRITE (W)", bytesPerIter: bufferBytes) { enc in
    enc.setComputePipelineState(pipeWrite)
    enc.setBuffer(bufC, offset: 0, index: 0)
}

print(String(repeating: "─", count: 56))
print("\nDone. TRIAD best ≈ closest to real peak bandwidth.")
print("M2 Ultra theoretical max: 800 GB/s\n")
