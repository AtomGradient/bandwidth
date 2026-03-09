import SwiftUI
import Metal

// ─── Metal Shader (embedded as string, no file I/O needed) ────────────────────
private let METAL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void stream_copy(
    device const float4* src [[buffer(0)]],
    device       float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src[gid];
}

kernel void stream_triad(
    device const float4* a      [[buffer(0)]],
    device const float4* b      [[buffer(1)]],
    device       float4* dst    [[buffer(2)]],
    constant     float&  scalar [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = a[gid] + scalar * b[gid];
}

kernel void stream_write(
    device float4* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = float4(1.0f, 2.0f, 3.0f, 4.0f);
}

kernel void stream_read(
    device const float4* src  [[buffer(0)]],
    device       float4* sink [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float4* tg [[threadgroup(0)]]
) {
    tg[tid] = src[gid];
    if (tid == 0) sink[0] = tg[0];
}
"""

// ─── Config ───────────────────────────────────────────────────────────────────
private let BUFFER_MB   = 512          // MB per buffer (keep ≤512 on iPad)
private let ITERATIONS  = 15
private let TG_SIZE     = 1024

// ─── Result model ─────────────────────────────────────────────────────────────
struct BenchResult: Identifiable {
    let id = UUID()
    let label: String
    let emoji: String
    let best: Double
    let avg: Double
    let theoretical: Double            // used for bar chart
}

// ─── Benchmark engine ─────────────────────────────────────────────────────────
@MainActor
class BenchmarkVM: ObservableObject {
    @Published var results: [BenchResult] = []
    @Published var log: [String] = []
    @Published var running = false
    @Published var progress: Double = 0
    @Published var deviceName = ""

    private var device: MTLDevice?
    private var queue:  MTLCommandQueue?
    private var library: MTLLibrary?

    func run() async {
        running  = true
        results  = []
        log      = []
        progress = 0

        guard let dev = MTLCreateSystemDefaultDevice() else {
            log.append("❌ No Metal device"); running = false; return
        }
        device = dev
        deviceName = dev.name
        queue  = dev.makeCommandQueue()

        do {
            library = try dev.makeLibrary(source: METAL_SRC, options: nil)
        } catch {
            log.append("❌ Shader compile failed: \(error)"); running = false; return
        }

        let elementSize = MemoryLayout<SIMD4<Float>>.stride   // 16 bytes
        let bufBytes    = BUFFER_MB * 1_000_000
        let count       = bufBytes / elementSize

        append("Allocating \(BUFFER_MB * 3) MB unified memory…")

        guard
            let bufA = dev.makeBuffer(length: bufBytes, options: .storageModeShared),
            let bufB = dev.makeBuffer(length: bufBytes, options: .storageModeShared),
            let bufC = dev.makeBuffer(length: bufBytes, options: .storageModeShared)
        else { append("❌ Allocation failed"); running = false; return }

        // init
        let pA = bufA.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        let pB = bufB.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        for i in 0..<count {
            pA[i] = SIMD4<Float>(1, 2, 3, 4)
            pB[i] = SIMD4<Float>(0.5, 0.5, 0.5, 0.5)
        }
        var scalar: Float = 2.0
        let scalarBuf = dev.makeBuffer(bytes: &scalar, length: 4, options: .storageModeShared)!

        append("Warming up GPU…\n")

        let tasks: [(String, String, Int, (MTLComputeCommandEncoder) -> Void)] = [
            ("COPY",  "📋", bufBytes * 2, { enc in
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufC, offset: 0, index: 1)
            }),
            ("TRIAD", "⚡️", bufBytes * 3, { enc in
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufB, offset: 0, index: 1)
                enc.setBuffer(bufC, offset: 0, index: 2)
                enc.setBuffer(scalarBuf, offset: 0, index: 3)
            }),
            ("READ",  "👁️",  bufBytes,     { enc in
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufC, offset: 0, index: 1)
                enc.setThreadgroupMemoryLength(TG_SIZE * elementSize, index: 0)
            }),
            ("WRITE", "✍️",  bufBytes,     { enc in
                enc.setBuffer(bufC, offset: 0, index: 0)
            }),
        ]

        let kernelNames = ["stream_copy", "stream_triad", "stream_read", "stream_write"]

        for (idx, (label, emoji, bytesPerIter, setup)) in tasks.enumerated() {
            let kernelName = kernelNames[idx]
            guard let fn = library?.makeFunction(name: kernelName),
                  let pipe = try? dev.makeComputePipelineState(function: fn)
            else { append("❌ Pipeline failed: \(kernelName)"); continue }

            var best = 0.0
            var total = 0.0

            // warmup
            for _ in 0..<3 {
                let cmd = queue!.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipe)
                setup(enc)
                enc.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }

            for i in 0..<ITERATIONS {
                let cmd = queue!.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipe)
                setup(enc)
                enc.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: TG_SIZE, height: 1, depth: 1))
                enc.endEncoding()

                // Use Metal timestamps for accuracy on-device
                let t0 = CACurrentMediaTime()
                cmd.commit()
                cmd.waitUntilCompleted()
                let elapsed = CACurrentMediaTime() - t0

                let gbps = Double(bytesPerIter) / elapsed / 1e9
                best   = max(best, gbps)
                total += gbps

                let p = (Double(idx) + Double(i + 1) / Double(ITERATIONS)) / Double(tasks.count)
                progress = p
                await Task.yield()
            }

            let avg = total / Double(ITERATIONS)
            append("\(emoji) \(label)  best: \(String(format: "%.1f", best)) GB/s   avg: \(String(format: "%.1f", avg)) GB/s")

            results.append(BenchResult(
                label: label, emoji: emoji,
                best: best, avg: avg,
                theoretical: 800          // change for your chip's spec
            ))
        }

        append("\n✅ Done. TRIAD best ≈ closest to true peak.")
        progress = 1.0
        running  = false
    }

    private func append(_ s: String) {
        log.append(s)
    }
}

// ─── Views ────────────────────────────────────────────────────────────────────
struct ResultRow: View {
    let r: BenchResult
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(r.emoji + " " + r.label)
                    .font(.system(.headline, design: .monospaced))
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f GB/s", r.best))
                        .font(.system(.title3, design: .monospaced).bold())
                        .foregroundColor(.cyan)
                    Text(String(format: "avg %.1f", r.avg))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            // Bar — best vs theoretical
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.white.opacity(0.08))
                        .frame(height: 8)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(LinearGradient(
                            colors: [.cyan, .blue],
                            startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * min(r.best / r.theoretical, 1.0), height: 8)
                }
            }
            .frame(height: 8)
            Text(String(format: "%.0f%% of %.0f GB/s spec", r.best / r.theoretical * 100, r.theoretical))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.white.opacity(0.05))
        .cornerRadius(12)
    }
}

struct ContentView: View {
    @StateObject private var vm = BenchmarkVM()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {

                    // Header card
                    VStack(spacing: 4) {
                        Text("🔬 Metal Bandwidth")
                            .font(.largeTitle.bold())
                        if !vm.deviceName.isEmpty {
                            Text(vm.deviceName)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        Text("Buffer: \(BUFFER_MB) MB × 3 arrays • \(ITERATIONS) iterations")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top)

                    // Run button + progress
                    VStack(spacing: 8) {
                        Button(action: {
                            Task { await vm.run() }
                        }) {
                            Label(vm.running ? "Running…" : "Run Benchmark",
                                  systemImage: vm.running ? "arrow.2.circlepath" : "bolt.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(vm.running ? Color.gray : Color.cyan)
                                .foregroundColor(.black)
                                .cornerRadius(14)
                                .font(.headline)
                        }
                        .disabled(vm.running)

                        if vm.running {
                            ProgressView(value: vm.progress)
                                .tint(.cyan)
                        }
                    }
                    .padding(.horizontal)

                    // Results
                    if !vm.results.isEmpty {
                        VStack(spacing: 10) {
                            ForEach(vm.results) { r in
                                ResultRow(r: r)
                            }
                        }
                        .padding(.horizontal)
                    }

                    // Log
                    if !vm.log.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Log")
                                .font(.caption.bold())
                                .foregroundColor(.secondary)
                            ForEach(Array(vm.log.enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.green)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding()
                        .background(Color.black.opacity(0.4))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    }

                    Spacer(minLength: 40)
                }
            }
            .background(Color.black.ignoresSafeArea())
            .foregroundColor(.white)
            .navigationBarHidden(true)
        }
    }
}

@main
struct MetalBandwidthApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
