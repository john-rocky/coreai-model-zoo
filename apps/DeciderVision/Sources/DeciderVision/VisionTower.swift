// VisionTower — one fixed-grid tower graph (`.aimodel` for the runtime's JIT, `.aimodelc` for an AOT asset):
// patches float32 [4 G^2, 1536] -> image_embeds float32 [G^2, 2048], G = 8 (g256) or 14 (g448). Stateless.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public final class VisionTower: @unchecked Sendable {
    public static let hidden = 2048

    public let grid: Int
    public let url: URL
    /// AIModel(contentsOf:) + loadFunction, in seconds.
    public let loadSeconds: Double
    public let descriptor: [String: Any]
    private let function: InferenceFunction
    private let patchesDescriptor: NDArrayDescriptor
    private let outputName = "image_embeds"

    public init(contentsOf url: URL, grid: Int, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let name = model.functionNames.first, let fd = model.functionDescriptor(for: name),
              let fn = try model.loadFunction(named: name)
        else { throw DeciderVisionError.contract("tower \(url.lastPathComponent): no function") }
        loadSeconds = secondsSince(t0)
        let n = 4 * grid * grid
        guard fd.stateNames.isEmpty, fd.inputNames == ["patches"], fd.outputNames == [outputName],
              let pin = ND.descriptor(fd.inputDescriptor(of: "patches")),
              pin.shape == [n, ImagePreprocess.patchVector], pin.scalarType == .float32,
              let out = TensorSpec.of(fd.outputDescriptor(of: outputName)),
              out == TensorSpec(shape: [grid * grid, Self.hidden], type: .float32)
        else {
            throw DeciderVisionError.contract(
                "tower \(url.lastPathComponent) is not patches f32 [\(n), 1536] -> image_embeds f32 [\(grid * grid), 2048]: \(describe(fd))")
        }
        self.grid = grid
        self.url = url
        self.function = fn
        self.patchesDescriptor = pin
        self.descriptor = describe(fd)
    }

    /// patches [4 G^2 * 1536] -> image_embeds [G^2 * 2048], row-major.
    public func encode(patches: [Float]) async throws -> [Float] {
        let input = ND.make(patches, patchesDescriptor)
        var outputs = try await function.run(inputs: ["patches": input])
        guard let array = outputs.remove(outputName)?.ndArray else {
            throw DeciderVisionError.contract("tower: no \(outputName) in the outputs")
        }
        let emb = ND.read(array, as: Float.self)
        guard emb.count == grid * grid * Self.hidden else {
            throw DeciderVisionError.contract("tower: \(emb.count) values for \(grid * grid) x \(Self.hidden)")
        }
        return emb
    }
}
