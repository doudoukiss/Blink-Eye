import AppKit
import AVFoundation
import CoreMedia
import CoreVideo
import Darwin
import Foundation

struct HelperConfig {
    let stateDir: URL
    let framerate: Double
    let maxWidth: Int

    static func parse(arguments: [String]) -> HelperConfig {
        var stateDir: URL?
        var framerate = 1.0
        var maxWidth = 640
        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "--state-dir", index + 1 < arguments.count {
                stateDir = URL(fileURLWithPath: arguments[index + 1], isDirectory: true)
                index += 2
            } else if argument == "--framerate", index + 1 < arguments.count {
                framerate = max(0.1, Double(arguments[index + 1]) ?? 1.0)
                index += 2
            } else if argument == "--max-width", index + 1 < arguments.count {
                maxWidth = max(64, Int(arguments[index + 1]) ?? 640)
                index += 2
            } else {
                index += 1
            }
        }

        let fallbackDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("blink-camera-helper-\(getpid())", isDirectory: true)
        return HelperConfig(
            stateDir: stateDir ?? fallbackDir,
            framerate: framerate,
            maxWidth: maxWidth
        )
    }
}

struct HelperStatus: Encodable {
    let state: String
    let updatedAt: String
    let frameSeq: Int
    let framePath: String
    let width: Int
    let height: Int
    let format: String
    let pid: Int32
    let reasonCodes: [String]

    enum CodingKeys: String, CodingKey {
        case state
        case updatedAt = "updated_at"
        case frameSeq = "frame_seq"
        case framePath = "frame_path"
        case width
        case height
        case format
        case pid
        case reasonCodes = "reason_codes"
    }
}

final class StatusWriter {
    private let stateDir: URL
    private let encoder = JSONEncoder()
    private let formatter = ISO8601DateFormatter()

    init(stateDir: URL) {
        self.stateDir = stateDir
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    }

    func prepare() throws {
        try FileManager.default.createDirectory(
            at: stateDir,
            withIntermediateDirectories: true
        )
    }

    func write(
        state: String,
        frameSeq: Int = 0,
        width: Int = 0,
        height: Int = 0,
        reasonCodes: [String] = []
    ) {
        let status = HelperStatus(
            state: state,
            updatedAt: formatter.string(from: Date()),
            frameSeq: frameSeq,
            framePath: "latest.rgb",
            width: width,
            height: height,
            format: "RGB",
            pid: getpid(),
            reasonCodes: reasonCodes
        )
        do {
            let data = try encoder.encode(status)
            try data.write(to: stateDir.appendingPathComponent("status.json"), options: .atomic)
        } catch {
            fputs("BlinkCameraHelper could not write status\n", stderr)
        }
    }

    func writeFrame(_ data: Data) throws {
        try data.write(to: stateDir.appendingPathComponent("latest.rgb"), options: .atomic)
    }
}

final class CameraFrameDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let config: HelperConfig
    private let writer: StatusWriter
    private var frameSeq = 0
    private var lastFrameTime = Date.distantPast

    init(config: HelperConfig, writer: StatusWriter) {
        self.config = config
        self.writer = writer
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = Date()
        let minimumInterval = 1.0 / max(0.1, config.framerate)
        if now.timeIntervalSince(lastFrameTime) < minimumInterval {
            return
        }
        lastFrameTime = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            writer.write(state: "error", reasonCodes: ["pixel_buffer_missing"])
            return
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            writer.write(state: "error", reasonCodes: ["pixel_buffer_base_missing"])
            return
        }

        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        guard sourceWidth > 0, sourceHeight > 0 else {
            writer.write(state: "error", reasonCodes: ["pixel_buffer_size_invalid"])
            return
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let targetWidth = min(config.maxWidth, sourceWidth)
        let targetHeight = max(1, Int(Double(sourceHeight) * Double(targetWidth) / Double(sourceWidth)))
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rgb = Data(count: targetWidth * targetHeight * 3)

        rgb.withUnsafeMutableBytes { outputBuffer in
            guard let destination = outputBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return
            }
            for y in 0..<targetHeight {
                let sourceY = y * sourceHeight / targetHeight
                for x in 0..<targetWidth {
                    let sourceX = x * sourceWidth / targetWidth
                    let sourceOffset = sourceY * bytesPerRow + sourceX * 4
                    let targetOffset = (y * targetWidth + x) * 3
                    destination[targetOffset] = source[sourceOffset + 2]
                    destination[targetOffset + 1] = source[sourceOffset + 1]
                    destination[targetOffset + 2] = source[sourceOffset]
                }
            }
        }

        do {
            try writer.writeFrame(rgb)
            frameSeq += 1
            writer.write(state: "running", frameSeq: frameSeq, width: targetWidth, height: targetHeight)
        } catch {
            writer.write(state: "error", reasonCodes: ["frame_write_failed"])
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let config = HelperConfig.parse(arguments: CommandLine.arguments)
    private lazy var writer = StatusWriter(stateDir: config.stateDir)
    private var session: AVCaptureSession?
    private var frameDelegate: CameraFrameDelegate?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        do {
            try writer.prepare()
        } catch {
            fputs("BlinkCameraHelper could not prepare state directory\n", stderr)
            NSApp.terminate(nil)
            return
        }
        writer.write(state: "starting")
        requestCameraAccess()
    }

    func applicationWillTerminate(_ notification: Notification) {
        writer.write(state: "stopped", reasonCodes: ["helper_stopped"])
    }

    private func requestCameraAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startCapture()
        case .notDetermined:
            writer.write(state: "awaiting_permission", reasonCodes: ["camera_permission_pending"])
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.startCapture()
                    } else {
                        self.writer.write(state: "denied", reasonCodes: ["camera_permission_denied"])
                    }
                }
            }
        case .denied, .restricted:
            writer.write(state: "denied", reasonCodes: ["camera_permission_denied"])
        @unknown default:
            writer.write(state: "error", reasonCodes: ["camera_permission_unknown"])
        }
    }

    private func startCapture() {
        guard let device = AVCaptureDevice.default(for: .video) else {
            writer.write(state: "error", reasonCodes: ["camera_device_missing"])
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            let captureSession = AVCaptureSession()
            captureSession.sessionPreset = .medium
            guard captureSession.canAddInput(input) else {
                writer.write(state: "error", reasonCodes: ["camera_input_unavailable"])
                return
            }
            captureSession.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            let delegate = CameraFrameDelegate(config: config, writer: writer)
            output.setSampleBufferDelegate(delegate, queue: DispatchQueue(label: "ai.blink.camera-helper.frames"))
            guard captureSession.canAddOutput(output) else {
                writer.write(state: "error", reasonCodes: ["camera_output_unavailable"])
                return
            }
            captureSession.addOutput(output)

            session = captureSession
            frameDelegate = delegate
            captureSession.startRunning()
            writer.write(state: "running", reasonCodes: ["waiting_for_first_frame"])
        } catch {
            writer.write(state: "error", reasonCodes: ["camera_start_failed"])
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
