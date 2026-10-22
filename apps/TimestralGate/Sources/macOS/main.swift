// main.swift — TimestralGateMac: the same gate on the Mac, from the command line — the dry run that shows the harness
// computes what the Python engine computes before it goes to the phone, and the Swift runtime's Mac timing. Only
// macOS bundles: a JIT .aimodel or an h16c .aimodelc (an .aimodelc for another architecture is refused before the load).
//
//   TimestralGateMac --root <dir with out/> --inputs <inputs dir> --mode gate --bundle <path> --compute gpu --tag <t>
//
// Progress goes to stderr and <root>/out/<tag>/log.txt, the output folder's path to stdout. Exit 0 = DONE, 1 = error.

import Foundation

let arguments = Array(CommandLine.arguments.dropFirst())
do {
    let options = try GateOptions.parse(arguments, root: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    let log = try GateLog(directory: options.out) { line in
        FileHandle.standardError.write(Data(line.utf8))
    }
    do {
        let done = try await Gate(options: options, log: log).run()
        log.write("DONE \(done)")
        print(options.out.path)
        exit(0)
    } catch {
        log.write("ERROR: \(error)")
        exit(1)
    }
} catch {
    FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
    exit(1)
}
