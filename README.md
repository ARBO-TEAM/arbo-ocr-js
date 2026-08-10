# arbo-ocr-js

Node.js and Bun wrapper for [arboOCR](https://github.com/wafik/ArboOCR) — a
standalone C++ OCR engine (detection, orientation, recognition) built on
PP-OCRv6 ONNX models.

No C++ toolchain, no vcpkg, no CMake, and **no runtime dependencies**. The
package downloads a prebuilt `arboocr_demo` binary on first use and runs it as
a subprocess, parsing its JSON output.

```bash
npm install github:ARBO-TEAM/arbo-ocr-js
# or
bun add github:ARBO-TEAM/arbo-ocr-js
```

> Not on the npm registry yet — install from GitHub. The `prepare` script builds
> `dist/` on install, so a git install works the same as a registry one.

```ts
import { Engine } from "arbo-ocr-js";

const engine = new Engine({ modelType: "small" });
const page = await engine.recognize("receipt.jpg");

console.log(page.backend, `${page.lines.length} lines in ${page.elapsedMs}ms`);
for (const line of page.lines) {
  console.log(line.text, line.score.toFixed(3));
}
```

That is the whole setup. The binary and the OCR model weights both download
themselves on first run and are cached afterwards.

## Requirements

| | |
|---|---|
| Runtime | Node.js ≥ 18, or Bun ≥ 1.0 |
| Platform | Windows x64, Linux x64 |
| Dependencies | none at runtime |

macOS and ARM have no published release asset. See
[Unsupported platforms](#unsupported-platforms).

## How it works

```text
your code
    │  await new Engine({...}).recognize("page.jpg")
    ▼
arbo-ocr-js ── spawn ──► arboocr_demo --image page.jpg --json
    ▲                            │
    └───── JSON on stdout ───────┘
```

This is the same design as the [Python](https://github.com/ARBO-TEAM/arbo-ocr-python),
[Go](https://github.com/ARBO-TEAM/arbo-ocr-go),
[Rust](https://github.com/ARBO-TEAM/arbo-ocr-rust) and
[PHP](https://github.com/ARBO-TEAM/ArboOcrPhp) wrappers — all five pull the
*same* release asset and differ only in how they spawn it.

### The trade-off, stated honestly

You pay a process spawn on every `recognize()` call — roughly **130–320 ms per
image** on top of inference time, because each call reloads the ONNX models
from scratch. There is no warm in-process engine and no batching.

What you get for it: an install that is one line long and works on a machine
with no compiler. That is the right trade for scripting, queue workers, and web
backends where a request already costs tens of milliseconds. If it is not the
right trade for you, use the C++ library directly.

## API

### `new Engine(config?)`

Construction is synchronous and touches nothing. The binary is resolved lazily
on the first `recognize()` or `ensureModels()` call, and the result is memoized,
so concurrent first calls share a single download.

### `engine.recognize(imagePath): Promise<PageResult>`

Runs OCR on one image.

```ts
interface PageResult {
  backend: string;      // "cpu" | "cuda" | "tensorrt" — what actually ran
  image: string;
  elapsedMs: number;
  lines: LineResult[];  // empty is a normal result, not an error
}

interface LineResult {
  text: string;
  score: number;
  detScore: number;
  polygon: Point[];     // 4 points, clockwise from top-left-ish
  words?: WordResult[]; // only when config.wordBoxes is set
}
```

Rejects with an `OcrError` (carrying `exitCode` and `stderr`) if the process
cannot start, exits non-zero, or emits output that is not JSON.

### `engine.ensureModels(): Promise<void>`

Downloads the model weights for this engine's `modelType` and returns. Call it
from a Docker build step so the first `recognize()` does not pay for it.
Idempotent — an already-cached model is a no-op.

### `ensureInstalled(binDir?): Promise<string>`

Downloads the `arboocr_demo` binary if missing and returns its path. Exposed so
you can pull it during a container build and pass the result back as
`binPath` at runtime.

### Config

Every field is optional. **An unset field emits no CLI flag at all**, leaving
the binary's own default in place — which is what lets you point `binPath` at an
older arboOCR release without tripping over flags it does not know.

| Field | Type | Notes |
|---|---|---|
| `binPath` | `string` | Explicit binary path; unset = auto-download |
| `modelsDir` | `string` | A populated directory wins — no network access at all |
| `modelType` | `string` | `tiny` \| `small` (default) \| `medium` |
| `ocrVersion` | `string` | Default `PP-OCRv6` |
| `detModelPath` `clsModelPath` `recModelPath` `dictPath` | `string` | Per-file overrides; never substituted by a download |
| `logLevel` | `string` | `debug` \| `info` \| `warn` \| `error`; default silent |
| `minConfidence` | `number` | Drop lines below this score. CLI default 0.5; `0` disables filtering |
| `recBatchNum` | `number` | Crops per inference call. CLI default 6 |
| `detLimitSideLen` | `number` | Longest side for detection resize. CLI default 960 |
| `useAngleCls` | `boolean` | Rotated-text classification |
| `useCuda` / `useTensorrt` / `useFp16` | `boolean` | See [GPU](#gpu) |
| `useClahe` | `boolean` | Contrast enhancement for faded documents |
| `wordBoxes` | `boolean` | Adds `line.words` — larger JSON, off by default |
| `noDownload` | `boolean` | Fail instead of fetching a missing model |
| `modelsUrl` | `string` | Fetch missing models from an internal mirror |

## Models

**Nothing bundles OCR models, and they download anyway.** Not one line of this
package makes that happen — `arboocr_demo` fetches the stock weights it is
missing, verifies each against a SHA-256 baked into the binary, and writes it
atomically. A mirror serving different bytes is rejected rather than loaded.

Point `modelsDir` at files you already have and nothing touches the network.

| Platform | Model cache |
|---|---|
| Windows | `%LOCALAPPDATA%\arboOCR\models\models-v1` |
| Linux | `$XDG_CACHE_HOME/arboOCR/models/models-v1` (or `~/.cache/...`) |

Set `ARBOOCR_OFFLINE=1` to make a missing model an immediate error instead of a
network call — the setting you want in an air-gapped runtime, where a stalled
socket is indistinguishable from a hang. `ARBOOCR_CACHE_DIR` moves the cache
somewhere writable, which matters when your service runs as a user with no home
directory.

## GPU

```ts
const engine = new Engine({ useCuda: true });
const page = await engine.recognize("scan.png");
console.log(page.backend); // read this — it tells you what actually ran
```

Requesting a provider is not the same as getting one. The engine falls back to
CPU when CUDA or TensorRT is unavailable, so `backend` is the only thing that
confirms which one you got.

This package pins release **v0.3.0**, the first release whose archive ships
`onnxruntime_providers_shared`. Earlier archives could not load a GPU provider
at all on either platform — the engine fell back to CPU silently.

## Binary cache

| Platform | Location |
|---|---|
| Windows | `%LOCALAPPDATA%\arbo-ocr-js\v0.3.0\windows-x64\` |
| Linux | `$XDG_CACHE_HOME/arbo-ocr-js/v0.3.0/linux-x64/` (or `~/.cache/...`) |

The version is a path segment on purpose. The extracted binary has the same
name in every release, so a version-less path would report a stale binary as
current forever and make a version bump a silent no-op.

Extraction shells out to `tar`, which reads both release formats — GNU tar
handles the Linux `.tar.gz`, and Windows ships bsdtar in `System32`, which reads
the `.zip` too. That is two fewer dependencies than an unzip package plus a tar
package.

### Unsupported platforms

On macOS or ARM, `detectPlatform()` returns `null` and auto-install throws.
Build arboOCR yourself or grab a release manually, then:

```ts
const engine = new Engine({ binPath: "/opt/arboocr/arboocr_demo" });
```

## Development

```bash
npm install
npm test        # builds, then runs unit tests

# end-to-end: downloads ~13 MB of binary plus model weights
ARBO_OCR_TEST_IMAGE=/path/to/receipt.jpg npm test
```

## License

Apache-2.0. See [THIRD_PARTY_NOTICES](https://github.com/wafik/ArboOCR/blob/main/THIRD_PARTY_NOTICES.md)
in the arboOCR repository for the bundled runtime libraries.
