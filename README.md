# arbo-ocr-js

[![npm](https://img.shields.io/npm/v/arbo-ocr-js?style=flat-square)](https://www.npmjs.com/package/arbo-ocr-js)
[![license](https://img.shields.io/npm/l/arbo-ocr-js?style=flat-square)](LICENSE)

**Extract text from images in Node.js and Bun.** Text detection, orientation
correction, and recognition, on CPU or GPU.

No compiler, no `node-gyp`, no Python, and **zero runtime dependencies**.

```bash
npm install arbo-ocr-js     # or: bun add arbo-ocr-js
```

```ts
import { Engine } from "arbo-ocr-js";

const engine = new Engine({ modelType: "small" });
const page = await engine.recognize("receipt.jpg");

console.log(page.backend, `${page.lines.length} lines in ${page.elapsedMs}ms`);
for (const line of page.lines) {
  console.log(line.text, line.score.toFixed(3));
}
```

That is the whole setup — no model files to download by hand, no service to
run. The engine binary and the OCR weights fetch themselves on first use and
are cached afterwards.

Each line comes back with its text, a confidence score, and a 4-point polygon;
set `wordBoxes` for per-word boxes (per-character for CJK).

Under the hood it drives [arboOCR](https://github.com/wafik/ArboOCR), a
standalone C++ engine running PP-OCRv6 ONNX models, as a subprocess and parses
its JSON. That design is why there is nothing to build — and it has a real
cost, [stated plainly below](#the-trade-off-stated-honestly).

## Languages

The recognizer weights are multilingual in a single file — **50 languages**:
Chinese, English, Japanese, and 46 Latin-script languages. `tiny` covers the
same set **without** Japanese.

There is deliberately no `language` option. The model already handles all of
them, so a language selector would either do nothing or invite you to pick the
wrong one.

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

    │  await new Engine({...}).recognizeBatch(["a.jpg", "b.jpg"])
    ▼
arbo-ocr-js ── spawn ──► arboocr_demo --images-from list.txt --json
    ▲             once            │   (models load once, not per image)
    └──── JSON array on stdout ───┘
```

This is the same design as the [Python](https://github.com/ARBO-TEAM/arbo-ocr-python),
[Go](https://github.com/ARBO-TEAM/arbo-ocr-go),
[Rust](https://github.com/ARBO-TEAM/arbo-ocr-rust) and
[PHP](https://github.com/ARBO-TEAM/ArboOcrPhp) wrappers — all five pull the
*same* release asset and differ only in how they spawn it.

### The trade-off, stated honestly

You pay a process spawn on every `recognize()` call — roughly **130–320 ms per
image** on top of inference time, because each call reloads the ONNX models
from scratch. There is no warm in-process engine.

**If you have more than one image, use [`recognizeBatch`](#enginerecognizebatchimagepaths-promisepageresult)** —
it spawns once for the whole list, so the model load is paid once instead of N
times. Measured on 20 SROIE receipts at `modelType: "tiny"`:

| | total | per image |
|---|---:|---:|
| `recognize()` in a loop | 7290 ms | 364 ms |
| `recognizeBatch()` | **4365 ms** | **218 ms** |

1.67× faster, byte-identical text. 218 ms/image is essentially the engine's own
inference time, so batching removes nearly all of the wrapper's overhead.

What you get for the subprocess design: an install that is one line long and
works on a machine with no compiler. That is the right trade for scripting,
queue workers, and web backends where a request already costs tens of
milliseconds. If you need a warm engine for *single* images on a hot path, use
the C++ library directly.

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

### `engine.recognizeBatch(imagePaths): Promise<PageResult[]>`

Runs OCR on many images in **one** process. Prefer this whenever you have more
than one image — see [the trade-off](#the-trade-off-stated-honestly) for the
numbers.

```ts
const pages = await engine.recognizeBatch(["a.jpg", "b.jpg", "c.jpg"]);
pages[0].lines; // ← a.jpg, always
```

- **Results are positional.** `pages[i]` is always `imagePaths[i]`. Do not match
  on `PageResult.image` — it carries only a basename, so two same-named files in
  different directories are indistinguishable.
- **Every input gets exactly one entry**, including unreadable files, which come
  back as a normal result with `lines: []`. One bad path does not discard the
  rest of the batch.
- An empty array returns `[]` without spawning anything.
- Rejects with an `OcrError` if a path contains a newline or begins with `#`
  (the image-list format cannot represent either), or if the binary returns a
  different number of results than there were inputs.

Batching only removes the *per-image* process and model-load cost. It does not
change recognition — output is identical to calling `recognize()` in a loop.

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

This package pins release **v0.4.0**. `v0.3.0` was the first release whose
archive ships `onnxruntime_providers_shared` — earlier archives could not load
a GPU provider at all on either platform, and the engine fell back to CPU
silently. `v0.4.0` keeps that and adds ppu-style recognition batching plus
`--min-det-box-area`, `--space-recovery` and `--enable-cpu-mem-arena`.

## Binary cache

| Platform | Location |
|---|---|
| Windows | `%LOCALAPPDATA%\arbo-ocr-js\v0.4.0\windows-x64\` |
| Linux | `$XDG_CACHE_HOME/arbo-ocr-js/v0.4.0/linux-x64/` (or `~/.cache/...`) |

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
