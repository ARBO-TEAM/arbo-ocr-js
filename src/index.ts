/**
 * Node.js and Bun wrapper for arboOCR — runs the prebuilt `arboocr_demo`
 * binary via `child_process`, no C++ build required.
 *
 * @packageDocumentation
 */

import { execFile } from "node:child_process";
import { stat } from "node:fs/promises";
import { promisify } from "node:util";

import { ensureInstalled } from "./installer.js";

export { PINNED_VERSION, detectPlatform, ensureInstalled } from "./installer.js";
export type { Platform } from "./installer.js";

const execFileAsync = promisify(execFile);

/** One vertex of a {@link LineResult} polygon. */
export interface Point {
  x: number;
  y: number;
}

/**
 * One word inside a {@link LineResult}, present only when `wordBoxes` is set.
 * Scripts that delimit words with spaces group into words; CJK characters
 * stand alone.
 */
export interface WordResult {
  text: string;
  score: number;
  polygon: Point[];
}

/** One recognized text line. Polygon points run clockwise from top-left-ish. */
export interface LineResult {
  text: string;
  score: number;
  detScore: number;
  polygon: Point[];
  /** Undefined unless `wordBoxes` was set — arboOCR omits the key entirely. */
  words?: WordResult[];
}

/**
 * A full-page result. An empty `lines` array is a normal, successful result
 * (no text found), not an error.
 */
export interface PageResult {
  backend: string;
  image: string;
  elapsedMs: number;
  lines: LineResult[];
}

/** Thrown when the binary cannot be started, exits non-zero, or emits non-JSON. */
export class OcrError extends Error {
  readonly exitCode?: number;
  readonly stderr?: string;

  constructor(message: string, opts: { exitCode?: number; stderr?: string } = {}) {
    super(message);
    this.name = "OcrError";
    this.exitCode = opts.exitCode;
    this.stderr = opts.stderr;
  }
}

/**
 * Engine configuration. Every field is optional: an unset field emits no flag
 * at all, leaving `arboocr_demo`'s own default in place. That rule is what
 * keeps a `binPath` pointed at an older binary working — cxxopts exits 1 on an
 * unknown option, so a flag we never emit can never break it.
 */
export interface Config {
  /** Explicit path to arboocr_demo; unset means download to the cache on first use. */
  binPath?: string;

  modelsDir?: string;
  ocrVersion?: string;
  /** Recognizer size: `tiny` | `small` (default) | `medium`. */
  modelType?: string;
  detModelPath?: string;
  clsModelPath?: string;
  recModelPath?: string;
  dictPath?: string;
  /** Log engine events to stderr: `debug` | `info` | `warn` | `error`. */
  logLevel?: string;
  /** Directory URL to fetch missing models from, instead of the default. */
  modelsUrl?: string;

  /** Drop lines below this recognition confidence. CLI default 0.5. */
  minConfidence?: number;
  /** Crops per recognition inference call. CLI default 6. */
  recBatchNum?: number;
  /** Longest image side for detection resize. CLI default 960. */
  detLimitSideLen?: number;

  useAngleCls?: boolean;
  useCuda?: boolean;
  useTensorrt?: boolean;
  useFp16?: boolean;
  useClahe?: boolean;
  /** Emit a polygon per word (per character for CJK) as {@link LineResult.words}. */
  wordBoxes?: boolean;
  /** Fail instead of fetching a missing model — the flag form of `ARBOOCR_OFFLINE=1`. */
  noDownload?: boolean;
}

const STRING_FLAGS = {
  modelsDir: "models-dir",
  ocrVersion: "ocr-version",
  modelType: "model-type",
  detModelPath: "det-model",
  clsModelPath: "cls-model",
  recModelPath: "rec-model",
  dictPath: "dict",
  logLevel: "log-level",
  modelsUrl: "models-url",
} as const;

const NUMBER_FLAGS = {
  minConfidence: "min-confidence",
  recBatchNum: "rec-batch-num",
  detLimitSideLen: "det-limit-side-len",
} as const;

const BOOL_FLAGS = {
  useAngleCls: "angle",
  useCuda: "cuda",
  useTensorrt: "tensorrt",
  useFp16: "fp16",
  useClahe: "clahe",
  wordBoxes: "word-boxes",
  noDownload: "no-download",
} as const;

/**
 * arboocr_demo can write ~200 KB of ONNXRuntime schema-registration warnings
 * to stderr, and `--word-boxes` JSON grows with the page. Node's 1 MB default
 * would turn either into an ENOBUFS kill rather than a result.
 */
const MAX_BUFFER = 64 * 1024 * 1024;

/**
 * Runs the prebuilt arboocr_demo binary and parses its `--json` output.
 *
 * The binary is resolved lazily on the first call, not in the constructor, so
 * construction stays synchronous and a program that builds an Engine but never
 * uses it never touches the network.
 *
 * ```ts
 * const engine = new Engine({ modelType: "small" });
 * const page = await engine.recognize("receipt.jpg");
 * for (const line of page.lines) console.log(line.text, line.score);
 * ```
 */
export class Engine {
  readonly config: Config;
  #binPath?: Promise<string>;

  constructor(config: Config = {}) {
    this.config = config;
  }

  /** Resolves the binary path, memoized so concurrent calls share one install. */
  async binaryPath(): Promise<string> {
    this.#binPath ??= this.#resolveBinary();
    return this.#binPath;
  }

  async #resolveBinary(): Promise<string> {
    const explicit = this.config.binPath;
    if (!explicit) return ensureInstalled();

    try {
      await stat(explicit);
    } catch {
      throw new OcrError(
        `arboocr_demo binary not found at ${explicit}. Pass a valid binPath or ` +
          `leave it unset to auto-install.`,
      );
    }
    return explicit;
  }

  /**
   * Runs `arboocr_demo --image <imagePath> --json` plus the flags derived from
   * {@link Config}, and parses the JSON on stdout.
   */
  async recognize(imagePath: string): Promise<PageResult> {
    const stdout = await this.#run(["--image", imagePath, "--json"]);
    try {
      return JSON.parse(stdout) as PageResult;
    } catch {
      throw new OcrError(
        `arboocr_demo --json produced unparseable output: ${stdout.slice(0, 500)}`,
      );
    }
  }

  /**
   * Runs `arboocr_demo --download-models` to fetch the weights for this
   * engine's `ocrVersion`/`modelType` into arboOCR's model cache, then returns
   * — the binary downloads and exits without doing any OCR.
   *
   * Idempotent: an already-cached model is a no-op, and the binary's own
   * precedence rules still apply (an explicit `recModelPath` is never
   * substituted by a download, and a file already in `modelsDir` wins without
   * touching the network). Call it from a Docker build step so the first
   * `recognize` does not pay for the download.
   */
  async ensureModels(): Promise<void> {
    await this.#run(["--download-models"]);
  }

  async #run(baseArgs: string[]): Promise<string> {
    const bin = await this.binaryPath();
    const args = [...baseArgs, ...flagsFrom(this.config)];

    try {
      // execFile drains stdout and stderr concurrently. Reading one to
      // completion before the other would deadlock: the child blocks writing a
      // full stderr pipe while the parent waits on stdout that never comes.
      const { stdout } = await execFileAsync(bin, args, { maxBuffer: MAX_BUFFER });
      return stdout.trim();
    } catch (err) {
      const e = err as NodeJS.ErrnoException & { code?: number | string; stderr?: string };
      if (typeof e.code === "number") {
        throw new OcrError(`arboocr_demo exited with code ${e.code}`, {
          exitCode: e.code,
          stderr: e.stderr,
        });
      }
      throw new OcrError(`could not start process: ${e.message}`, { stderr: e.stderr });
    }
  }
}

/**
 * Builds the argv tail from a config. Unset fields emit nothing; booleans use
 * the single-token `--flag=value` form because cxxopts only binds a bool that
 * way — `["--angle", "false"]` as two tokens leaves `--angle` implicitly true
 * and `false` as an ignored stray positional.
 */
export function flagsFrom(config: Config): string[] {
  const flags: string[] = [];

  for (const [key, flag] of Object.entries(STRING_FLAGS)) {
    const value = config[key as keyof typeof STRING_FLAGS];
    if (value !== undefined && value !== "") flags.push(`--${flag}`, value);
  }
  for (const [key, flag] of Object.entries(NUMBER_FLAGS)) {
    const value = config[key as keyof typeof NUMBER_FLAGS];
    if (value !== undefined) flags.push(`--${flag}`, String(value));
  }
  for (const [key, flag] of Object.entries(BOOL_FLAGS)) {
    const value = config[key as keyof typeof BOOL_FLAGS];
    if (value !== undefined) flags.push(`--${flag}=${value}`);
  }

  return flags;
}
