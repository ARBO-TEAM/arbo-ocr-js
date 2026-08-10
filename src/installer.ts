// Downloads and caches the prebuilt arboocr_demo binary from ArboOCR's GitHub
// Releases. Kept separate from index.ts so "how to get the binary" stays
// independently readable from "how to run it", matching the Go wrapper's
// installer/ package split.

import { execFile } from "node:child_process";
import { chmod, copyFile, mkdir, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const REPO = "wafik/ArboOCR";

/**
 * The release tag this package downloads. The release binary is
 * language-agnostic, so every arboOCR wrapper (Python, Go, Rust, PHP, JS)
 * tracks the same tag — keep them in step when bumping.
 *
 * Bumping this also changes the cache directory (see {@link ensureInstalled}) —
 * that is deliberate, not incidental.
 *
 * v0.3.0 is the release that added model auto-download, so the binary this
 * downloads understands `--no-download` / `--models-url` / `--download-models`
 * and the `ARBOOCR_OFFLINE` / `ARBOOCR_CACHE_DIR` / `ARBOOCR_MODELS_URL`
 * environment variables. It is also the first release to ship
 * `onnxruntime_providers_shared`, without which `useCuda` / `useTensorrt` could
 * not load a GPU execution provider from a release archive at all.
 */
export const PINNED_VERSION = "v0.3.0";

/** How long {@link ensureInstalled} waits for the release asset. */
const DOWNLOAD_TIMEOUT_MS = 120_000;

export type Platform = "windows-x64" | "linux-x64";

/** Returns the release platform for this process, or null if unsupported. */
export function detectPlatform(): Platform | null {
  if (process.arch !== "x64") return null;
  if (process.platform === "win32") return "windows-x64";
  if (process.platform === "linux") return "linux-x64";
  return null;
}

function binaryName(platform: Platform): string {
  return platform === "windows-x64" ? "arboocr_demo.exe" : "arboocr_demo";
}

function assetName(platform: Platform): string {
  return platform === "windows-x64"
    ? "arboocr-windows-x64.zip"
    : "arboocr-linux-x64.tar.gz";
}

function defaultCacheRoot(): string {
  if (process.platform === "win32") {
    return process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local");
  }
  return process.env.XDG_CACHE_HOME || join(homedir(), ".cache");
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Makes sure the arboocr_demo binary exists locally, downloading it from GitHub
 * Releases if missing, and returns its absolute path.
 *
 * `binDir` defaults to a version-scoped directory under the platform cache.
 * The version segment is load-bearing, not decoration: the extracted binary is
 * named `arboocr_demo[.exe]` in every release, so a version-less path is the
 * same path for every version and the "already installed" check below would
 * report a v0.1.0 binary as current forever — bumping PINNED_VERSION would be a
 * silent no-op for everyone who ever ran an older pin. Callers passing an
 * explicit `binDir` own its lifecycle and should version it themselves for the
 * same reason.
 */
export async function ensureInstalled(binDir?: string): Promise<string> {
  const platform = detectPlatform();
  if (!platform) {
    throw new Error(
      `arboocr: unsupported platform ${process.platform}/${process.arch}; ` +
        `download a release manually from https://github.com/${REPO}/releases ` +
        `and pass binPath explicitly`,
    );
  }

  const dir =
    binDir || join(defaultCacheRoot(), "arbo-ocr-js", PINNED_VERSION, platform);
  const binPath = join(dir, binaryName(platform));

  // Safe precisely because dir is version-scoped: a hit means a binary from
  // *this* PINNED_VERSION, not merely some arboocr_demo. And because install
  // publishes by rename (below), a directory that exists is a complete one —
  // an interrupted download can't leave the .exe present but its DLLs missing.
  if (await exists(binPath)) return binPath;

  const asset = assetName(platform);
  const url = `https://github.com/${REPO}/releases/download/${PINNED_VERSION}/${asset}`;

  await mkdir(dir, { recursive: true });
  const staging = join(tmpdir(), `arbo-ocr-js-${process.pid}-${platform}`);
  await rm(staging, { recursive: true, force: true });
  await mkdir(staging, { recursive: true });

  try {
    const archive = join(staging, asset);
    await downloadTo(url, archive);
    await extract(archive, staging);
    await rm(archive, { force: true });
    await flattenSingleSubdir(staging);
    await publish(staging, dir, binaryName(platform));
  } finally {
    await rm(staging, { recursive: true, force: true });
  }

  if (platform === "linux-x64") await chmod(binPath, 0o755);
  return binPath;
}

async function downloadTo(url: string, destPath: string): Promise<void> {
  // ponytail: buffers the whole ~13 MB asset in memory rather than streaming
  // it. One allocation, once per version, and it behaves identically on Node
  // and Bun — Readable.fromWeb + pipeline would be three more imports to save
  // memory nobody is short of. Stream it if the asset ever grows past ~100 MB.
  const res = await fetch(url, { signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
  if (!res.ok) {
    throw new Error(`arboocr: download failed: ${url}: HTTP ${res.status}`);
  }
  await writeFile(destPath, Buffer.from(await res.arrayBuffer()));
}

async function extract(archive: string, targetDir: string): Promise<void> {
  // tar reads both release formats: GNU tar handles the Linux .tar.gz, and
  // Windows ships bsdtar (libarchive) in System32, which reads the .zip too.
  // That is two fewer dependencies than an unzip package plus a tar package.
  //
  // The absolute path on Windows is not paranoia: Git for Windows puts GNU tar
  // on PATH, and GNU tar cannot read a zip. Resolving `tar` by PATH would work
  // or not depending on which shell installed what, which is the worst kind of
  // bug to receive a report about.
  const tarBin =
    process.platform === "win32"
      ? join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe")
      : "tar";
  try {
    await execFileAsync(tarBin, ["-xf", archive, "-C", targetDir]);
  } catch (err) {
    throw new Error(
      `arboocr: could not extract ${archive} with ${tarBin}: ${(err as Error).message}`,
    );
  }
}

/**
 * The Linux tarball wraps everything in one top-level folder
 * (`arboocr-linux-x64/`); the Windows zip does not. Flattening here means the
 * caller gets `<dir>/arboocr_demo` on both.
 */
async function flattenSingleSubdir(dir: string): Promise<void> {
  const entries = await readdir(dir, { withFileTypes: true });
  if (entries.length !== 1 || !entries[0]!.isDirectory()) return;

  const subdir = join(dir, entries[0]!.name);
  for (const name of await readdir(subdir)) {
    await rename(join(subdir, name), join(dir, name));
  }
  await rm(subdir, { recursive: true, force: true });
}

/**
 * Moves the staged files into their final home, `exe` last. That ordering is
 * what makes the `exists(binPath)` short-circuit above trustworthy: the binary
 * only appears once every DLL beside it is already there, so an interrupted
 * install is retried rather than mistaken for a complete one that then fails
 * to load.
 *
 * Per-file rather than renaming the staging directory itself, because staging
 * lives in the OS temp dir — often a different filesystem, where a directory
 * rename fails with EXDEV.
 */
async function publish(staging: string, dir: string, exe: string): Promise<void> {
  const names = await readdir(staging);
  for (const name of [...names.filter((n) => n !== exe), ...names.filter((n) => n === exe)]) {
    const dest = join(dir, name);
    await rm(dest, { recursive: true, force: true });
    try {
      await rename(join(staging, name), dest);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "EXDEV") throw err;
      await copyFile(join(staging, name), dest);
    }
  }
}
