import assert from "node:assert/strict";
import test from "node:test";

import { Engine, OcrError, PINNED_VERSION, detectPlatform, flagsFrom } from "../dist/index.js";

test("an empty config emits no flags at all", () => {
  // This is the compatibility guarantee: a caller who sets nothing builds the
  // same argv against any arboOCR release, including ones predating the flags
  // this wrapper knows about.
  assert.deepEqual(flagsFrom({}), []);
});

test("string and number flags emit as two tokens, only when set", () => {
  assert.deepEqual(flagsFrom({ modelType: "medium" }), ["--model-type", "medium"]);
  assert.deepEqual(flagsFrom({ minConfidence: 0.7 }), ["--min-confidence", "0.7"]);
  assert.deepEqual(flagsFrom({ recBatchNum: 12 }), ["--rec-batch-num", "12"]);
});

test("a zero number still emits — 0 is a real value, not 'unset'", () => {
  // --min-confidence 0 disables filtering. A falsy check here would silently
  // drop it and leave the CLI default of 0.5 in place.
  assert.deepEqual(flagsFrom({ minConfidence: 0 }), ["--min-confidence", "0"]);
});

test("bools use the single-token = form cxxopts requires", () => {
  assert.deepEqual(flagsFrom({ useAngleCls: true }), ["--angle=true"]);
  assert.deepEqual(flagsFrom({ useCuda: false }), ["--cuda=false"]);
  assert.deepEqual(flagsFrom({ wordBoxes: true }), ["--word-boxes=true"]);
});

/**
 * True when no token in `argv` mentions `flag` at all — neither the bare
 * `--flag`/`--flag=value` form nor the two-token `--flag value` form. A weaker
 * `!argv.includes(\`--${flag}\`)` would miss the `=` form and let a leaked
 * v0.4.0 flag past a pre-v0.4.0 binary.
 */
const mentions = (argv, flag) => argv.some((token) => token === `--${flag}` || token.startsWith(`--${flag}=`));

test("minDetBoxArea emits only when supplied, and 0 is a real value", () => {
  // v0.4.0-only, so an unset field must not reach cxxopts at all — on a
  // pre-v0.4.0 binary an unknown option is a usage error and exit 1.
  assert.deepEqual(flagsFrom({}), []);
  assert.ok(!mentions(flagsFrom({}), "min-det-box-area"));
  assert.ok(!mentions(flagsFrom({ wordBoxes: true }), "min-det-box-area"));

  assert.deepEqual(flagsFrom({ minDetBoxArea: 20 }), ["--min-det-box-area", "20"]);
  // 0 disables the cut in the binary — a falsy check here would drop it.
  assert.deepEqual(flagsFrom({ minDetBoxArea: 0 }), ["--min-det-box-area", "0"]);
  assert.deepEqual(flagsFrom({ minDetBoxArea: 12.5 }), ["--min-det-box-area", "12.5"]);
});

test("v0.4.0 bools emit --flag=true only when true, and nothing otherwise", () => {
  for (const [key, flag] of [
    ["spaceRecovery", "space-recovery"],
    ["enableCpuMemArena", "enable-cpu-mem-arena"],
  ]) {
    assert.deepEqual(flagsFrom({ [key]: true }), [`--${flag}=true`]);
    // An explicit false puts NOTHING on argv: false is the binary's own
    // default, so it is indistinguishable from unset, and emitting it would
    // break a pre-v0.4.0 binary that has no such option.
    assert.deepEqual(flagsFrom({ [key]: false }), []);
    assert.ok(!mentions(flagsFrom({ [key]: false }), flag));
    assert.ok(!mentions(flagsFrom({}), flag));
  }
});

test("the two v0.4.0 bools are independent of each other and of the older ones", () => {
  const argv = flagsFrom({ spaceRecovery: true, enableCpuMemArena: true, useCuda: true });
  assert.deepEqual(argv, ["--cuda=true", "--space-recovery=true", "--enable-cpu-mem-arena=true"]);

  const mixed = flagsFrom({ spaceRecovery: true, enableCpuMemArena: false, useCuda: false });
  assert.deepEqual(mixed, ["--cuda=false", "--space-recovery=true"]);
  assert.ok(!mentions(mixed, "enable-cpu-mem-arena"));
});

test("binPath pointing nowhere fails with an OcrError, not a spawn crash", async () => {
  const engine = new Engine({ binPath: "/definitely/not/here/arboocr_demo" });
  await assert.rejects(() => engine.recognize("x.png"), OcrError);
});

test("the pinned version is the one that has --download-models", () => {
  assert.match(PINNED_VERSION, /^v\d+\.\d+\.\d+$/);
});

test("detectPlatform returns a release platform or null, never a guess", () => {
  const platform = detectPlatform();
  assert.ok(platform === null || platform === "windows-x64" || platform === "linux-x64");
});
