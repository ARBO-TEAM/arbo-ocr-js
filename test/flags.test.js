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
