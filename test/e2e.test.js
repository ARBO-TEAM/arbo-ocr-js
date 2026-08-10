// End-to-end: downloads the pinned release, downloads the models, OCRs a real
// image. Opt-in because it pulls ~13 MB of binary plus model weights.
//
//   ARBO_OCR_TEST_IMAGE=/path/to/receipt.jpg npm test

import assert from "node:assert/strict";
import test from "node:test";

import { Engine, detectPlatform } from "../dist/index.js";

const image = process.env.ARBO_OCR_TEST_IMAGE;
const skip = !image
  ? "set ARBO_OCR_TEST_IMAGE to run"
  : !detectPlatform()
    ? "no release asset for this platform"
    : false;

test("recognizes text from a real image", { skip, timeout: 600_000 }, async () => {
  const engine = new Engine({ modelType: "tiny" });

  await engine.ensureModels();
  const page = await engine.recognize(image);

  assert.equal(typeof page.backend, "string");
  assert.ok(page.elapsedMs > 0, "elapsedMs should be positive");
  assert.ok(page.lines.length > 0, "expected at least one line of text");

  for (const line of page.lines) {
    assert.equal(typeof line.text, "string");
    assert.ok(line.score >= 0 && line.score <= 1, `score out of range: ${line.score}`);
    assert.equal(line.polygon.length, 4);
  }
});

test("wordBoxes adds per-word polygons", { skip, timeout: 600_000 }, async () => {
  const engine = new Engine({ modelType: "tiny", wordBoxes: true });
  const page = await engine.recognize(image);

  const withWords = page.lines.filter((line) => line.words?.length);
  assert.ok(withWords.length > 0, "expected words on at least one line");
});
