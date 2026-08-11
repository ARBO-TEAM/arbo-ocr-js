import assert from "node:assert/strict";
import test from "node:test";

import { Engine, OcrError, detectPlatform, rejectUnlistablePath } from "../dist/index.js";

test("an empty batch resolves to [] without spawning anything", async () => {
  // binPath is bogus on purpose: if this spawned, it would throw instead.
  const engine = new Engine({ binPath: "/definitely/not/here/arboocr_demo" });
  assert.deepEqual(await engine.recognizeBatch([]), []);
});

test("paths the list format cannot represent are rejected up front", () => {
  assert.throws(() => rejectUnlistablePath("", 0), OcrError);
  assert.throws(() => rejectUnlistablePath("a\nb.jpg", 1), OcrError);
  assert.throws(() => rejectUnlistablePath("a\rb.jpg", 1), OcrError);
  // '#' is a comment to arboocr_demo — silently skipping it would shift every
  // later result onto the wrong input file.
  assert.throws(() => rejectUnlistablePath("#weird.jpg", 2), OcrError);
  assert.throws(() => rejectUnlistablePath("  #weird.jpg", 2), OcrError);
});

test("ordinary paths pass validation, including spaces and Windows separators", () => {
  assert.doesNotThrow(() => rejectUnlistablePath("C:\\tmp\\my receipt.jpg", 0));
  assert.doesNotThrow(() => rejectUnlistablePath("/var/tmp/a#b.jpg", 0));
});

test("a bad path is rejected before the binary is even resolved", async () => {
  const engine = new Engine({ binPath: "/definitely/not/here/arboocr_demo" });
  await assert.rejects(() => engine.recognizeBatch(["ok.jpg", "bad\nname.jpg"]), OcrError);
});

// --- e2e: needs the real binary + models ---
//   ARBO_OCR_TEST_IMAGE=/path/to/receipt.jpg npm test
const image = process.env.ARBO_OCR_TEST_IMAGE;
const skip = !image
  ? "set ARBO_OCR_TEST_IMAGE to run"
  : !detectPlatform()
    ? "no release asset for this platform"
    : false;

test("batch returns one result per input, in input order", { skip, timeout: 600_000 }, async () => {
  const engine = new Engine({ modelType: "tiny" });
  await engine.ensureModels();

  // The same image three times: any result count other than 3 means the
  // positional contract is broken.
  const pages = await engine.recognizeBatch([image, image, image]);
  assert.equal(pages.length, 3);
  for (const page of pages) {
    assert.ok(page.lines.length > 0, "expected text");
    assert.equal(typeof page.backend, "string");
  }
  assert.equal(pages[0].lines[0].text, pages[2].lines[0].text);
});

test("a missing file yields an empty entry, not a thrown batch", { skip, timeout: 600_000 }, async () => {
  // arboocr_demo exits 1 when any page came back empty; the batch must treat
  // that as data, not as failure, or one bad file would discard good results.
  const engine = new Engine({ modelType: "tiny" });
  const pages = await engine.recognizeBatch([image, "definitely-not-an-image.jpg"]);

  assert.equal(pages.length, 2);
  assert.ok(pages[0].lines.length > 0, "the real image should still be recognized");
  assert.deepEqual(pages[1].lines, [], "the missing one should be an empty result");
});

test("batch beats a recognize() loop on the same images", { skip, timeout: 600_000 }, async () => {
  const engine = new Engine({ modelType: "tiny" });
  const images = [image, image, image, image];

  const loopStart = performance.now();
  for (const img of images) await engine.recognize(img);
  const loopMs = performance.now() - loopStart;

  const batchStart = performance.now();
  await engine.recognizeBatch(images);
  const batchMs = performance.now() - batchStart;

  // Only asserting "not slower" — the real margin is machine-dependent, and a
  // strict speedup threshold would make this a flaky test rather than a useful
  // one. The number is logged so a regression is visible.
  console.log(`  loop=${loopMs.toFixed(0)}ms batch=${batchMs.toFixed(0)}ms`);
  assert.ok(batchMs < loopMs, `batch (${batchMs.toFixed(0)}ms) should beat loop (${loopMs.toFixed(0)}ms)`);
});
