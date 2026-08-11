'use strict';

// Fetching a .dem.bz2 from Valve's replay CDN and decompressing it to disk.
//
// Download and decompression are deliberately separate steps. Doing both in one
// pipeline is faster but makes failures unattributable: if the bz2 decoder
// throws, the pipeline destroys the HTTP stream and undici reports
// "terminated — other side closed", which reads exactly like a network fault.
// Two phases means an error names the thing that actually went wrong.
//
// Split out from fetch.js so it can be tested on its own — fetch.js runs its
// main() on import, which makes it useless as a module.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { Readable, Transform } = require('stream');
const { pipeline } = require('stream/promises');

const bz2 = require('unbzip2-stream');

// The CDN drops long connections fairly readily: a run of large sequential
// downloads tends to end in undici's "terminated". Retrying with a growing
// pause clears it without needing a second run of the whole job.
const DOWNLOAD_ATTEMPTS = 3;
const RETRY_BACKOFF_MS = 3_000;

// Abort if no bytes arrive for this long. A total timeout is wrong here —
// a 250 MB demo on a slow line is fine, a stalled socket is not.
const STALL_TIMEOUT_MS = 60_000;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Network errors from undici hide the useful detail on .cause. */
function describeError(err) {
  const cause = err.cause && err.cause.message ? ` — ${err.cause.message}` : '';
  return `${err.message}${cause}`;
}

function tagError(err, phase, message) {
  const tagged = new Error(message);
  tagged.phase = phase;
  tagged.cause = err;
  if (err.status) tagged.status = err.status;
  return tagged;
}

/**
 * Passthrough that counts bytes and fires `onStall` if the flow goes quiet.
 */
function stallGuard(stallTimeoutMs, onStall, onProgress) {
  let bytes = 0;
  let timer = stallTimeoutMs ? setTimeout(onStall, stallTimeoutMs) : null;

  const reset = () => {
    if (!stallTimeoutMs) return;
    clearTimeout(timer);
    timer = setTimeout(onStall, stallTimeoutMs);
  };

  const stream = new Transform({
    transform(chunk, _enc, cb) {
      bytes += chunk.length;
      reset();
      if (onProgress) onProgress(bytes);
      cb(null, chunk);
    },
    flush(cb) {
      clearTimeout(timer);
      cb();
    },
  });

  stream.on('close', () => clearTimeout(timer));
  return stream;
}

/**
 * Phase 1 — fetch the compressed demo to disk.
 * @returns {Promise<number>} compressed bytes written
 */
async function fetchCompressed(url, bz2Path, { signal, stallTimeoutMs, onProgress } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal) {
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
  }

  let stalled = false;
  const timeout = stallTimeoutMs === undefined ? STALL_TIMEOUT_MS : stallTimeoutMs;

  let response;
  try {
    response = await fetch(url, { signal: controller.signal });
  } catch (err) {
    throw tagError(err, 'network', `could not connect: ${describeError(err)}`);
  }

  if (!response.ok) {
    const err = new Error(`HTTP ${response.status}`);
    err.status = response.status;
    err.phase = 'network';
    throw err;
  }

  const guard = stallGuard(
    timeout,
    () => {
      stalled = true;
      controller.abort();
    },
    onProgress
  );

  try {
    await pipeline(Readable.fromWeb(response.body), guard, fs.createWriteStream(bz2Path));
  } catch (err) {
    fs.rmSync(bz2Path, { force: true });
    if (stalled) {
      throw tagError(err, 'network', `stalled: no data for ${Math.round(timeout / 1000)}s`);
    }
    if (signal && signal.aborted) throw tagError(err, 'cancelled', 'cancelled');
    throw tagError(err, 'network', `download interrupted: ${describeError(err)}`);
  } finally {
    if (signal) signal.removeEventListener('abort', abort);
  }

  return fs.statSync(bz2Path).size;
}

/**
 * Phase 2 — decompress to the final .dem.
 *
 * On failure the .bz2 is kept: a decode error is worth inspecting, and
 * re-downloading 250 MB to look at it would be silly.
 *
 * @returns {Promise<number>} decompressed bytes written
 */
async function decompress(bz2Path, destPath) {
  // Write to .part so an interrupted decode is never mistaken for a finished
  // demo by the Sync Folder scan.
  const partPath = `${destPath}.part`;
  try {
    await pipeline(fs.createReadStream(bz2Path), bz2(), fs.createWriteStream(partPath));
  } catch (err) {
    fs.rmSync(partPath, { force: true });
    throw tagError(
      err,
      'decompress',
      `could not decompress: ${err.message} (kept ${bz2Path} to inspect)`
    );
  }

  fs.renameSync(partPath, destPath);
  fs.rmSync(bz2Path, { force: true });
  return fs.statSync(destPath).size;
}

/**
 * One attempt: fetch, then decompress.
 *
 * The compressed file lands in a scratch directory rather than next to the
 * demo. Under Docker the demo folder is usually a bind mount to the host, and
 * those are slow: writing 250 MB through one while the HTTP connection is open
 * invites the CDN to give up on a slow consumer and close it. Downloading the
 * smaller .bz2 to local disk keeps the connection short-lived, and the big
 * decompressed write then happens with no socket held open at all.
 */
async function downloadDemo(url, destPath, options = {}) {
  const scratchDir = options.scratchDir || os.tmpdir();
  const bz2Path = path.join(scratchDir, `${path.basename(destPath)}.bz2`);

  fs.mkdirSync(scratchDir, { recursive: true });
  await fetchCompressed(url, bz2Path, options);
  return decompress(bz2Path, destPath);
}

/**
 * Download with retries for transient faults.
 *
 * Only network faults are retried. HTTP status errors are terminal (a 404 or an
 * aged-out 502 means the demo is gone), and so are decode failures — the bytes
 * on disk will not decompress any better the second time.
 *
 * @param {object}      options
 * @param {number}      options.attempts   total tries, default 3
 * @param {number}      options.backoffMs  base pause, multiplied by attempt number
 * @param {number}      options.stallTimeoutMs  abort after this long with no data
 * @param {AbortSignal} options.signal     cancel an in-flight download
 * @param {function}    options.onRetry    called with (attempt, maxRetries, error)
 * @param {function}    options.onProgress called with bytes received so far
 */
async function downloadWithRetry(url, destPath, options = {}) {
  const attempts = options.attempts || DOWNLOAD_ATTEMPTS;
  const backoffMs = options.backoffMs === undefined ? RETRY_BACKOFF_MS : options.backoffMs;
  const onRetry = options.onRetry || (() => {});

  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await downloadDemo(url, destPath, options);
    } catch (err) {
      if (err.status || err.phase === 'decompress' || err.phase === 'cancelled') throw err;

      lastError = err;
      if (attempt < attempts) {
        onRetry(attempt, attempts - 1, err);
        await sleep(backoffMs * attempt);
      }
    }
  }

  throw lastError;
}

module.exports = {
  downloadDemo,
  downloadWithRetry,
  fetchCompressed,
  decompress,
  describeError,
  DOWNLOAD_ATTEMPTS,
  STALL_TIMEOUT_MS,
};
