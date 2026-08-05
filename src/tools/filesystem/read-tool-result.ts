import { DynamicStructuredTool } from '@langchain/core/tools';
import { constants } from 'node:fs';
import { lstat, open, realpath } from 'node:fs/promises';
import { basename, isAbsolute, join, relative, resolve } from 'node:path';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { dexterPath } from '../../utils/paths.js';

/**
 * Read back a tool result that exceeded the size cap and was persisted to
 * .dexter/tool-results.
 *
 * read_file can do this too, but it can read anything else as well. Headless
 * runs research untrusted web content with no operator watching, so they run
 * without read_file and use this instead: same recovery, no arbitrary reads.
 *
 * Containment: basename() drops any directory part, the file is opened with
 * O_NOFOLLOW so the kernel refuses a symlinked final component instead of the
 * check racing the read, and the resolved path must still sit inside the
 * results directory.
 *
 * Size: the agent persists any tool result whose *serialised* form exceeds
 * MAX_TOOL_RESULT_CHARS, so slicing a fixed number of characters is not enough.
 * A persisted result is often itself JSON, and 40,000 characters of quote- and
 * backslash-heavy text serialise past 50,000 -- the recovery would be persisted
 * again, leaving the caller in a loop. The slice shrinks until the formatted
 * result fits.
 *
 * Persisted results reach several hundred thousand characters in practice
 * (469,362 was the largest on this machine), so a full recovery can exceed the
 * run's iteration budget. The description tells the model to read the part it
 * needs rather than stream the file.
 */
// The agent persists anything over 50,000, so 48,000 keeps a margin while
// spending as little of the iteration budget per read as possible.
const SAFE_FORMATTED_CHARS = 48_000;

export const READ_TOOL_RESULT_DESCRIPTION = `
Read the full content of a previously persisted tool result, in chunks.

## When to Use

- A tool result was replaced by "[Result persisted to ...]" and you need the
  part beyond the preview

## How

- Call with the file name from the notice; read further with the returned
  next_offset until eof is true
- Results can be hundreds of thousands of characters, and each call costs one
  step of your budget: read the part you need rather than the whole file

## When NOT to Use

- For any other file (this tool only reads persisted tool results)
`.trim();

const schema = z.object({
  file: z.string().describe('The persisted result file, as named in the notice.'),
  offset: z.number().int().min(0).optional()
    .describe('Character offset to read from (default 0).'),
});

/**
 * The largest slice at `offset` whose formatted result stays under the cap,
 * found by binary search.
 *
 * Halving instead would waste most of each call on escape-heavy content -- if
 * 40,000 does not fit, the next try is 20,000 even when 34,000 would. That
 * matters because the caller has maxIterations (5 by default) for the whole
 * research turn, and every read spends one.
 */
function fitSlice(
  content: string,
  offset: number,
  build: (chunk: string, next: number) => string,
): { chunk: string; formatted: string } {
  const remaining = content.length - offset;
  if (remaining <= 0) return { chunk: '', formatted: '' };

  // lo starts at 1: with one character left, (0 + 1) / 2 floors to 0 and the
  // search returned nothing, so the final character and eof were unreachable.
  let lo = 1;
  let hi = remaining;
  let best = { chunk: '', formatted: '' };
  while (lo <= hi) {
    const size = Math.floor((lo + hi) / 2);
    const chunk = content.slice(offset, offset + size);
    const formatted = build(chunk, offset + chunk.length);
    if (formatted.length <= SAFE_FORMATTED_CHARS) {
      best = { chunk, formatted };
      lo = size + 1;
    } else {
      hi = size - 1;
    }
  }
  return best;
}

export const readToolResultTool = new DynamicStructuredTool({
  name: 'read_tool_result',
  description:
    'Read a persisted tool result by file name, in chunks with next_offset/eof.',
  schema,
  func: async (input) => {
    const name = basename(input.file);
    const root = resolve(dexterPath('tool-results'));
    const path = join(root, name);
    const offset = input.offset ?? 0;

    // relative(), not a startsWith prefix test: that compares raw strings with a
    // hardcoded POSIX separator and would reject valid paths on Windows.
    const rel = relative(root, resolve(path));
    if (rel === '' || rel.startsWith('..') || isAbsolute(rel)) {
      return formatToolResult({ file: name, error: 'Refused: outside the results directory.' });
    }

    // O_NOFOLLOW only guards the final component, so the directories above it
    // have to be checked separately: with `.dexter/tool-results -> /etc`, a
    // request for "passwd" is lexically fine and passwd is not itself a link,
    // and the read succeeds (reproduced 2026-08-05). Both `.dexter` and the
    // results directory must be real directories, and the resolved root must
    // still sit inside the workspace this process was started in.
    try {
      const anchor = await realpath(process.cwd());
      for (const dir of [resolve(dexterPath()), root]) {
        const info = await lstat(dir);
        if (info.isSymbolicLink() || !info.isDirectory()) {
          return formatToolResult({ file: name, error: 'Refused: results directory is not a real directory.' });
        }
      }
      const realRoot = await realpath(root);
      const fromAnchor = relative(anchor, realRoot);
      if (fromAnchor.startsWith('..') || isAbsolute(fromAnchor)) {
        return formatToolResult({ file: name, error: 'Refused: results directory escapes the workspace.' });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return formatToolResult({ file: name, error: `Could not verify the results directory: ${message}` });
    }

    let handle;
    try {
      handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
      const stat = await handle.stat();
      if (!stat.isFile()) {
        return formatToolResult({ file: name, error: 'Refused: not a regular file.' });
      }
      // A hard link is a regular file and not a symlink, so every check above
      // passes for `ln .env .dexter/tool-results/leak.txt` (reproduced). A
      // result this tool is meant to read was written once and linked once.
      if (stat.nlink !== 1) {
        return formatToolResult({ file: name, error: 'Refused: file has more than one link.' });
      }
      const content = await handle.readFile('utf-8');
      const build = (chunk: string, next: number): string => formatToolResult({
        file: name,
        offset,
        chunk,
        next_offset: next,
        eof: next >= content.length,
        total_chars: content.length,
      });
      const { formatted } = fitSlice(content, offset, build);
      if (!formatted) {
        return formatToolResult({ file: name, error: 'Refused: no readable chunk at that offset.' });
      }
      return formatted;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return formatToolResult({ file: name, error: `Could not read: ${message}` });
    } finally {
      await handle?.close().catch(() => {});
    }
  },
});
