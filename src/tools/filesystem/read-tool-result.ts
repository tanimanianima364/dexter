import { DynamicStructuredTool } from '@langchain/core/tools';
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
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
 */
const SAFE_FORMATTED_CHARS = 45_000;
const MAX_SLICE_CHARS = 40_000;

export const READ_TOOL_RESULT_DESCRIPTION = `
Read the full content of a previously persisted tool result, in chunks.

## When to Use

- A tool result was replaced by "[Result persisted to ...]" and you need the
  part beyond the preview

## How

- Call with the file name from the notice; read further with the returned
  next_offset until eof is true

## When NOT to Use

- For any other file (this tool only reads persisted tool results)
`.trim();

const schema = z.object({
  file: z.string().describe('The persisted result file, as named in the notice.'),
  offset: z.number().int().min(0).optional()
    .describe('Character offset to read from (default 0).'),
});

/** The largest slice at `offset` whose formatted result stays under the cap. */
function fitSlice(
  content: string,
  offset: number,
  build: (chunk: string, next: number) => string,
): { chunk: string; formatted: string } {
  let size = Math.min(MAX_SLICE_CHARS, content.length - offset);
  while (size > 0) {
    const chunk = content.slice(offset, offset + size);
    const formatted = build(chunk, offset + chunk.length);
    if (formatted.length <= SAFE_FORMATTED_CHARS) {
      return { chunk, formatted };
    }
    // Escaping expands a character by a bounded factor, so halving converges.
    size = Math.floor(size / 2);
  }
  return { chunk: '', formatted: '' };
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

    let handle;
    try {
      handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
      const stat = await handle.stat();
      if (!stat.isFile()) {
        return formatToolResult({ file: name, error: 'Refused: not a regular file.' });
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
