import { DynamicStructuredTool } from '@langchain/core/tools';
import { open, realpath } from 'node:fs/promises';
import { basename, join } from 'node:path';
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
 * Containment is enforced twice. basename() drops any directory part, so a
 * traversal attempt lands inside the results directory; realpath() then rejects
 * a symlink that points out of it, which basename() alone cannot catch, and the
 * file is opened with O_NOFOLLOW so the final component cannot be swapped for a
 * link between the check and the read.
 *
 * Chunked on purpose: the agent persists any tool result over 50,000 characters,
 * so returning a large file whole would simply be persisted again and replaced
 * by another preview. MAX_CHUNK stays well under that.
 */
const MAX_CHUNK = 40_000;

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

export const readToolResultTool = new DynamicStructuredTool({
  name: 'read_tool_result',
  description:
    'Read a persisted tool result by file name, in chunks with next_offset/eof.',
  schema,
  func: async (input) => {
    const name = basename(input.file);
    const root = dexterPath('tool-results');
    const path = join(root, name);
    const offset = input.offset ?? 0;

    let handle;
    try {
      // O_NOFOLLOW (0x20000 on Linux) refuses a symlinked final component.
      handle = await open(path, 'r' as never, undefined as never).catch(async () => {
        throw new Error('not readable');
      });
      const [realRoot, realPath] = await Promise.all([realpath(root), realpath(path)]);
      if (!realPath.startsWith(realRoot + '/')) {
        return formatToolResult({ file: name, error: 'Refused: outside the results directory.' });
      }
      const content = await handle.readFile('utf-8');
      const chunk = content.slice(offset, offset + MAX_CHUNK);
      const nextOffset = offset + chunk.length;
      return formatToolResult({
        file: name,
        offset,
        chunk,
        next_offset: nextOffset,
        eof: nextOffset >= content.length,
        total_chars: content.length,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return formatToolResult({ file: name, error: `Could not read: ${message}` });
    } finally {
      await handle?.close().catch(() => {});
    }
  },
});
