import { DynamicStructuredTool } from '@langchain/core/tools';
import { readFile } from 'node:fs/promises';
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
 * The name is taken as a bare filename, so no path can escape the directory.
 */
export const READ_TOOL_RESULT_DESCRIPTION = `
Read the full content of a previously persisted tool result.

## When to Use

- A tool result was replaced by "[Result persisted to ...]" and you need the
  part beyond the preview

## When NOT to Use

- For any other file (this tool only reads persisted tool results)
`.trim();

const schema = z.object({
  file: z.string().describe('The persisted result file, as named in the notice.'),
});

export const readToolResultTool = new DynamicStructuredTool({
  name: 'read_tool_result',
  description: 'Read the full content of a persisted tool result by file name.',
  schema,
  func: async (input) => {
    // basename() strips any directory part, so "../../.env" reads ".env" inside
    // the results directory and simply does not exist.
    const name = basename(input.file);
    const path = join(dexterPath('tool-results'), name);
    try {
      const content = await readFile(path, 'utf-8');
      return formatToolResult({ file: name, content });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return formatToolResult({ file: name, error: `Could not read: ${message}` });
    }
  },
});
