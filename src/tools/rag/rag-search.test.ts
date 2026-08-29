import { afterEach, describe, expect, test } from 'bun:test';
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ragSearchTool } from './rag-search.js';

let workspace = '';
const previousRoot = process.env.AI_TRADER_REPO_ROOT;
const previousKey = process.env.OPENAI_API_KEY;

afterEach(() => {
  if (workspace) rmSync(workspace, { recursive: true, force: true });
  workspace = '';
  if (previousRoot === undefined) delete process.env.AI_TRADER_REPO_ROOT;
  else process.env.AI_TRADER_REPO_ROOT = previousRoot;
  if (previousKey === undefined) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = previousKey;
});

function makeWorkspace(output = '{"build_id":"' + 'a'.repeat(64) + '","hits":[]}\n') {
  workspace = mkdtempSync(join(tmpdir(), 'dexter-rag-tool-'));
  const bin = join(workspace, 'bin');
  mkdirSync(bin);
  const command = join(bin, 'rag-search');
  writeFileSync(command, `#!/bin/sh
printf '%s\\n' "$@" > "$PWD/args.txt"
printf '%s' "$OPENAI_API_KEY" > "$PWD/key.txt"
printf '%s\\n%s' "$PATH" "$PYTHONNOUSERSITE" > "$PWD/env.txt"
printf '%s' '${output.replaceAll("'", "'\\''")}'
`);
  chmodSync(command, 0o755);
  process.env.AI_TRADER_REPO_ROOT = workspace;
  process.env.OPENAI_API_KEY = 'must-not-cross-boundary';
}

describe('rag_search tool boundary', () => {
  test('passes structured arguments without a shell and returns provenance JSON', async () => {
    makeWorkspace();
    const raw = await ragSearchTool.invoke({
      query: 'quoted; $(touch escaped)',
      k: 2,
      symbol: 'NVDA',
      market: 'US',
    });
    const result = JSON.parse(raw).data;
    expect(result.build_id).toBe('a'.repeat(64));
    expect(result.hits).toEqual([]);
    const args = readFileSync(join(workspace, 'args.txt'), 'utf8').trimEnd().split('\n');
    expect(args).toEqual(['--k', '2', '--symbol', 'NVDA', '--market', 'US', '--', 'quoted; $(touch escaped)']);
    expect(readFileSync(join(workspace, 'key.txt'), 'utf8')).toBe('');
    expect(readFileSync(join(workspace, 'env.txt'), 'utf8')).toBe('/usr/local/bin:/usr/bin:/bin\n1');
    expect(() => readFileSync(join(workspace, 'escaped'), 'utf8')).toThrow();
  });

  test('rejects missing workspace anchor instead of choosing a caller path', async () => {
    delete process.env.AI_TRADER_REPO_ROOT;
    const result = JSON.parse(await ragSearchTool.invoke({ query: 'history' })).data;
    expect(result.error).toBe('rag_search_unavailable');
  });

  test('normalizes command and output failures into tool results', async () => {
    makeWorkspace('not-json\n');
    const result = JSON.parse(await ragSearchTool.invoke({ query: 'history' })).data;
    expect(result.error).toBe('rag_search_failed');
  });
});
