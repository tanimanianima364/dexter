import { execFile } from 'node:child_process';
import { lstat, realpath } from 'node:fs/promises';
import { isAbsolute, join, relative } from 'node:path';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';

const COMMAND_TIMEOUT_MS = 30_000;
const MAX_OUTPUT_BYTES = 512 * 1024;
const SHA256_RE = /^[0-9a-f]{64}$/;

export const RAG_SEARCH_DESCRIPTION = `
Search the ai-trader workspace's published, immutable RAG index.

This is a read-only evidence lookup. It never changes reports, signals, orders,
or the index. Use it for prior internal reports and settled outcomes when they
would help answer the question. The workspace and published index are fixed by
the launcher; there is no path or build override. Results include build and
record provenance so claims can be checked. Right-censored outcomes are hidden
unless explicitly requested with include_censored=true.
`.trim();

const schema = z.object({
  query: z.string().trim().min(1).max(2_000)
    .describe('Natural-language search query.'),
  k: z.number().int().min(1).max(20).optional()
    .describe('Maximum number of evidence hits (default 8, maximum 20).'),
  as_of: z.string().trim().min(1).max(64).optional()
    .describe('Optional RFC 3339 decision cutoff for historical replay.'),
  symbol: z.string().trim().min(1).max(32).optional()
    .describe('Optional canonical symbol filter, such as NVDA or 7203.'),
  market: z.enum(['US', 'JP']).optional()
    .describe('Optional market filter.'),
  record_type: z.enum([
    'decision_run', 'decision_evidence', 'outcome', 'trade',
    'order_intent', 'report',
  ]).optional().describe('Optional registered record-type filter.'),
  outcome_kind: z.enum(['simulated_counterfactual', 'paper_trade', 'live_trade'])
    .optional().describe('Optional outcome-kind filter.'),
  include_censored: z.boolean().optional()
    .describe('Show right-censored outcomes only when explicitly true.'),
});

type CommandError = Error & {
  code?: string | number;
  signal?: string;
  stdout?: string;
  stderr?: string;
};

type SearchResult = {
  build_id: string;
  hits: unknown[];
  [key: string]: unknown;
};

function runSearch(
  command: string,
  args: string[],
  cwd: string,
): Promise<{ stdout: string; stderr: string }> {
  // execFile passes an argv array with shell=false. The query and filters are
  // therefore data, never shell syntax, even when they contain punctuation.
  return new Promise((resolve, reject) => {
    execFile(command, args, {
      cwd,
      env: cleanEnvironment(),
      shell: false,
      timeout: COMMAND_TIMEOUT_MS,
      maxBuffer: MAX_OUTPUT_BYTES,
      windowsHide: true,
    }, (error, stdout, stderr) => {
      if (error) {
        const commandError = error as CommandError;
        commandError.stdout = stdout;
        commandError.stderr = stderr;
        reject(commandError);
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function cleanEnvironment(): NodeJS.ProcessEnv {
  // The search command is local and read-only. Do not pass API keys, PYTHONPATH,
  // virtualenv overrides, or arbitrary startup settings into its interpreter.
  const env: NodeJS.ProcessEnv = {
    // The shell wrapper uses this only for its system-python FTS fallback.
    // Never let a caller-controlled PATH select that interpreter.
    PATH: '/usr/local/bin:/usr/bin:/bin',
    PYTHONNOUSERSITE: '1',
    PYTHONSAFEPATH: '1',
    PYTHONUTF8: '1',
  };
  for (const key of ['HOME', 'LANG', 'LC_ALL', 'TZ']) {
    const value = process.env[key];
    if (value !== undefined) env[key] = value;
  }
  return env;
}

async function workspaceCommand(): Promise<{ root: string; command: string } | null> {
  const configured = process.env.AI_TRADER_REPO_ROOT;
  if (!configured || !isAbsolute(configured)) return null;
  try {
    const root = await realpath(configured);
    const rootInfo = await lstat(root);
    if (!rootInfo.isDirectory()) return null;

    const command = join(root, 'bin', 'rag-search');
    const rel = relative(root, command);
    if (rel.startsWith('..') || isAbsolute(rel)) return null;
    const commandInfo = await lstat(command);
    if (commandInfo.isSymbolicLink() || !commandInfo.isFile()) return null;
    return { root, command };
  } catch {
    return null;
  }
}

function commandArgs(input: z.infer<typeof schema>): string[] {
  const args: string[] = [];
  if (input.k !== undefined) args.push('--k', String(input.k));
  if (input.as_of !== undefined) args.push('--as-of', input.as_of);
  if (input.symbol !== undefined) args.push('--symbol', input.symbol);
  if (input.market !== undefined) args.push('--market', input.market);
  if (input.record_type !== undefined) args.push('--record-type', input.record_type);
  if (input.outcome_kind !== undefined) args.push('--outcome-kind', input.outcome_kind);
  if (input.include_censored === true) args.push('--include-censored');
  // `--` must come after the options: argparse treats everything after it as
  // positional data, and keeps a query beginning with '-' unambiguous.
  args.push('--', input.query);
  return args;
}

function parseResult(stdout: string): SearchResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.trim());
  } catch {
    throw new Error('search command returned invalid JSON');
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('search command returned a non-object result');
  }
  const result = parsed as Record<string, unknown>;
  if (typeof result.build_id !== 'string' || !SHA256_RE.test(result.build_id)) {
    throw new Error('search result has no verified build_id');
  }
  if (!Array.isArray(result.hits)) {
    throw new Error('search result has no hits array');
  }
  return result as SearchResult;
}

export const ragSearchTool = new DynamicStructuredTool({
  name: 'rag_search',
  description: RAG_SEARCH_DESCRIPTION,
  schema,
  func: async (input) => {
    const target = await workspaceCommand();
    if (!target) {
      return formatToolResult({
        error: 'rag_search_unavailable',
        message: 'The registered ai-trader workspace root is unavailable.',
      });
    }

    try {
      const { stdout, stderr } = await runSearch(
        target.command, commandArgs(input), target.root);
      const result = parseResult(stdout);
      return formatToolResult({ ...result, stderr: stderr.trim() || undefined });
    } catch (error) {
      const commandError = error as CommandError;
      const detail = commandError.code === 'ETIMEDOUT'
        ? 'search timed out'
        : 'search command failed';
      return formatToolResult({
        error: 'rag_search_failed',
        message: detail,
        stderr: (commandError.stderr || '').trim().slice(-1_000) || undefined,
      });
    }
  },
});
