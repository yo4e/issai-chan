const { readLines } = require("./io");

const DEFAULT_CONFIG = {
  MAX_OUT_WORDS: 2,
  TEMP: 1.0
};

function parseLine(line) {
  const match = String(line).match(/^([UA]):\s?(.*)$/);
  const role = match ? match[1] : "U";
  const text = match ? match[2] : String(line);
  return { role, text };
}

function isHiragana(ch) {
  const code = ch.charCodeAt(0);
  return code >= 0x3040 && code <= 0x309f;
}

function isKatakana(ch) {
  const code = ch.charCodeAt(0);
  return code >= 0x30a0 && code <= 0x30ff;
}

function isKanji(ch) {
  const code = ch.charCodeAt(0);
  return code >= 0x4e00 && code <= 0x9fff;
}

function isAsciiAlnum(ch) {
  const code = ch.charCodeAt(0);
  return (code >= 48 && code <= 57) || (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
}

function charType(ch) {
  if (isHiragana(ch)) return "hiragana";
  if (isKatakana(ch)) return "katakana";
  if (isKanji(ch)) return "kanji";
  if (isAsciiAlnum(ch)) return "alnum";
  return "other";
}

function tokenize(text) {
  const tokens = [];
  let buffer = "";
  let lastType = null;

  for (const ch of text) {
    if (ch === " " || ch === "\t") {
      if (buffer) tokens.push(buffer);
      buffer = "";
      lastType = null;
      continue;
    }

    const type = charType(ch);
    if (type === "other") {
      if (buffer) tokens.push(buffer);
      buffer = "";
      lastType = null;
      if (ch !== "\n" && ch !== "\r") tokens.push(ch);
      continue;
    }

    if (type === lastType || !lastType) {
      buffer += ch;
    } else {
      if (buffer) tokens.push(buffer);
      buffer = ch;
    }
    lastType = type;
  }

  if (buffer) tokens.push(buffer);
  return tokens;
}

function buildEmptyModel(existingModel) {
  if (existingModel && existingModel.counts) {
    return {
      counts: existingModel.counts,
      totals: existingModel.totals || {},
      config: { ...DEFAULT_CONFIG, ...(existingModel.config || {}) },
      stats: existingModel.stats || { trainedTokens: 0 }
    };
  }
  return {
    counts: {},
    totals: {},
    config: { ...DEFAULT_CONFIG },
    stats: { trainedTokens: 0 }
  };
}

function ensureModel(existingModel) {
  return buildEmptyModel(existingModel);
}

function incCount(model, prev, next) {
  if (!model.counts[prev]) model.counts[prev] = {};
  if (!model.counts[prev][next]) model.counts[prev][next] = 0;
  model.counts[prev][next] += 1;
  model.totals[prev] = (model.totals[prev] || 0) + 1;
}

function trainDiff(lines, state, model) {
  const start = Math.min(state.trainedLine || 0, lines.length);
  const newLines = lines.slice(start);
  if (!newLines.length) return { trainedTokens: 0, state };

  let trained = 0;
  for (const line of newLines) {
    const { role, text } = parseLine(line);
    if (role !== "U") continue;
    const tokens = tokenize(text);
    const seq = ["<BOS>", ...tokens, "<EOS>"];
    for (let i = 0; i < seq.length - 1; i++) {
      incCount(model, seq[i], seq[i + 1]);
      trained += 1;
    }
  }

  state.trainedLine = lines.length;
  model.stats.trainedTokens = (model.stats.trainedTokens || 0) + trained;
  return { trainedTokens: trained, state };
}

function softmaxFromCounts(counts, temp) {
  const keys = Object.keys(counts || {});
  if (!keys.length) return { keys: [], probs: [] };
  const temperature = temp > 0 ? temp : 1.0;
  const logits = keys.map((k) => Math.log(counts[k] + 1) / temperature);
  let max = -Infinity;
  for (const val of logits) if (val > max) max = val;
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  const probs = exps.map((v) => v / sum);
  return { keys, probs };
}

function sampleKey(keys, probs) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < keys.length; i++) {
    acc += probs[i];
    if (r <= acc) return keys[i];
  }
  return keys[keys.length - 1];
}

function generateReply(model, options = {}) {
  const maxWords = options.maxWords || model.config.MAX_OUT_WORDS;
  const temp = options.temp || model.config.TEMP;

  const out = [];
  let prev = "<BOS>";
  for (let i = 0; i < maxWords; i++) {
    const counts = model.counts[prev] || model.counts["<BOS>"] || {};
    const { keys, probs } = softmaxFromCounts(counts, temp);
    if (!keys.length) break;
    const next = sampleKey(keys, probs);
    if (next === "<EOS>") break;
    out.push(next);
    prev = next;
  }

  return out.join("");
}

function ensureVocab() {
  return null;
}

module.exports = {
  DEFAULT_CONFIG,
  parseLine,
  tokenize,
  ensureModel,
  trainDiff,
  generateReply,
  ensureVocab
};
