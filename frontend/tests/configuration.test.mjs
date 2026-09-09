import assert from "node:assert/strict";
import test from "node:test";
import { parseConfiguration, sweepValues, CONFIG_STARTERS } from "../src/configuration.ts";

test("safe maximum seeds remain exact and uint64 seeds are refused before submission", () => {
  assert.equal(parseConfiguration('{"sampling":{"seed":9007199254740991}}').sampling.seed, 9007199254740991);
  for (const seed of ["9007199254740992", "9007199254740993", "18446744073709551615"]) {
    assert.throws(() => parseConfiguration(`{"sampling":{"seed":${seed}}}`), /exact numeric range/);
  }
});

test("JSON objects preserve unrecognised fields for server validation", () => {
  const raw = '{"mode":"device","extension":{"nested":["keep",false,1]},"noise":{"future":0.2}}';
  assert.deepEqual(parseConfiguration(raw), JSON.parse(raw));
  for (const invalid of ["null", "[]", "7", "{invalid"]) {
    assert.throws(() => parseConfiguration(invalid));
  }
});

test("sweeps preserve zero and false and refuse missing, nonfinite or unsafe values", () => {
  assert.deepEqual(sweepValues("0, 0.01, false, true"), [0, 0.01, false, true]);
  for (const invalid of ["", "1,", "1,,2", "Infinity", "NaN", "no", "9007199254740993"]) {
    assert.throws(() => sweepValues(invalid));
  }
});

test("starter clones retain exact hardware hashes without changing other presets", () => {
  const copy = structuredClone(CONFIG_STARTERS.device);
  copy.sampling.seed = 3;
  assert.equal(CONFIG_STARTERS.device.sampling.seed, 12345);
  assert.equal(CONFIG_STARTERS.hardware.hardware.expected.table_sha256.length, 64);
  assert.equal(CONFIG_STARTERS.dynamic.noise.drift.enabled, true);
  assert.equal(CONFIG_STARTERS.device.noise.drift.enabled, false);
});

test("duplicate fields are rejected at the root, nested objects, and inside arrays", () => {
  const duplicates = [
    '{"version":1,"version":2}',
    '{"sampling":{"seed":1,"seed":2}}',
    '{"noise":{"probabilities":{"measurement":0.01,"measurement":0.02}}}',
    '{"rows":[{"seed":1,"seed":2}]}',
    '{"sampling":{"seed":1,"s\\u0065ed":2}}',
    '{"\\u0076ersion":1,"version":2}',
    '{"":1,"":2}',
    '{"\\\\":1,"\\u005c":2}',
  ];
  for (const raw of duplicates) {
    assert.throws(() => parseConfiguration(raw), /Duplicate configuration field/, raw);
  }
});

test("duplicate detection respects object scopes, escaped strings, and JSON numeric syntax", () => {
  const value = {
    sampling: { seed: 1 }, independent: { seed: 2 }, rows: [{ seed: 3 }, { seed: 4 }],
    description: 'Quotes "seed":1, "seed":2 and braces { } [ ] and slash \\ are text.',
    'quoted"key': { 'slash\\key': false }, empty: {}, array: [], nullValue: null,
  };
  assert.deepEqual(parseConfiguration(JSON.stringify(value, null, 2)), value);
  for (const number of ["0", "-0", "1.2", "-1.2", "1e2", "1E+2", "1e-2", "-1.23E-4"]) {
    const raw = ` { "number" : ${number}, "other" : [true, false, null, {"number":2}] } `;
    assert.deepEqual(parseConfiguration(raw), JSON.parse(raw));
  }
});
