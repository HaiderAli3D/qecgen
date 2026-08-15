/**
 * What every setting means, in the words of the thing it controls.
 *
 * These are drawn from the domain rather than invented for the form: the noise-channel
 * table, the drift-axis semantics, the reproducibility rules and the contract wording all
 * come from README.md, DATA_CONTRACT.md and the enum docstrings. If one of those changes,
 * this file is part of the change — a form that explains a rule the code no longer follows
 * is worse than a form that explains nothing.
 *
 * Each entry has a one-line `summary` for the quick answer, `detail` for the reasoning,
 * and an optional `note` for the trap — the thing that produces a well-formed file
 * containing the wrong data.
 */

export interface Explainer {
  title: string;
  summary: string;
  detail: string[];
  note?: string;
}

export const EXPLAINERS = {
  mode: {
    title: "Run kind",
    summary: "Which of the three dataset shapes to produce.",
    detail: [
      "Single builds one environment and writes one file. Use it when you want a plain training or evaluation set at a fixed noise level.",
      "Multi-environment builds several environments at different points on a noise axis, pools every shot into one file, and labels each shot with the environment it came from. Shots are then shuffled so row order carries no signal. This is the shape invariant-learning methods want: several distinct environments rather than one averaged distribution.",
      "Drift builds a training set plus one test set per drifted value, and records where each test file's error model came from. That last part is the whole point — see the Condition setting.",
    ],
  },

  distance: {
    title: "Code distance",
    summary: "The size of the surface code patch, and how many errors it can correct.",
    detail: [
      "A rotated distance-d patch uses d × d data qubits and d² − 1 stabilizer measurements. It corrects up to ⌊(d−1)/2⌋ errors, so d=3 corrects one, d=5 corrects two, and so on.",
      "Below the threshold error rate, the logical error rate falls roughly exponentially as distance increases — that suppression is the thing decoder benchmarks are usually measuring. Above threshold the relationship inverts and more distance makes results worse.",
      "Cost grows quickly. Detectors per shot is (d² − 1) × rounds, and rounds defaults to the distance, so the per-shot work scales roughly as d³. At d=3 that is 24 detectors per shot; at d=5, 120; at d=7, 336.",
    ],
  },

  rounds: {
    title: "Rounds",
    summary: "How many times the stabilizers are measured before the final data readout.",
    detail: [
      "A memory experiment prepares a logical state, measures the stabilizers repeatedly, then reads out the data qubits. Each round contributes d² − 1 detectors, so the total detector count is (d² − 1) × rounds for the rotated layout.",
      "Left blank, this equals the distance, which is the usual convention for a memory experiment: it gives the code as many rounds to detect a fault as it has distance to correct one.",
      "Fewer rounds than the distance under-measures the code and makes time-like errors look cheaper than they are. More rounds is legitimate but costs proportionally more per shot.",
    ],
    note: "Code capacity noise has perfect measurements, so extra rounds add no information. Selecting it forces one round, and passing a conflicting value is refused rather than silently overridden — a silent override would make the manifest disagree with the file.",
  },

  basis: {
    title: "Memory basis",
    summary: "Which logical observable the experiment preserves and measures.",
    detail: [
      "Memory Z prepares the logical |0⟩ state and measures the logical Z observable at the end; memory X does the same in the conjugate basis.",
      "The choice decides which pair of boundaries the logical operator runs between, and therefore which family of stabilizers dominates the detection events. A decoder that does well on one is not automatically doing well on the other.",
      "For a symmetric code with symmetric noise the two are equivalent by duality. They stop being equivalent as soon as the noise is biased — see the XZ bias drift axis.",
    ],
  },

  rotated: {
    title: "Rotated layout",
    summary: "The compact surface code patch, using roughly half the qubits for the same distance.",
    detail: [
      "The rotated layout needs d² data qubits for distance d. The unrotated layout needs d² + (d−1)², nearly twice as many, and protects against the same number of errors.",
      "Rotated is the standard choice and the default here; unrotated is mostly of interest when comparing against older literature or a specific hardware layout.",
      "The layout changes the qubit count, the stabilizer count and the detector count, so datasets built with different layouts are not comparable even at the same distance.",
    ],
  },

  noise_model: {
    title: "Noise model",
    summary: "Which of Stim's four error channels the error rate is applied to.",
    detail: [
      "Code capacity sets data-qubit depolarisation only, and leaves gate, reset and measurement noise at zero. Measurements are perfect, so it always runs one round. It is the simplest model and the least like hardware.",
      "Phenomenological adds measurement flips to that: data depolarisation and measurement error, still with no gate noise. Multiple rounds now matter, because a measurement can lie.",
      "Uniform circuit level sets all four channels — after-Clifford depolarisation, after-reset flips, before-measurement flips and before-round data depolarisation — to the same rate. It is the default and the closest of the three to a real device.",
    ],
    note: "Setting all four channels to one value is one valid synthetic convention, not a universal definition of \"circuit-level noise\". Hardware-motivated models weight these channels differently and the threshold moves when the convention changes. The resolved channel values are stored in every manifest for exactly this reason — a model name alone is not enough to reproduce a result.",
  },

  p: {
    title: "Physical error rate",
    summary: "The probability applied to each channel the noise model activates.",
    detail: [
      "This is a per-operation probability, not a per-shot one. Which operations it touches depends on the noise model.",
      "For uniform circuit-level noise on the surface code the threshold sits somewhere near 0.5–1%. Below it, increasing the distance suppresses logical errors; above it, increasing the distance makes them worse. Datasets are usually most interesting at a few tenths of a percent up to a couple of percent.",
      "Very small rates produce very few logical errors, so a useful measurement needs proportionally more shots. Very large rates saturate the code and the data stops being informative.",
    ],
  },

  axis: {
    title: "Drift axis",
    summary: "Which property of the noise varies between environments.",
    detail: [
      "Sweeping the uniform rate alone mostly tests interpolation inside one synthetic family. A held-out noise family is much stronger evidence that a decoder has learned something invariant than a held-out rate is, which is why this is not limited to p.",
      "p varies the single uniform rate; the axis value is the rate itself.",
      "Measurement ratio scales measurement error independently of gate error. The axis value multiplies the before-measurement flip probability relative to the base rate, so 1.0 reproduces the plain model and larger values make readout the dominant error source.",
      "XZ bias sets pz / (px + py) on single-qubit noise. The axis value is eta, and 0.5 is the unbiased depolarising point. Higher values make Z errors dominate, which is what biased-noise hardware looks like.",
    ],
    note: "XZ bias rewrites every DEPOLARIZE1 in the generated circuit — both data noise and the depolarisation after single-qubit Cliffords — preserving the total single-qubit error probability. Two-qubit noise is left symmetric, because biasing it correctly needs a 15-parameter channel and a convention nobody has specified. The manifest records the true scope.",
  },

  base_p: {
    title: "Base error rate",
    summary: "The underlying rate that a non-p axis modulates.",
    detail: [
      "When the axis is something other than p, the axis values are not error rates — they are bias ratios or scaling factors. Something still has to say how noisy the device is underneath, and that is this.",
      "For a measurement-ratio sweep, this is the gate error rate, and the axis value scales measurement error relative to it. For an XZ bias sweep, this is the total single-qubit error probability that the rewrite preserves while redistributing it between X, Y and Z.",
    ],
    note: "Required whenever the axis is not p, and ignored when it is. Leaving it out on a bias sweep is refused rather than defaulted, because a guessed base rate would silently change what the dataset measures.",
  },

  axis_values: {
    title: "Environment values",
    summary: "One value per environment, read as coordinates on the chosen axis.",
    detail: [
      "Each value produces its own circuit, its own error model and its own entry in the manifest. Shots from all of them are pooled into a single file and tagged with an environment id.",
      "When the axis is p these are error rates. On any other axis they are that axis's own units — a bias eta, or a measurement scaling factor — and the base rate carries the actual noise level.",
      "Spread matters more than count. Several clearly distinct environments give a learning method something to be invariant across; a tight cluster of near-identical values mostly adds shots.",
    ],
  },

  shots_per_env: {
    title: "Shots per environment",
    summary: "How many shots to draw from each environment before pooling.",
    detail: [
      "Total shots in the file is this multiplied by the number of environment values, and that total is what the preview reports.",
      "Every environment gets the same count, so the pooled dataset is balanced by construction. If you want an unbalanced pool, generate the environments separately.",
    ],
    note: "Multi-environment always holds every shot in memory, because the seeded shuffle needs the whole array resident before it can permute it. There is no streaming variant, so keep an eye on the estimated size in the preview.",
  },

  shuffle: {
    title: "Shuffle shots",
    summary: "Permute the pooled shots so row order carries no environment signal.",
    detail: [
      "Without shuffling, the file is environment 0's shots followed by environment 1's, and so on. A model can then read the environment off the row index instead of learning anything from the physics, and it will score well for entirely the wrong reason.",
      "The permutation is seeded from the master seed and applied to every per-shot array with the same index vector, so detectors, observables, environment ids and mechanism labels all stay in correspondence. The shuffle seed is recorded in the manifest.",
    ],
    note: "Leave this on unless you are debugging and want to see the environments in blocks.",
  },

  train_p: {
    title: "Training error rate",
    summary: "The noise level of the training environment in a drift study.",
    detail: [
      "The drift study produces one training file at this rate and one test file per drifted value, so this is the environment a decoder is calibrated on.",
      "On a non-p axis this is the base rate, and the training environment sits at that axis's unbiased point — 0.5 for XZ bias, 1.0 for measurement ratio — so training represents the undrifted device.",
    ],
  },

  test_values: {
    title: "Test values",
    summary: "One drifted test file per value, on the chosen axis.",
    detail: [
      "Each value becomes its own environment and its own file, named for the value. Together with the training file these form the set a generalisation claim is made over.",
      "Values close to the training point measure interpolation; values well away from it measure whether anything actually transferred.",
    ],
    note: "Two values that print identically at the shown precision — 0.01 and 0.010 — would name the same file and one would overwrite the other. That is refused before any sampling starts, so you find out immediately rather than after the run.",
  },

  condition: {
    title: "Drift condition",
    summary: "Where each test file's error model came from. The most consequential setting here.",
    detail: [
      "Frozen prior ships every test file with the training environment's error model. The decoder is calibrated on training noise and evaluated on drifted noise, so it has to genuinely generalise. This is the honest generalisation experiment.",
      "Oracle calibrated ships each test file with its own environment's error model, handing the decoder the true test-time noise distribution. That is a legitimate measurement — it tells you the ceiling a perfectly calibrated decoder could reach — but it invalidates any claim about generalisation.",
      "Whichever you pick is written into every file's manifest and is never inferred from whichever error model happened to be in scope when the file was built.",
    ],
    note: "Reporting an oracle-calibrated number as evidence of generalisation is the single easiest mistake to make with a drift study, and it is invisible in the results. The recorded condition is what makes it auditable.",
  },

  shots: {
    title: "Shots",
    summary: "How many times to run the whole experiment.",
    detail: [
      "One shot is one complete execution of the circuit: prepare, measure the stabilizers for every round, read out. It produces one row of detection events and one row of logical observable flips.",
      "How many you need depends on how rare the outcome you care about is. Measuring a logical error rate of 1e-4 to within a few percent takes on the order of a million shots; a quick structural check needs a few thousand.",
      "Cost is linear in shots and roughly cubic in distance, so raising the distance is far more expensive than raising the shot count.",
    ],
  },

  seed: {
    title: "Master seed",
    summary: "The single number every random choice in the run is derived from.",
    detail: [
      "Per-environment seeds and the shuffle seed are all spawned from this one value, so the same seed with the same parameters reproduces the same dataset exactly. No global random state is touched anywhere.",
      "Changing the seed gives an independent sample of the same distribution — that is how you get error bars on a measurement without changing what you are measuring.",
    ],
  },

  chunk_size: {
    title: "Chunk size",
    summary: "How many shots are drawn per call to the sampler.",
    detail: [
      "Sampling is chunked so a large run never needs the whole dataset in memory at once. For HDF5, any run larger than one chunk streams straight to disk and peak memory stays flat regardless of how many shots you ask for.",
      "Smaller chunks mean more frequent progress updates and lower peak memory; larger chunks mean slightly less overhead. The default is a reasonable balance.",
    ],
    note: "Chunk size is part of the reproducibility contract, not just a performance knob. It changes the sequence of sampler calls and therefore the exact stream of shots, so the same seed with a different chunk size produces a different — equally valid — dataset. It is recorded in every manifest.",
  },

  fmt: {
    title: "Output format",
    summary: "Which file format to write, and what it can carry.",
    detail: [
      "HDF5 is the primary format and the only one with an incremental writer, so it is the only one that can stream a large run instead of building it in memory first. It round-trips structure fully.",
      "NPZ is a compressed NumPy archive: convenient from Python, round-trips structure, but has to materialise the whole dataset before writing.",
      "Parquet is columnar and widely readable, one row per shot. It stores structure in its metadata but does not reconstruct it on read.",
      "JSONL is one JSON object per shot, with the manifest on the first line. It is the format to reach for when something has to read this without a scientific stack.",
    ],
    note: "Parquet cannot round-trip structure, so a file written with a structure level records the level it can actually reproduce — none — rather than over-claiming. JSONL gets large quickly: it is text, so expect it to dwarf the packed formats.",
  },

  structure_level: {
    title: "Structural detail",
    summary: "How much of the error model to ship alongside the shots.",
    detail: [
      "None writes shots and the manifest only. That is all a decoder needs if it brings its own model of the noise.",
      "Coords adds detector coordinates in space and time, giving the graph layout without the error model attached to it.",
      "DEM adds the parity-check matrix, the logical observable matrix, the per-mechanism prior probabilities and the decomposed component structure. This is what a matching decoder needs to be calibrated.",
      "Full adds the circuit and error-model text as well, stored in a provenance block kept physically apart from the decoder-visible manifest.",
    ],
    note: "One control, not two: the level a file is exported at must match the level it was built at, so there is no way to ask for a file that claims more structure than it holds. For a drift study this setting is what the condition is actually about — with no structure, there is nothing to freeze.",
  },

  emit_mechanisms: {
    title: "Record DEM mechanisms",
    summary: "Additionally record which error mechanisms fired on each shot.",
    detail: [
      "By default a file maps detection events to logical observable flips. Turning this on adds a third array recording which mechanisms of the decomposed error model fired, which is a much richer target.",
      "All three arrays then come from one draw of the error-model sampler rather than from the circuit, so the labels genuinely explain the detection events stored beside them. Sampling the two separately would give two independent random streams and labels that describe a different shot.",
      "It costs space: the mechanism array is far wider than the detector array. A distance-3 patch has 24 detectors but 286 mechanisms.",
    ],
    note: "These are abstract mechanisms in the decomposed noise model, not gate-level physical Pauli faults. The mechanism index is an artifact of construction order and is not portable across noise models or distances — a model trained on these targets is learning to invert one particular error-model construction, not to identify physical faults.",
  },

  out: {
    title: "Output location",
    summary: "Where the run writes, relative to the server's data directory.",
    detail: [
      "Paths are resolved inside the data directory shown beneath the field and cannot escape it. An absolute path or one climbing out with .. is refused.",
      "For single and multi-environment runs this names a file. A drift run names a directory instead, and the individual files inside it are named for the training and test values.",
      "Nothing is written here until the run finishes. Output is staged elsewhere and moved into place atomically at the end, so a cancelled or failed run leaves no file rather than a plausible-looking broken one — and cannot destroy a good file it was about to replace.",
    ],
  },
} as const satisfies Record<string, Explainer>;

export type ExplainerKey = keyof typeof EXPLAINERS;
