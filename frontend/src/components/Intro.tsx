import { useState } from "react";

const STORAGE_KEY = "qecgen.intro.open";

/**
 * What the tool is, and what a run actually does.
 *
 * Numbered here because the four stages genuinely are a sequence — the circuit has to
 * exist before the error model can be derived from it, and the error model before shots
 * can be labelled — so the numbers carry information rather than decorating the layout.
 *
 * Collapsible and remembered, because it is orientation rather than reference: useful
 * the first few times, in the way after that.
 */
export function Intro() {
  // Read in the initializer, not an effect: an effect paints the panel open first and
  // collapses it a frame later, a flash on every visit for exactly the returning users
  // who dismissed it.
  const [open, setOpen] = useState(() => window.localStorage.getItem(STORAGE_KEY) !== "closed");

  const toggle = () => {
    setOpen((current) => {
      window.localStorage.setItem(STORAGE_KEY, current ? "closed" : "open");
      return !current;
    });
  };

  return (
    <section className="panel intro">
      <div className="intro-head">
        <div>
          <h2>Syndrome datasets for decoder work</h2>
          <p className="intro-lead">
            A logical qubit is built from many noisy physical ones. This tool simulates one
            and writes down what a decoder would see.
          </p>
        </div>
        <button type="button" onClick={toggle} aria-expanded={open}>
          {open ? "Hide" : "What is this?"}
        </button>
      </div>

      {open && (
        <div className="intro-body">
          <div className="intro-prose">
            <p>
              Physical qubits are unreliable, so error correction spreads one{" "}
              <strong>logical</strong> qubit across a grid of physical ones — a{" "}
              <strong>surface code</strong>. You never measure the data qubits directly,
              because that would destroy the state. Instead, extra qubits repeatedly measure
              parity checks between neighbours. When a check disagrees with its previous
              value it fires, and the pattern of firings is called the{" "}
              <strong>syndrome</strong>.
            </p>
            <p>
              A <strong>decoder</strong> reads that pattern and decides one thing: did the
              logical bit end up flipped? It never sees the errors themselves, only their
              shadow. Building or benchmarking a decoder means feeding it a great many
              worked examples — which is what this generates. Everything here is simulated,
              not measured on hardware.
            </p>
          </div>

          <ol className="pipeline">
            <li>
              <span className="pipeline-step">01</span>
              <h3>Build the circuit</h3>
              <p>
                Stim generates a surface-code memory experiment at your distance, rounds and
                basis, with noise inserted at the rate you set.
              </p>
            </li>
            <li>
              <span className="pipeline-step">02</span>
              <h3>Derive the error model</h3>
              <p>
                The circuit is reduced to a list of independent things that can go wrong:
                what each one would trip, and how likely it is. This is what a decoder gets
                calibrated on.
              </p>
            </li>
            <li>
              <span className="pipeline-step">03</span>
              <h3>Sample shots</h3>
              <p>
                The experiment runs many times, in chunks so memory stays flat. Each run
                yields which checks fired and whether the logical bit flipped.
              </p>
            </li>
            <li>
              <span className="pipeline-step">04</span>
              <h3>Write the file</h3>
              <p>
                Results are bit-packed and written with a manifest recording every
                parameter, library versions and a content hash — so the file says exactly
                how it was made.
              </p>
            </li>
          </ol>

          <p className="intro-outcome">
            Every row of the finished file is one shot:{" "}
            <strong>detection events in, logical flip out</strong>. That mapping is the
            dataset. The error model can travel alongside it, which is what the Structure
            setting controls.
          </p>
        </div>
      )}
    </section>
  );
}
