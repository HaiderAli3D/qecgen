import { useId, type ReactNode } from "react";
import type { Explainer } from "../explainers";
import { Info } from "./Info";

interface Labelled {
  label: string;
  /** Opens an "i" beside the label explaining what the setting does. */
  topic?: Explainer;
  hint?: ReactNode;
  /** Rendered under the control and announced. Presence marks the control invalid. */
  error?: string;
  disabled?: boolean;
}

/**
 * A labelled control with an optional explainer.
 *
 * The label and the control are associated by id rather than by nesting. Wrapping both in
 * a `<label>` and putting the info button inside it costs the control its accessible name
 * -- the button is interactive content, and the name computation stops giving a clean
 * result. `htmlFor` keeps the name and lets the button sit outside the label entirely,
 * where clicking it cannot also focus the field.
 *
 * `children` is a function so the generated id reaches whatever control the caller
 * renders, without cloning elements and guessing at their props.
 */
export function Field({
  label,
  topic,
  hint,
  error,
  children,
}: Labelled & { children: (id: string) => ReactNode }) {
  const id = useId();
  return (
    <div className="field">
      <span className="field-head">
        <label className="label" htmlFor={id}>
          {label}
        </label>
        {topic ? <Info topic={topic} /> : null}
      </span>
      {children(id)}
      {hint ? <span className="hint">{hint}</span> : null}
      {error ? (
        <span className="field-error" role="alert">
          {error}
        </span>
      ) : null}
    </div>
  );
}

interface CheckboxProps extends Labelled {
  checked: boolean;
  onChange: (value: boolean) => void;
}

export function Checkbox({ label, topic, checked, onChange, disabled }: CheckboxProps) {
  const id = useId();
  // Same shape as every other cell -- label row, then control. Putting the box and a
  // wrappable label on one line looks fine until the label wraps, and then the explainer
  // ends up stranded on a line of its own.
  return (
    <div className="field check">
      <span className="field-head">
        <label className="label" htmlFor={id}>
          {label}
        </label>
        {topic ? <Info topic={topic} /> : null}
      </span>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        autoComplete="off"
        onChange={(event) => onChange(event.target.checked)}
      />
    </div>
  );
}

interface SelectProps extends Labelled {
  value: string;
  options: readonly string[];
  onChange: (value: string) => void;
}

export function Select({ label, topic, value, options, onChange, hint, disabled }: SelectProps) {
  return (
    <Field label={label} topic={topic} hint={hint}>
      {(id) => (
        <select
          id={id}
          value={value}
          disabled={disabled}
          autoComplete="off"
          onChange={(e) => onChange(e.target.value)}
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      )}
    </Field>
  );
}

interface NumberFieldProps extends Labelled {
  value: number | "";
  onChange: (value: number | "") => void;
  step?: number | string;
  min?: number;
  placeholder?: string;
}

export function NumberField({
  label,
  topic,
  value,
  onChange,
  step,
  min,
  hint,
  disabled,
  placeholder,
}: NumberFieldProps) {
  return (
    <Field label={label} topic={topic} hint={hint}>
      {(id) => (
        <input
          id={id}
          type="number"
          value={value}
          step={step}
          min={min}
          disabled={disabled}
          placeholder={placeholder}
          autoComplete="off"
          onChange={(event) => {
            const raw = event.target.value;
            onChange(raw === "" ? "" : Number(raw));
          }}
        />
      )}
    </Field>
  );
}

interface ListFieldProps extends Labelled {
  value: string;
  onChange: (value: string) => void;
}

/** Comma-separated numbers. Typing a list is faster than managing a row of inputs. */
export function ListField({ label, topic, value, onChange, hint, error }: ListFieldProps) {
  return (
    <Field label={label} topic={topic} hint={hint} error={error}>
      {(id) => (
        <input
          id={id}
          value={value}
          autoComplete="off"
          aria-invalid={error ? true : undefined}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </Field>
  );
}

/**
 * The parseable numbers in a comma-separated list.
 *
 * Never enough on its own to decide what a run does: pair it with `invalidListTokens`.
 * Submitting only the tokens that parsed would quietly narrow the run — a typo in one
 * value and the file holds fewer environments or test sets than were typed.
 */
export function parseList(text: string): number[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map(Number)
    .filter((value) => Number.isFinite(value));
}

/** The tokens `parseList` would drop, so a form can show them and refuse to submit. */
export function invalidListTokens(text: string): string[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0 && !Number.isFinite(Number(part)));
}
