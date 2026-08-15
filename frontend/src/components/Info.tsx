import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Explainer } from "../explainers";

const GAP = 6;
const MARGIN = 8;

/**
 * The panel currently open, if any.
 *
 * Each trigger owns its own state, so without this two could be open at once. Clicking a
 * second trigger normally dismisses the first through the outside-mousedown handler — but
 * activating one from the keyboard fires no mousedown at all, and then both stay up.
 */
let dismissOpenPanel: (() => void) | null = null;

/**
 * The "i" beside a field label, and the panel it opens.
 *
 * Rendered through a portal onto the body rather than inline, because the form is a grid
 * of fixed-width columns and an inline panel would be clipped by the first one with
 * `overflow` set. Portalling also means the panel is never trapped inside the sticky
 * sidebar.
 *
 * Position is computed from the trigger's rect after the panel has been measured, so it
 * can flip above the trigger near the bottom of the window and clamp horizontally near
 * the right edge, and it is recomputed as the page scrolls so it stays anchored.
 *
 * It deliberately does *not* close on scroll. The panel carries its own scrollbar for the
 * longer entries, and a scroll listener that closes on any event dismisses the panel the
 * moment you scroll the text you are reading.
 */
export function Info({ topic }: { topic: Explainer }) {
  const [open, setOpen] = useState(false);
  const [placed, setPlaced] = useState<{ top: number; left: number } | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);

  const toggle = useCallback(() => setOpen((current) => !current), []);

  /*
   * Claiming the singleton happens here, not inside the setState updater.
   *
   * An updater that mutates module state and calls a setter is impure, and StrictMode
   * deliberately invokes updaters twice to surface exactly that: the first call installs this
   * panel's closer, the second sees it and immediately calls it. The panel then opens and
   * shuts within one click and `aria-expanded` never leaves "false".
   *
   * That double-invocation only happens in development, so the bug was invisible in the
   * bundle `qecgen ui` serves and appeared only under `npm run dev` — the one mode in which
   * the frontend is actually worked on.
   *
   * The identity check in the cleanup is what makes this safe when a second trigger takes
   * over: by then the module holds the newcomer's closer, so this one must not null it.
   */
  useEffect(() => {
    if (!open) return;
    const closeSelf = () => setOpen(false);
    dismissOpenPanel?.();
    dismissOpenPanel = closeSelf;
    return () => {
      if (dismissOpenPanel === closeSelf) dismissOpenPanel = null;
    };
  }, [open]);

  const place = useCallback(() => {
    const button = trigger.current;
    const box = panel.current;
    if (!button || !box) return;

    const anchor = button.getBoundingClientRect();
    const { width, height } = box.getBoundingClientRect();
    const left = Math.max(MARGIN, Math.min(anchor.left, window.innerWidth - width - MARGIN));

    const below = anchor.bottom + GAP;
    const fitsBelow = below + height <= window.innerHeight - MARGIN;
    const above = anchor.top - height - GAP;
    const top = fitsBelow
      ? below
      : above >= MARGIN
        ? above
        : Math.max(MARGIN, window.innerHeight - height - MARGIN);

    setPlaced((current) =>
      current && current.top === top && current.left === left ? current : { top, left },
    );
  }, []);

  useLayoutEffect(() => {
    if (!open) {
      setPlaced(null);
      return;
    }
    place();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    // Closing is enough: the effect above owns the singleton and releases it on cleanup.
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      trigger.current?.focus();
    };
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (panel.current?.contains(target) || trigger.current?.contains(target)) return;
      setOpen(false);
    };
    const onViewportChange = (event: Event) => {
      // Scrolling inside the panel must not move it. Only page-level scrolling does.
      if (event.target instanceof Node && panel.current?.contains(event.target)) return;
      place();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onPointerDown);
    window.addEventListener("resize", onViewportChange);
    window.addEventListener("scroll", onViewportChange, true);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("resize", onViewportChange);
      window.removeEventListener("scroll", onViewportChange, true);
    };
  }, [open, place]);

  return (
    <>
      <button
        ref={trigger}
        type="button"
        className="info-trigger"
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={`What is ${topic.title}?`}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          toggle();
        }}
      >
        i
      </button>
      {open &&
        createPortal(
          <div
            ref={panel}
            className="info-panel"
            role="dialog"
            aria-label={topic.title}
            // Rendered off-screen for one frame so it can be measured before it is placed;
            // showing it at 0,0 first would make every popup flash in the corner.
            style={
              placed
                ? { top: placed.top, left: placed.left }
                : { top: 0, left: 0, visibility: "hidden" }
            }
          >
            <h4>{topic.title}</h4>
            <p className="info-summary">{topic.summary}</p>
            {topic.detail.map((paragraph) => (
              <p key={paragraph.slice(0, 32)}>{paragraph}</p>
            ))}
            {topic.note && <p className="info-note">{topic.note}</p>}
          </div>,
          document.body,
        )}
    </>
  );
}
