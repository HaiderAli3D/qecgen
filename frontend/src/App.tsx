import { useEffect, useState } from "react";
import { ApiError, api } from "./api";
import { Datasets } from "./pages/Datasets";
import { NewRun } from "./pages/NewRun";
import { Registry } from "./pages/Registry";
import { Runs } from "./pages/Runs";
import { Score } from "./pages/Score";
import { Sweep } from "./pages/Sweep";
import type { Capabilities } from "./types";

type Route =
  | { page: "new" }
  | { page: "sweep" }
  | { page: "score" }
  | { page: "runs"; id: string | null }
  | { page: "datasets" }
  | { page: "registry" };

/**
 * Hash routing. The server never sees these paths, so a deep link or a refresh works
 * without depending on the SPA catch-all being reachable — which matters when the API is
 * behind the Vite dev proxy.
 */
function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/");
  if (parts[0] === "runs") return { page: "runs", id: parts[1] ?? null };
  if (parts[0] === "datasets") return { page: "datasets" };
  if (parts[0] === "sweep") return { page: "sweep" };
  if (parts[0] === "score") return { page: "score" };
  if (parts[0] === "registry") return { page: "registry" };
  return { page: "new" };
}

const TABS = [
  { href: "#/new", label: "New run", page: "new" },
  { href: "#/sweep", label: "Sweep", page: "sweep" },
  { href: "#/score", label: "Score", page: "score" },
  { href: "#/runs", label: "Runs", page: "runs" },
  { href: "#/datasets", label: "Datasets", page: "datasets" },
  { href: "#/registry", label: "Registry", page: "registry" },
] as const;

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    api
      .capabilities()
      .then(setCaps)
      .catch((err: unknown) =>
        setError(err instanceof ApiError ? err.message : String(err)),
      );
  }, []);

  return (
    <div className="shell">
      <header className="masthead">
        <h1 className="wordmark">
          qec<span>gen</span>
        </h1>
        <p className="tagline">
          Surface-code syndrome datasets. Detection events in, logical frame
          out.
        </p>
        <nav>
          {TABS.map((tab) => (
            <a
              key={tab.page}
              href={tab.href}
              aria-current={route.page === tab.page ? "page" : undefined}
            >
              {tab.label}
            </a>
          ))}
        </nav>
      </header>

      {error && (
        <span className="flag flag--bad">
          Cannot reach the qecgen server: {error}. Is <code>qecgen ui</code>{" "}
          still running?
        </span>
      )}

      {!caps && !error && <p className="empty">Connecting…</p>}

      {caps && route.page === "new" && (
        <NewRun
          caps={caps}
          onSubmitted={(id) => {
            window.location.hash = `#/runs/${id}`;
          }}
        />
      )}

      {caps && route.page === "runs" && (
        <Runs
          selected={route.id}
          onSelect={(id) => {
            window.location.hash = id ? `#/runs/${id}` : "#/runs";
          }}
        />
      )}

      {caps && route.page === "sweep" && (
        <Sweep
          caps={caps}
          onSubmitted={(id) => {
            window.location.hash = `#/runs/${id}`;
          }}
        />
      )}

      {caps && route.page === "score" && (
        <Score
          onSubmitted={(id) => {
            window.location.hash = `#/runs/${id}`;
          }}
        />
      )}

      {caps && route.page === "datasets" && (
        <Datasets
          onSubmitted={(id) => {
            window.location.hash = `#/runs/${id}`;
          }}
        />
      )}

      {caps && route.page === "registry" && <Registry caps={caps} />}

      {caps && (
        <footer
          className="row spread"
          style={{ marginTop: "2rem", color: "var(--muted)", fontSize: "11px" }}
        >
          <span>
            v{caps.version} · writing to <code>{caps.data_root}</code>
          </span>
          <span>
            {caps.max_concurrent_jobs === 1
              ? "One run at a time"
              : `${caps.max_concurrent_jobs} runs at a time`}
          </span>
        </footer>
      )}
    </div>
  );
}
