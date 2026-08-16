import type { Capabilities } from "../types";

/**
 * What this build can actually do: the export formats and the decoders.
 *
 * Every value here is read from a live registry on the server rather than restated, so a
 * newly registered exporter or a newly installed decoder backend appears with no change
 * to this page.
 */
export function Registry({ caps }: { caps: Capabilities }) {
  return (
    <div className="stack">
      <div className="panel" style={{ padding: "1.25rem" }}>
        <div className="section-head">
          <h2>Export formats</h2>
          <span className="tag">{caps.formats.length} registered</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Format</th>
              <th>Extension</th>
              <th>Streams</th>
              <th>Round-trips structure</th>
              <th>Carries provenance</th>
            </tr>
          </thead>
          <tbody>
            {caps.formats.map((format) => (
              <tr key={format.name}>
                <td>
                  <span className="tag">{format.name}</span>
                </td>
                <td>{format.extension}</td>
                <td>{format.streaming ? "yes" : "no"}</td>
                <td>
                  {format.structure_round_trip
                    ? "yes"
                    : "no (arrays + manifest only)"}
                </td>
                <td>
                  {format.carries_provenance
                    ? "yes (circuit + DEM text at full)"
                    : "no (full records dem)"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          Only HDF5 streams, so it is the only format whose memory stays flat
          regardless of shot count. A format that cannot round-trip structure
          records <code>structure_level: none</code> rather than claiming more
          than it holds, and a format that declines provenance records{" "}
          <code>dem</code> at full for the same reason — the recorded level is
          the one field a reader cannot check against the file.
        </p>
        <p className="note">
          No Nexus exporter is registered. The Nexus input format is not yet
          known; adding it is one module plus one line in{" "}
          <code>qecgen/exporters/__init__.py</code>.
        </p>
      </div>

      <div className="panel" style={{ padding: "1.25rem" }}>
        <div className="section-head">
          <h2>Decoders</h2>
          <span className="tag">
            {caps.decoders.filter((decoder) => decoder.usable).length} usable of{" "}
            {caps.decoders.length}
          </span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Backing package</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {caps.decoders.map((decoder) => (
              <tr key={decoder.name}>
                <td>{decoder.name}</td>
                <td>{decoder.backing_package ?? "built in"}</td>
                <td>
                  {decoder.usable ? (
                    "available"
                  ) : (
                    <span className="flag" style={{ marginTop: 0 }}>
                      {decoder.problem ?? "unavailable"}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          These names are resolved against sinter's built-in decoders and their
          backends are probed by module name — this tool implements no decoder
          of its own and imports none of these packages. A bespoke decoder needs
          no code here: <code>sinter.collect(custom_decoders=…)</code> is the
          extension point.
        </p>
        <p className="note">
          The statistical QA oracle is always PyMatching and is not affected by
          any of this. An oracle that could be set to a decoder under test would
          not be an oracle.
        </p>
      </div>

      <div className="panel" style={{ padding: "1.25rem" }}>
        <h2>Drift axes</h2>
        {/* A table rather than the `.checks` list: that list's first column is 3.4rem
            wide, which breaks `measurement_ratio` across three lines mid-word. */}
        <table>
          <thead>
            <tr>
              <th>Axis</th>
              <th>What it varies</th>
            </tr>
          </thead>
          <tbody>
            {caps.drift_axes.map((axis) => (
              <tr key={axis}>
                <td>{axis}</td>
                <td>
                  {axis === "p"
                    ? "the physical error rate itself"
                    : axis === "xz_bias"
                      ? "the ratio of Z-type to X-type noise, at a fixed total rate"
                      : axis === "measurement_ratio"
                        ? "measurement error relative to gate error"
                        : "a registered drift axis"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          Every axis varies the <em>noise</em> and never the code, which is what
          makes the correction schema stable across a drift study.
        </p>
      </div>

      <div className="panel" style={{ padding: "1.25rem" }}>
        <h2>This server</h2>
        <dl className="readout">
          <dt>Version</dt>
          <dd>{caps.version}</dd>
          <dt>Data root</dt>
          <dd className="truncate" title={caps.data_root}>
            {caps.data_root}
          </dd>
          <dt>Run records</dt>
          <dd className="truncate" title={caps.runs_dir}>
            {caps.runs_dir}
          </dd>
          <dt>Concurrent jobs</dt>
          <dd>{caps.max_concurrent_jobs}</dd>
        </dl>
        <p className="note">
          Loopback only, with no authentication, and every path the browser
          sends is resolved and required to sit under the data root before it is
          used.
        </p>
      </div>
    </div>
  );
}
