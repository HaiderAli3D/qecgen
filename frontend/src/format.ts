export function bytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let scaled = value / 1024;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled < 10 ? scaled.toFixed(1) : Math.round(scaled)} ${units[unit]}`;
}

export function count(value: number): string {
  return value.toLocaleString("en-US");
}

export function elapsed(from: string | null, to: string | null): string {
  if (!from) return "—";
  const start = Date.parse(from);
  const end = to ? Date.parse(to) : Date.now();
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export function shortHash(value: string | null | undefined): string {
  return value ? `${value.slice(0, 12)}…` : "—";
}

export function when(value: string | number | null): string {
  if (value === null) return "—";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
