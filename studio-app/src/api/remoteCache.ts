interface CacheEntry {
  data: unknown;
  expiresAt: number;
}

const store = new Map<string, CacheEntry>();

export async function cached<T>(
  key: string,
  ttlMs: number,
  fetcher: () => Promise<T>,
): Promise<T> {
  const entry = store.get(key);
  if (entry && (entry.expiresAt === Infinity || Date.now() < entry.expiresAt)) {
    return entry.data as T;
  }
  const data = await fetcher();
  store.set(key, {
    data,
    expiresAt: ttlMs === Infinity ? Infinity : Date.now() + ttlMs,
  });
  return data;
}

export function cacheSet<T>(key: string, data: T, ttlMs: number): void {
  store.set(key, {
    data,
    expiresAt: ttlMs === Infinity ? Infinity : Date.now() + ttlMs,
  });
}

export function invalidate(...prefixes: string[]): void {
  for (const key of store.keys()) {
    if (prefixes.some((p) => key.startsWith(p))) {
      store.delete(key);
    }
  }
}

export function invalidateAll(): void {
  store.clear();
}
