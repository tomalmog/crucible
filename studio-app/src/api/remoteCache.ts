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

export function cacheGet<T>(key: string): T | undefined {
  const entry = store.get(key);
  if (entry && (entry.expiresAt === Infinity || Date.now() < entry.expiresAt)) {
    return entry.data as T;
  }
  return undefined;
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

// Simple semaphore to limit concurrent SSH calls
const MAX_CONCURRENT_SSH = 3;
let running = 0;
const queue: Array<() => void> = [];

export function sshLimited<T>(fn: () => Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const run = () => {
      running++;
      fn().then(resolve, reject).finally(() => {
        running--;
        if (queue.length > 0) queue.shift()!();
      });
    };
    if (running < MAX_CONCURRENT_SSH) run();
    else queue.push(run);
  });
}
