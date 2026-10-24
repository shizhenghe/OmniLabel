let sequence = 0;

/** Annotation identifiers also work on HTTP intranet deployments. */
export function annotationId(prefix: string): string {
  const crypto = globalThis.crypto;
  if (typeof crypto?.randomUUID === 'function') return `${prefix}-${crypto.randomUUID()}`;
  if (typeof crypto?.getRandomValues === 'function') {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return `${prefix}-${Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')}`;
  }
  sequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${sequence.toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
}
