const base = '/api';

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(base + path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail ?? payload);
    } catch { /* Preserve HTTP status for non-JSON errors. */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const jsonRequest = <T>(path: string, data: unknown, method = 'POST') =>
  request<T>(path, {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});

export function download(name: string, data: BlobPart, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([data], {type}));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
