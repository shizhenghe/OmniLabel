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

/** Load a C-order (N, 4) little-endian Float32 NumPy array without JSON expansion. */
export async function requestPointNpy(path: string): Promise<Float32Array> {
  const response = await fetch(base + path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12 || bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== 'NUMPY') {
    throw new Error('点云二进制文件不是有效的 NPY');
  }
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const dataOffset = headerOffset + headerLength;
  const header = new TextDecoder('ascii').decode(bytes.subarray(headerOffset, dataOffset));
  const shape = header.match(/['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*4\s*,?\s*\)/);
  const dtype = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  if (!shape || dtype?.[1] !== '<f4' || !/['"]fortran_order['"]\s*:\s*False/.test(header)) {
    throw new Error('点云 NPY 必须是 C-order (N, 4) little-endian float32');
  }
  const count = Number(shape[1]);
  const valueCount = count * 4;
  if (!Number.isSafeInteger(count) || count <= 0 || dataOffset + valueCount * 4 !== buffer.byteLength) {
    throw new Error('点云 NPY 的形状或文件长度不一致');
  }
  if (dataOffset % 4 === 0) return new Float32Array(buffer, dataOffset, valueCount);
  return new Float32Array(buffer.slice(dataOffset));
}

export function download(name: string, data: BlobPart, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([data], {type}));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
