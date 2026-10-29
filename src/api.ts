import type {PointBuffer} from './types';

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

/** Parse an XYZI NumPy cloud without downcasting world coordinates. */
export function parsePointNpy(buffer: ArrayBuffer): PointBuffer {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== 'NUMPY') {
    throw new Error('点云二进制文件不是有效的 NPY');
  }
  const major = bytes[6];
  if (![1, 2, 3].includes(major) || bytes[7] !== 0 || (major !== 1 && bytes.length < 12)) {
    throw new Error('不支持的点云 NPY 版本');
  }
  const view = new DataView(buffer);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const dataOffset = headerOffset + headerLength;
  if (!headerLength || dataOffset > buffer.byteLength) throw new Error('点云 NPY 的头部长度不一致');
  const header = new TextDecoder(major === 3 ? 'utf-8' : 'ascii').decode(bytes.subarray(headerOffset, dataOffset));
  const shape = header.match(/['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*4\s*,?\s*\)/);
  const dtype = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  if (!/^\s*\{[\s\S]*\}\s*$/.test(header) || !header.endsWith('\n') || !shape
    || !['<f4', '<f8'].includes(dtype?.[1] ?? '') || !/['"]fortran_order['"]\s*:\s*False\b/.test(header)) {
    throw new Error('点云 NPY 必须是 C-order (N, 4) little-endian float32 或 float64');
  }
  const count = Number(shape[1]);
  const valueCount = count * 4;
  const itemSize = dtype![1] === '<f8' ? 8 : 4;
  const byteCount = valueCount * itemSize;
  if (!Number.isSafeInteger(count) || !Number.isSafeInteger(valueCount) || !Number.isSafeInteger(byteCount)
    || dataOffset + byteCount !== buffer.byteLength) {
    throw new Error('点云 NPY 的形状或文件长度不一致');
  }
  // Unaligned headers need a byte copy, never a Float32 numeric conversion.
  const alignedBuffer = dataOffset % itemSize === 0 ? buffer : buffer.slice(dataOffset);
  const offset = alignedBuffer === buffer ? dataOffset : 0;
  return itemSize === 8 ? new Float64Array(alignedBuffer, offset, valueCount)
    : new Float32Array(alignedBuffer, offset, valueCount);
}

/** Load a C-order (N, 4) little-endian Float32/Float64 array without JSON expansion. */
export async function requestPointNpy(path: string): Promise<PointBuffer> {
  const response = await fetch(base + path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return parsePointNpy(await response.arrayBuffer());
}

export function download(name: string, data: BlobPart, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([data], {type}));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
