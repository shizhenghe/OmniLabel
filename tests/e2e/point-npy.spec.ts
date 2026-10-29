import {expect, test} from '@playwright/test';
import {parsePointNpy} from '../../src/api';
import {toRenderPoint} from '../../src/geometry/renderCoordinates';

function npy(dtype: string, values: number[], options: {
  major?: number; shape?: string; fortran?: boolean; unaligned?: boolean;
} = {}): ArrayBuffer {
  const major = options.major ?? 1;
  const offset = major === 1 ? 10 : 12;
  let header = `{'descr': '${dtype}', 'fortran_order': ${options.fortran ? 'True' : 'False'}, 'shape': ${options.shape ?? `(${values.length / 4}, 4)`}, }`;
  const alignment = options.unaligned ? 8 : 64;
  const targetRemainder = options.unaligned ? 1 : 0;
  while ((offset + header.length + 1) % alignment !== targetRemainder) header += ' ';
  header += '\n';
  const itemSize = dtype.endsWith('8') ? 8 : 4;
  const buffer = new ArrayBuffer(offset + header.length + values.length * itemSize);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, ...new TextEncoder().encode('NUMPY'), major, 0]);
  const view = new DataView(buffer);
  if (major === 1) view.setUint16(8, header.length, true);
  else view.setUint32(8, header.length, true);
  bytes.set(new TextEncoder().encode(header), offset);
  values.forEach((value, index) => {
    const position = offset + header.length + index * itemSize;
    if (itemSize === 8) view.setFloat64(position, value, true);
    else view.setFloat32(position, value, true);
  });
  return buffer;
}

for (const major of [1, 2, 3]) {
  for (const dtype of ['<f4', '<f8']) {
    test(`NPY v${major} ${dtype} loads all XYZI points with its original precision`, () => {
      const values = [1.25, -2.5, 0.125, 0.5, 3.5, 4.25, -1.5, 0.75];
      const buffer = npy(dtype, values, {major});
      const points = parsePointNpy(buffer);
      expect(points).toBeInstanceOf(dtype === '<f8' ? Float64Array : Float32Array);
      expect(Array.from(points)).toEqual(values);
      // Aligned network buffers are viewed directly, rather than duplicated.
      expect(points.buffer).toBe(buffer);
    });
  }
}

test('Float64 UTM centimetre detail survives NPY loading and local GPU conversion', () => {
  const values = [500000.001, 3500000.002, 140.003, 0.4,
    500000.011, 3500000.012, 140.008, 0.8];
  for (const unaligned of [false, true]) {
    const points = parsePointNpy(npy('<f8', values, {unaligned}));
    expect(points).toBeInstanceOf(Float64Array);
    expect(Array.from(points)).toEqual(values);
    const origin = Array.from(points.slice(0, 3)) as [number, number, number];
    const second = Array.from(points.slice(4, 7));
    const gpu = new Float32Array(toRenderPoint(second, origin));
    expect(gpu[0]).toBeCloseTo(.01, 7);
    expect(gpu[1]).toBeCloseTo(.01, 7);
    expect(gpu[2]).toBeCloseTo(.005, 7);
    expect(points[5]).toBe(values[5]);
  }
});

test('NPY rejects nonfloating, big-endian, Fortran and non-XYZI layouts', () => {
  const values = [1, 2, 3, 4];
  for (const dtype of ['<i4', '<f2', '>f4', '>f8', '|O']) {
    expect(() => parsePointNpy(npy(dtype, values))).toThrow(/C-order/);
  }
  expect(() => parsePointNpy(npy('<f4', values, {fortran: true}))).toThrow(/C-order/);
  for (const shape of ['(1, 3)', '(4,)', '(1, 4, 1)', '(-1, 4)']) {
    expect(() => parsePointNpy(npy('<f4', values, {shape}))).toThrow(/C-order/);
  }
});

test('NPY rejects invalid versions and inconsistent header or data lengths', () => {
  const original = npy('<f8', [1, 2, 3, 4]);
  expect(() => parsePointNpy(original.slice(0, 8))).toThrow(/NPY/);
  const wrongMagic = original.slice(0);
  new Uint8Array(wrongMagic)[0] = 0;
  expect(() => parsePointNpy(wrongMagic)).toThrow(/有效/);
  const wrongVersion = original.slice(0);
  new Uint8Array(wrongVersion)[6] = 4;
  expect(() => parsePointNpy(wrongVersion)).toThrow(/版本/);
  const longHeader = original.slice(0);
  new DataView(longHeader).setUint16(8, 65535, true);
  expect(() => parsePointNpy(longHeader)).toThrow(/头部长度/);
  expect(() => parsePointNpy(original.slice(0, original.byteLength - 1))).toThrow(/文件长度/);
  const extraByte = new Uint8Array(original.byteLength + 1);
  extraByte.set(new Uint8Array(original));
  expect(() => parsePointNpy(extraByte.buffer)).toThrow(/文件长度/);
  expect(() => parsePointNpy(npy('<f8', [1, 2, 3, 4], {shape: '(9007199254740992, 4)'}))).toThrow(/文件长度/);
});

test('empty XYZI arrays are valid without manufacturing cloud points', () => {
  expect(parsePointNpy(npy('<f8', []))).toHaveLength(0);
});
