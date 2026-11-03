import {test, expect} from '@playwright/test';
import {request, jsonRequest, requestPointNpy, ApiError} from '../../src/api';

for (const detail of ['实际图像分辨率与内参声明不一致。', [{loc: ['body', 'vectors', 0, 'label'], msg: 'String should have at least 1 character', input: 'DO NOT DISPLAY FULL INPUT'}]]) {
  test(`API errors retain operation and actionable detail: ${typeof detail}`, async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({detail}), {status: 422, statusText: 'Unprocessable Entity'});
    try {
      await expect(jsonRequest('/example/annotations', {}, 'PUT')).rejects.toThrow('422 · PUT /api/example/annotations');
      await expect(request('/example')).rejects.toBeInstanceOf(ApiError);
      await expect(requestPointNpy('/cloud.npy')).rejects.toThrow('422 · GET /api/cloud.npy');
      await expect(request('/example')).rejects.toThrow(typeof detail === 'string' ? detail : 'vectors.0.label');
      try {await request('/example');} catch (error) {expect(String(error)).not.toContain('DO NOT DISPLAY FULL INPUT');}
    } finally {globalThis.fetch = originalFetch;}
  });
}

test('non-JSON errors still report the failing method and endpoint', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('<html>proxy</html>', {status: 422, statusText: 'Unprocessable Entity'});
  try {
    await expect(jsonRequest('/save', {}, 'PUT')).rejects.toThrow('422 · PUT /api/save：Unprocessable Entity');
  } finally {globalThis.fetch = originalFetch;}
});
