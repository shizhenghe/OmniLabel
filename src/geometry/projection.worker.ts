import type {Camera} from '../types';
import {projectCameraPoint, transformPoint} from './cameraProjection';
import {labelColor} from '../labelColors';
let coordinates: Float32Array | Float64Array = new Float64Array();
let stride = 1, components = 3, version = 0;
const colorCanvas = new OffscreenCanvas(1, 1);
const colorContext = colorCanvas.getContext('2d', {willReadFrequently: true})!;
function rgb(value: string) {
  colorContext.clearRect(0, 0, 1, 1);
  colorContext.fillStyle = value;
  colorContext.fillRect(0, 0, 1, 1);
  return colorContext.getImageData(0, 0, 1, 1).data.slice(0, 3);
}
const depthColors = Array.from({length: 211}, (_, hue) => rgb(`hsl(${hue},90%,64%)`));
self.onmessage = event => {
  const message = event.data;
  if (message.type === 'source') {
    coordinates = message.coordinates; stride = message.stride; components = message.components ?? 3; version = message.version;
    return;
  }
  const output: {id: string; key: string; buffer: Float64Array; raster?: {bitmap: ImageBitmap; pointCount: number; pixelCount: number; sourceCount: number}}[] = [];
  const transfers: Transferable[] = [];
  if (message.version === version) for (const item of message.cameras as {camera: Camera; matrix: number[][]; key: string; raster?: boolean; labels?: Record<string, string>}[]) {
    if (!item.raster) {
      const values: number[] = [];
      for (let offset = 0; offset < coordinates.length; offset += components) {
        const pixel = projectCameraPoint(transformPoint(item.matrix, [coordinates[offset], coordinates[offset+1], coordinates[offset+2]]), item.camera);
        if (pixel && pixel[0] >= 0 && pixel[0] < item.camera.width && pixel[1] >= 0 && pixel[1] < item.camera.height)
          values.push(offset/components*stride, pixel[0], pixel[1], pixel[2]);
      }
      const buffer = new Float64Array(values);
      output.push({id: item.camera.id, key: item.key, buffer}); transfers.push(buffer.buffer);
      continue;
    }
    // Every map point participates. Only same-pixel occlusion is resolved;
    // there is no global point budget and no source-order sampling.
    const {width, height, intrinsics: k} = item.camera, m = item.matrix;
    const [k1=0,k2=0,p1=0,p2=0,k3=0] = item.camera.distortion ?? [];
    const distorted=Boolean(k1||k2||p1||p2||k3);
    const depths = new Float64Array(width*height); depths.fill(Infinity);
    const pixels = new Uint8ClampedArray(width*height*4);
    const labelColors = new Map<string, Uint8ClampedArray>();
    let pointCount=0, pixelCount=0;
    const first: number[] = [];
    for (let offset=0;offset<coordinates.length;offset+=components) {
      const a=coordinates[offset], b=coordinates[offset+1], c=coordinates[offset+2];
      const z=m[2][0]*a+m[2][1]*b+m[2][2]*c+m[2][3];
      if (z<=.05) continue;
      const x=(m[0][0]*a+m[0][1]*b+m[0][2]*c+m[0][3])/z;
      const y=(m[1][0]*a+m[1][1]*b+m[1][2]*c+m[1][3])/z;
      let xd=x, yd=y;
      if (distorted) {
        const r2=x*x+y*y, radial=1+k1*r2+k2*r2**2+k3*r2**3;
        xd=x*radial+2*p1*x*y+p2*(r2+2*x*x); yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y;
      }
      const u=k[0][0]*xd+k[0][1]*yd+k[0][2], v=k[1][0]*xd+k[1][1]*yd+k[1][2];
      if (!Number.isFinite(u)||!Number.isFinite(v)||u<0||u>=width||v<0||v>=height) continue;
      const index=offset/components*stride;
      if (!pointCount) first.push(index,u,v,z);
      pointCount++;
      const cell=Math.floor(v)*width+Math.floor(u);
      if (z>=depths[cell]) continue;
      if (depths[cell]===Infinity) pixelCount++;
      depths[cell]=z;
      const label=item.labels?.[String(index)];
      if (label&&!labelColors.has(label)) labelColors.set(label,rgb(labelColor(label)));
      const color=label?labelColors.get(label)!:depthColors[Math.max(0,Math.min(210,Math.round(z*3.4)))];
      const dst=cell*4;pixels[dst]=color[0];pixels[dst+1]=color[1];pixels[dst+2]=color[2];pixels[dst+3]=255;
    }
    // A 3px glyph remains visible when the original image is fitted smaller.
    // Resolve overlapping glyphs by depth; source coordinates are unchanged.
    const glyphPixels=pixels.slice(), glyphDepths=depths.slice();
    for(let row=0;row<height;row++)for(let col=0;col<width;col++){
      const cell=row*width+col, depth=depths[cell];
      if(depth===Infinity)continue;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
        const y=row+dy,x=col+dx;
        if(x<0||x>=width||y<0||y>=height)continue;
        const target=y*width+x;
        if(depth>=glyphDepths[target])continue;
        glyphDepths[target]=depth;
        const from=cell*4,to=target*4;
        glyphPixels[to]=pixels[from];glyphPixels[to+1]=pixels[from+1];glyphPixels[to+2]=pixels[from+2];glyphPixels[to+3]=255;
      }
    }
    const canvas=new OffscreenCanvas(width,height);
    canvas.getContext('2d')!.putImageData(new ImageData(glyphPixels,width,height),0,0);
    const bitmap=canvas.transferToImageBitmap(), buffer=new Float64Array(first);
    output.push({id:item.camera.id,key:item.key,buffer,raster:{bitmap,pointCount,pixelCount,sourceCount:coordinates.length/components}});
    transfers.push(bitmap,buffer.buffer);
  }
  self.postMessage({id:message.id,version,output},{transfer:transfers});
};
