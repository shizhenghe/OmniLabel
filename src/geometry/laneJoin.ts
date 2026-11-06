import type {Vector,Vec3} from '../types';
export const joinableLane=(v:Vector)=>!v.closed&&['lane','lane_line','lane_marking','lane_solid','lane_dashed'].includes(v.label)&&v.points.length>=2;
export const endpointDistance=(a:number[],b:number[])=>Math.hypot(...a.map((v,i)=>v-b[i]));
export interface LanePart {id:string;reverse:boolean}
export function orderLaneParts(vectors:Vector[],parts:LanePart[]):LanePart[]{
 const byId=new Map(vectors.map(v=>[v.id,v]));
 const ends=(p:LanePart)=>{const v=byId.get(p.id)!;return p.reverse?[v.points.at(-1)!,v.points[0]]:[v.points[0],v.points.at(-1)!];};
 if(parts.length<2)return parts;
 const chain=[{...parts[0]}],remaining=parts.slice(1).map(p=>({...p}));
 while(remaining.length){
  const head=ends(chain[0])[0],tail=ends(chain.at(-1)!)[1];
  let best={distance:Infinity,index:0,prepend:false,reverse:false};
  remaining.forEach((p,index)=>{for(const reverse of [false,true]){
   const candidate={id:p.id,reverse},[start,end]=ends(candidate);
   for(const prepend of [false,true]){const distance=prepend?endpointDistance(end,head):endpointDistance(tail,start);
    if(distance<best.distance)best={distance,index,prepend,reverse};}
  }});
  const p=remaining.splice(best.index,1)[0];p.reverse=best.reverse;
  if(best.prepend)chain.unshift(p);else chain.push(p);
 }
 return chain;
}
export function connectLanes(vectors:Vector[],parts:LanePart[],maxGap:number){
 if(parts.length<2||new Set(parts.map(p=>p.id)).size!==parts.length)throw new Error('请选择至少两条不同的车道线。');
 if(!Number.isFinite(maxGap)||maxGap<=0)throw new Error('连接最大间距必须大于零。');
 const selected=parts.map(p=>vectors.find(v=>v.id===p.id));
 if(selected.some(v=>!v||!joinableLane(v)))throw new Error('只能连接当前存在的开放车道线。');
 const lanes=selected as Vector[];
 if(lanes.some(v=>v.label!==lanes[0].label))throw new Error('所选车道线类别不同，请先统一类别。');
 const points:Vec3[]=[],gaps:number[]=[];
 parts.forEach((part,i)=>{
  const current=part.reverse?lanes[i].points.slice().reverse():lanes[i].points;
  if(points.length){const gap=endpointDistance(points.at(-1)!,current[0]);gaps.push(gap);
   if(gap>maxGap)throw new Error(`第 ${i}、${i+1} 段相距 ${gap.toFixed(2)} 米，超过连接最大间距；请调整顺序、方向或间距。`);}
  current.forEach(p=>{if(!p.every(Number.isFinite))throw new Error('车道线包含无效坐标。');
   if(!points.length||endpointDistance(points.at(-1)!,p)>1e-8)points.push([...p] as Vec3);});
 });
 const kept={...lanes[0],points,closed:false};
 const ids=new Set(parts.map(p=>p.id));
 return {vectors:vectors.flatMap(v=>v.id===kept.id?[kept]:ids.has(v.id)?[]:[v]),gaps,
  report:{input_count:vectors.length,output_count:vectors.length-parts.length+1,merged_count:parts.length-1,
   groups:[{kept_id:kept.id,source_ids:parts.map(p=>p.id)}]}};
}
