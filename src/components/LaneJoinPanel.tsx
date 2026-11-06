import {useEffect,useMemo,useState} from 'react';
import type {Vector} from '../types';
import {connectLanes,joinableLane,orderLaneParts,type LanePart} from '../geometry/laneJoin';
import type {LaneMergeResult} from './LaneMergePanel';
interface Props {vectors:Vector[];pick:{id:string;sequence:number}|null;disabled:boolean;onPreview:(r:LaneMergeResult|null)=>void;onApply:(r:LaneMergeResult)=>void;onClose:()=>void}
export function LaneJoinPanel(props:Props){
 const [parts,setParts]=useState<LanePart[]>([]),[gap,setGap]=useState(3),[preview,setPreview]=useState(false);
 const candidates=props.vectors.filter(joinableLane);
 const result=useMemo(()=>{try{return {value:connectLanes(props.vectors,parts,gap),error:''};}catch(e){return {value:null,error:(e as Error).message};}},[props.vectors,parts,gap]);
 useEffect(()=>{
  setPreview(false);
  setParts(old=>{
   const valid=old.filter(p=>props.vectors.some(v=>v.id===p.id&&joinableLane(v)));
   return valid.length===old.length?old:valid;
  });
 },[props.vectors]);
 const toggle=(id:string)=>{setPreview(false);setParts(old=>old.some(p=>p.id===id)?old.filter(p=>p.id!==id):[...old,{id,reverse:false}]);};
 useEffect(()=>{if(!props.disabled&&props.pick&&props.vectors.some(v=>v.id===props.pick!.id&&joinableLane(v)))toggle(props.pick.id);},[props.pick]);
 useEffect(()=>{props.onPreview(preview&&result.value?result.value:null);},[preview,result,props.onPreview]);
 useEffect(()=>()=>props.onPreview(null),[props.onPreview]);
 return <section aria-label="连接选定车道线" className="lane-merge-panel">
 <h3>连接选定车道线</h3><p>勾选列表或在点云上点击片段。按顺序首尾连接，保留原始顶点和高度；间隔处用直线衔接。</p>
 <div style={{maxHeight:180,overflowY:'auto'}}>{candidates.map(v=><label key={v.id} style={{display:'block',padding:5}}><input type="checkbox" aria-label={`连接选择 ${v.id}`} checked={parts.some(p=>p.id===v.id)} disabled={props.disabled} onChange={()=>toggle(v.id)}/>{v.id} · {v.points.length} 点</label>)}</div>
 <p>已选 {parts.length} 段，列表从上到下为连接顺序。</p>
 {parts.map((p,i)=><div key={p.id} style={{display:'flex',gap:5,alignItems:'center',margin:'7px 0'}}><span style={{flex:1,overflowWrap:'anywhere'}}>{i+1}. {p.id}{p.reverse?'（反向）':''}</span>
 <button aria-label={`上移连接段 ${p.id}`} disabled={props.disabled||i===0} onClick={()=>{setPreview(false);setParts(old=>{const a=[...old];[a[i-1],a[i]]=[a[i],a[i-1]];return a;});}}>↑</button>
 <button aria-label={`反转连接段 ${p.id}`} disabled={props.disabled} onClick={()=>{setPreview(false);setParts(old=>old.map(q=>q.id===p.id?{...q,reverse:!q.reverse}:q));}}>反转</button></div>)}
 <label className="field-label">连接最大间距 · 米<input aria-label="连接最大间距" type="number" min="0.01" step=".1" value={gap} disabled={props.disabled} onChange={e=>{setPreview(false);setGap(e.target.valueAsNumber);}}/></label>
 <button className="button subtle full-width" aria-label="自动排列连接段" disabled={props.disabled||parts.length<2} onClick={()=>{setPreview(false);setParts(orderLaneParts(props.vectors,parts));}}>按最近端点自动排序</button>
 {parts.length>=2&&result.error&&<p role="alert">{result.error}</p>}
 {preview&&result.value&&<p role="status">连接间距：{result.value.gaps.map(v=>v.toFixed(2)+'m').join(' / ')}。请检查点云及图像投影。</p>}
 <button className="button subtle full-width" aria-label="预览车道线连接" disabled={props.disabled||!result.value} onClick={()=>setPreview(true)}>预览连接</button>
 <button className="button primary full-width" aria-label="应用车道线连接" disabled={props.disabled||!preview||!result.value} onClick={()=>result.value&&props.onApply(result.value)}>应用连接</button>
 <button className="button ghost full-width" onClick={props.onClose}>取消连接</button>
 </section>;
}
