'use client';
import {useCallback, useEffect, useRef, useState} from 'react';

type Clip = {id:string; split:string; title:string; video_url:string; options:string[]; normal_line:string; decision_at_s:number; context?:string; role?:string};
type Saved = {choice:number; reason:string; request_id:string; trajectory_url:string};
export default function Review() {
  const [clips,setClips] = useState<Clip[]>([]);
  const [reviews,setReviews] = useState<Record<string,Saved>>({});
  const [index,setIndex] = useState(0);
  const [choice,setChoice] = useState<number|null>(null);
  const [reason,setReason] = useState('');
  const [connected,setConnected] = useState(false);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const requestId = useRef('');
  const player = useRef<HTMLVideoElement>(null);
  const selectClip = useCallback((nextIndex:number, saved?:Saved) => {
    setIndex(nextIndex);setChoice(saved?.choice??null);setReason(saved?.reason||'');
    requestId.current=saved?.request_id||'';setError('');
  }, []);
  useEffect(()=>{
    Promise.all([fetch('/commentary-learning/catalog.json').then(r=>r.json()),fetch('/api/commentary-learning').then(r=>r.json())])
      .then(([catalog,state])=>{setClips(catalog.cases);setConnected(!!state.connected);setReviews(state.reviews||{});
        const first=catalog.cases.findIndex((c:Clip)=>!state.reviews?.[c.id]);const next=first<0?0:first;
        selectClip(next,state.reviews?.[catalog.cases[next]?.id]);})
      .catch(()=>setError('The review set is not ready yet. Refresh shortly.'));
  },[selectClip]);
  const clip=clips[index]; const saved=clip?reviews[clip.id]:undefined;
  function listen(text:string) {
    player.current?.pause();
    speechSynthesis.cancel();
    if(text!=='[SILENCE]')speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
  async function save() {
    if(!clip||choice===null) return;
    if(!requestId.current)requestId.current=crypto.randomUUID();
    setBusy(true);setError('');
    try {
      const response=await fetch('/api/commentary-learning',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:'review',case_id:clip.id,choice,reason,request_id:requestId.current})});
      const result=await response.json();if(!response.ok||result.error)throw Error(result.error||'Could not save.');
      setReviews(result.reviews);const next=Math.min(index+1,clips.length-1);selectClip(next,result.reviews[clips[next].id]);
    }catch(e){setError(e instanceof Error?e.message:'Could not save.');}finally{setBusy(false);}
  }
  return <main className="editor-shell">
    <header className="editor-header"><h1>Choose what to air</h1><span>{Object.keys(reviews).length} / {clips.length||0} saved</span></header>
    <p style={{color:'#666',margin:'20px 0 28px'}}>One short moment. Choose the line you would actually air.</p>
    {!connected&&<p>Open this page in the connected local session to save reviews to Trajectory.</p>}
    {clip&&<div className="editor-workspace">
      <section className="editor-footage"><h2>{index+1}. {clip.title} · {clip.role || (clip.split==='train'?'Learning clip':'Test clip — excluded from training')}</h2>
        <video key={clip.id} ref={player} src={clip.video_url} controls muted playsInline preload="metadata"/>
        <button className="editor-secondary editor-replay" onClick={()=>{if(player.current){player.current.currentTime=0;player.current.play();}}}>Replay</button>
        <p style={{color:'#777',fontSize:13}}>{clip.context} The line starts {clip.decision_at_s.toFixed(1)}s into this clip.</p>
      </section>
      <section className="editor-form"><h2>What would you air now?</h2>
        {clip.options.map((line,i)=><div key={i} style={{display:'flex',gap:12,padding:'14px 0',borderBottom:'1px solid #eee',alignItems:'baseline'}}>
          <label style={{display:'flex',gap:12,flex:1,cursor:saved?'default':'pointer'}}>
            <input type="radio" name="line" checked={choice===i} disabled={busy||!!saved||!connected} onChange={()=>{setChoice(i);requestId.current='';}}/>
            <span>{line==='[SILENCE]'?'Stay silent':line}</span>
          </label>
          {line!=='[SILENCE]'&&<button className="editor-text-button" onClick={()=>listen(line)}>Listen</button>}
        </div>)}
        <div className="editor-field" style={{marginTop:20}}><label htmlFor="why">Why? (optional)</label>
          <textarea id="why" rows={2} maxLength={1200} value={reason} disabled={busy||!!saved||!connected} onChange={e=>{setReason(e.target.value);requestId.current='';}}/>
        </div>
        {!saved?<button className="editor-primary" disabled={!connected||busy||choice===null} onClick={save}>{busy?'Saving to Trajectory…':'Save and next'}</button>:
          <p>Saved to Trajectory. <a href={saved.trajectory_url} target="_blank" rel="noreferrer">View review</a></p>}
        {error&&<p role="alert">{error}</p>}
      </section>
    </div>}
    <footer className="editor-footer"><div style={{display:'flex',gap:16}}>
      <button className="editor-text-button" disabled={busy||index===0} onClick={()=>selectClip(index-1,reviews[clips[index-1].id])}>Previous</button>
      <button className="editor-text-button" disabled={busy||index>=clips.length-1} onClick={()=>selectClip(index+1,reviews[clips[index+1].id])}>Next</button>
      </div><span>{clips.length>0&&Object.keys(reviews).length===clips.length?'Examples reviewed. No training has started.':'Saved to Trajectory when you choose a line.'}</span>
    </footer>
  </main>;
}
