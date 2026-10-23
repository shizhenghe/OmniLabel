type Name = 'cube' | 'layers' | 'lane' | 'map' | 'image' | 'points' | 'grid' | 'chevron' | 'save' | 'upload' | 'download' | 'play' | 'pause' | 'next' | 'prev' | 'select' | 'plus' | 'pen' | 'brush' | 'eye' | 'model' | 'close' | 'trash' | 'check' | 'settings' | 'expand' | 'help';
const paths: Record<Name, React.ReactNode> = {
  cube: <><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z"/><path d="m3 8 9 5 9-5M12 13v8M7 5.8l10 5.5"/></>,
  layers: <><path d="m12 3 10 5-10 5L2 8l10-5Z"/><path d="m2 12 10 5 10-5M2 16l10 5 10-5"/></>,
  lane: <><path d="M3 21 9 3M21 21 15 3M12 3v3m0 4v4m0 4v3"/></>,
  map: <><path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Z"/><path d="M9 3v16M15 5v16"/></>,
  image: <><rect x="3" y="3" width="18" height="18" rx="3"/><path d="m3 17 5-5 4 4 4-6 5 7"/><circle cx="8" cy="8" r="1"/></>,
  points: <>{[[4,4],[12,5],[20,4],[6,12],[15,11],[21,14],[3,20],[12,19],[19,21]].map(([cx,cy],i)=><circle key={i} cx={cx} cy={cy} r="1.2"/>)}</>,
  grid: <><path d="M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z"/></>,
  chevron: <path d="m9 5 7 7-7 7"/>, save: <><path d="M4 3h13l4 4v14H3V3h1ZM7 3v6h10V3M7 21v-8h10v8"/></>,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M3 16v5h18v-5"/></>,
  download: <><path d="M12 3v13m-5-5 5 5 5-5M3 16v5h18v-5"/></>,
  play: <path d="m8 4 12 8-12 8V4Z"/>, pause: <><path d="M7 4h3v16H7zM15 4h3v16h-3z"/></>,
  next: <><path d="m5 5 10 7-10 7V5ZM19 5v14"/></>, prev: <><path d="m19 5-10 7 10 7V5ZM5 5v14"/></>,
  select: <path d="m5 3 14 10-7 1-3 7L5 3Z"/>, plus: <path d="M12 4v16M4 12h16"/>,
  pen: <><path d="m4 16 12-12 4 4L8 20l-5 1 1-5ZM13 7l4 4"/></>,
  brush: <><path d="m9 13 8-10 4 4-10 8M11 15c2 8-8 6-8 6s4-1 2-4c-1-2 3-5 6-2Z"/></>,
  eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></>,
  model: <><path d="m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3 3-7Z"/><path d="m20 1 1 3 3 1-3 1-1 3-1-3-3-1 3-1 1-3Z"/></>,
  close: <path d="m5 5 14 14M5 19 19 5"/>, trash: <><path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/></>,
  check: <path d="m4 12 5 5L20 6"/>, settings: <><circle cx="12" cy="12" r="3"/><path d="m9 3-1 3-3 1-2 3 2 2-1 4 3 2 3-1 3 3 3-1 1-3 3-1v-4l-3-2V6l-4-1-1-2H9Z"/></>,
  expand: <path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5"/>,
  help: <><circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 1 1 5 2l-2 2v1M12 17h.01"/></>,
};
export function Icon({name, size = 18, ...props}: {name: Name; size?: number; className?: string}) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}
export type IconName = Name;
