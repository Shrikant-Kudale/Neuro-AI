import { Link } from "react-router-dom";

export default function StepNav({ prev, next }: { prev?: [string, string]; next?: [string, string] }) {
  return (
    <nav className="stepnav" aria-label="Workflow">
      {prev ? <Link to={prev[0]}><small>Previous step</small><b>{prev[1]}</b></Link> : <span />}
      {next && <Link to={next[0]} className="next"><small>Next step</small><b>{next[1]}</b></Link>}
    </nav>
  );
}

export function PageHead({ step, title, children }: { step: string; title: string; children?: React.ReactNode }) {
  return <header className="pagehead"><div className="crumb">{step}</div><h1>{title}</h1>{children && <p className="lede">{children}</p>}</header>;
}
