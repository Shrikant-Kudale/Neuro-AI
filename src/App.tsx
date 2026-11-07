import { NavLink, Route, Routes, Link, useLocation } from "react-router-dom";
import { Suspense, lazy, useEffect } from "react";
import { ProfileProvider } from "./lib/profile";
import Home from "./pages/Home";

const Lab = lazy(() => import("./pages/Lab"));
const Cohort = lazy(() => import("./pages/Cohort"));
const Results = lazy(() => import("./pages/Results"));
const Method = lazy(() => import("./pages/Method"));
const About = lazy(() => import("./pages/About"));

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="14" fill="var(--ink)" />
      <path d="M6 34 Q14 10 22 34 T38 34 T58 34" fill="none" stroke="var(--paper)" strokeWidth="5" strokeLinecap="round" />
      <path d="M10 46h44" stroke="var(--high)" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

function ScrollTop() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
}

export default function App() {
  return (
    <ProfileProvider>
      <header className="topbar">
        <div className="topbar-in">
          <Link to="/" className="brand"><Logo />NeuroPAC Lab</Link>
          <nav className="nav" aria-label="Main">
            {[["/lab", "Patient lab"], ["/cohort", "Cohort"], ["/results", "Results"], ["/method", "Method"], ["/about", "Paper"]].map(([to, label]) => (
              <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>{label}</NavLink>
            ))}
          </nav>
        </div>
      </header>
      <ScrollTop />
      <main>
        <Suspense fallback={<div className="page"><p>Loading…</p></div>}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/lab" element={<Lab />} />
            <Route path="/cohort" element={<Cohort />} />
            <Route path="/results" element={<Results />} />
            <Route path="/method" element={<Method />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<div className="page"><h1>Page not found</h1><p><Link to="/">Back to the lab home</Link></p></div>} />
          </Routes>
        </Suspense>
      </main>
      <footer className="footer">
        <div className="footer-in">
          <p><b>Research prototype — not a medical device.</b> All EEG in this application is simulated. Outputs are model estimates on synthetic data and must never inform a clinical decision.</p>
          <p>MIT School of Computing, MIT-ADT University, Pune · Kudale, Surani, Saud &amp; Prasad · IJARSCT 6(17), 2026 · DOI 10.48175/IJARSCT-34249</p>
        </div>
      </footer>
    </ProfileProvider>
  );
}
