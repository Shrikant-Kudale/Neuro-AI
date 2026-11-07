import { NavLink, Route, Routes, Link, useLocation } from "react-router-dom";
import { Suspense, lazy, useEffect } from "react";
import Home from "./pages/Home";

const Lab = lazy(() => import("./pages/Lab"));
const Analyse = lazy(() => import("./pages/Analyse"));
const Cohort = lazy(() => import("./pages/Cohort"));
const Results = lazy(() => import("./pages/Results"));
const Planner = lazy(() => import("./pages/Planner"));
const Method = lazy(() => import("./pages/Method"));

export const STEPS: [string, string][] = [["/", "Overview"], ["/lab", "Simulate"], ["/analyse", "Analyse"], ["/cohort", "Cohort"], ["/results", "Evidence"], ["/planner", "Plan study"], ["/method", "Method"]];

function ScrollTop() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
}

export default function App() {
  return (
    <>
      <div className="notice"><div className="notice-in"><span><b>Research use only.</b> Not a medical device; not for diagnosis or clinical decisions.</span><span>All processing happens locally in your browser. No data is uploaded.</span></div></div>
      <header className="topbar">
        <div className="topbar-in">
          <Link to="/" className="brand"><b>NeuroPAC</b><span>EEG cross-frequency coupling workbench</span></Link>
          <nav className="nav" aria-label="Main">
            {STEPS.map(([to, label], i) => (
              <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}><i>{i + 1}</i>{label}</NavLink>
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
            <Route path="/analyse" element={<Analyse />} />
            <Route path="/cohort" element={<Cohort />} />
            <Route path="/results" element={<Results />} />
            <Route path="/planner" element={<Planner />} />
            <Route path="/method" element={<Method />} />
            <Route path="*" element={<div className="page"><h1>Page not found</h1><p><Link to="/">Return to the overview</Link></p></div>} />
          </Routes>
        </Suspense>
      </main>
      <footer className="footer">
        <div className="footer-in">
          <p>NeuroPAC is a research prototype. Simulated recordings are generated from stated assumptions; the classifier has not been validated on patient data and must not inform care.</p>
          <p>Developed by Shrikant Kudale · Guide: Prof. Dr. Jayashree Prasad</p>
        </div>
      </footer>
    </>
  );
}
