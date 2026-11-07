import { useProfile } from "../lib/profile";

export default function ProfileSwitch({ compact = false }: { compact?: boolean }) {
  const { profile, setProfile } = useProfile();
  return (
    <div className="seg" role="group" aria-label="Simulation profile">
      <button aria-pressed={profile === "paper"} onClick={() => setProfile("paper")} title="Generator parameters exactly as stated in the paper">
        {compact ? "Paper" : "Paper spec"}
      </button>
      <button aria-pressed={profile === "hard"} onClick={() => setProfile("hard")} title="Adds class overlap, confounders, time-varying disruption and noise">
        {compact ? "Hard" : "Hard mode"}
      </button>
    </div>
  );
}
