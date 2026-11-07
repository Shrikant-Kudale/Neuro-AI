import { createContext, useContext, useState, ReactNode } from "react";
import type { ProfileName } from "./data";

const Ctx = createContext<{ profile: ProfileName; setProfile: (p: ProfileName) => void }>({ profile: "paper", setProfile: () => {} });
export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfile] = useState<ProfileName>("paper");
  return <Ctx.Provider value={{ profile, setProfile }}>{children}</Ctx.Provider>;
}
export const useProfile = () => useContext(Ctx);
