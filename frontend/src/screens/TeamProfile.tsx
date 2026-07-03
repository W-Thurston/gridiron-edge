import { useNav } from "../context/NavContext";
import { ScreenPlaceholder } from "./ScreenPlaceholder";

export function TeamProfile() {
  const { route } = useNav();
  const team = route.params.team ?? "(none)";
  return (
    <ScreenPlaceholder title="Team Profile" subtitle={`/teams?team=${team}`} />
  );
}
