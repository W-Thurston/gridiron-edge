import { useNav } from "../context/NavContext";
import { ScreenPlaceholder } from "./ScreenPlaceholder";

export function GameDetail() {
  const { route } = useNav();
  const gameId = route.params.gameId ?? "(none)";
  return (
    <ScreenPlaceholder
      title="Game Detail"
      subtitle={`/games?gameId=${gameId}`}
    />
  );
}
