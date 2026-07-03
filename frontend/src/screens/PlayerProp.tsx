import { useNav } from "../context/NavContext";
import { ScreenPlaceholder } from "./ScreenPlaceholder";

export function PlayerProp() {
  const { route } = useNav();
  const propId = route.params.propId ?? "(none)";
  return (
    <ScreenPlaceholder
      title="Player Prop"
      subtitle={`/players?propId=${propId}`}
    />
  );
}
``
