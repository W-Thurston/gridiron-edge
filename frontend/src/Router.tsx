import { useNav } from "./context/NavContext";
import { Bankroll } from "./screens/Bankroll";
import { BetSlip } from "./screens/BetSlip";
import { ComparePage } from "./screens/ComparePage";
import { Dashboard } from "./screens/Dashboard";
import { ExplainPage } from "./screens/ExplainPage";
import { GameDetail } from "./screens/GameDetail";
import { GamesList } from "./screens/GamesList";
import { LineShopping } from "./screens/LineShopping";
import { LiveGame } from "./screens/LiveGame";
import { NewsWire } from "./screens/NewsWire";
import { Onboarding } from "./screens/Onboarding";
import { PlayerProp } from "./screens/PlayerProp";
import { PlayersExplorer } from "./screens/PlayersExplorer";
import { PlayoffProjections } from "./screens/PlayoffProjections";
import { Settings } from "./screens/Settings";
import { TeamsScreen } from "./screens/TeamsScreen";
import { Tools } from "./screens/Tools";

export function Router() {
  const { route } = useNav();

  // Routes with query-param variants (detail views) come first —
  // they need to check for the param's presence before falling
  // through to the list variant.
  if (route.path === "/games" && route.params.gameId) return <GameDetail />;
  if (route.path === "/teams" && route.params.team) return <TeamsScreen />;
  if (route.path === "/players" && route.params.propId) return <PlayerProp />;

  switch (route.path) {
    case "/today":
      return <Dashboard />;
    case "/games":
      return <GamesList />;
    case "/teams":
      return <TeamsScreen />;
    case "/projections":
      return <PlayoffProjections />;
    case "/players":
      return <PlayersExplorer />;
    case "/compare":
      return <ComparePage />;
    case "/explain":
      return <ExplainPage />;
    case "/betslip":
      return <BetSlip />;
    case "/mybets":
      return <Bankroll />;
    case "/lines":
      return <LineShopping />;
    case "/live":
      return <LiveGame />;
    case "/news":
      return <NewsWire />;
    case "/tools":
      return <Tools />;
    case "/settings":
      return <Settings />;
    case "/onboarding":
      return <Onboarding />;
    default:
      // Unknown route → dashboard.
      return <Dashboard />;
  }
}
