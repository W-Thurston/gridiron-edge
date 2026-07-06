import { FeaturedMatchupsGrid } from "../components/dashboard/FeaturedMatchupsGrid";
import { ModelEdgesTable } from "../components/dashboard/ModelEdgesTable";
import { ModelPerformanceRail } from "../components/dashboard/ModelPerformanceRail";
import { PropEdgesRail } from "../components/dashboard/PropEdgesRail";

/**
 * Primary landing page for Gridiron Edge.
 *
 * Layout:
 * - Top row: FeaturedMatchupsGrid (3-card row of top matchups)
 * - Bottom row: Two-column split
 *   - Left (~60%): ModelEdgesTable with filter tabs
 *   - Right (~40%): Stacked ModelPerformanceRail + PropEdgesRail
 *
 * Each section handles its own data loading and error states.
 * When backend data is unavailable, sections render actionable
 * empty states with CLI hints.
 */
export function Dashboard() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Top row: Featured matchups (full width) */}
      <FeaturedMatchupsGrid />

      {/* Bottom row: Edges table + right rail */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "3fr 2fr",
          gap: 16,
        }}
      >
        <ModelEdgesTable />
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <ModelPerformanceRail />
          <PropEdgesRail />
        </div>
      </div>
    </div>
  );
}
