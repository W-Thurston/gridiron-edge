import { useTeamProfile } from "../api/hooks";
import { BlockedField } from "../components/field-status/BlockedField";
import { PendingField } from "../components/field-status/PendingField";
import type { FieldStatus } from "../components/field-status/types";
import { Spark } from "../components/primitives/Spark";
import { RecentResultsStrip } from "../components/teams/RecentResultsStrip";
import { TeamMark } from "../components/primitives/TeamMark";
import { useNav } from "../context/NavContext";
import { ErrorCard } from "../components/error/ErrorCard";

export function TeamProfile() {
  const { route, navigate } = useNav();
  const abbr = route.params.team ?? null;
  const { data, isLoading, error, refetch } = useTeamProfile(abbr);

  if (!abbr) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">No team selected.</div>
      </div>
    );
  }

  const backNav = (
    <div>
      <button
        type="button"
        onClick={() => navigate("/teams")}
        className="dim mono"
        style={{
          background: "transparent",
          border: "none",
          padding: 0,
          cursor: "pointer",
          font: "inherit",
          color: "var(--ink-3)",
          fontSize: 12,
        }}
      >
        ← Team Rankings
      </button>
    </div>
  );

  if (isLoading) {
    return (
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="dim">Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {backNav}
        <ErrorCard
          error={error}
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const fieldStatus = data._meta?.field_status;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {backNav}
      {/* Header card */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Team Profile — {data.season} through Week {data.as_of_week}
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontSize: 20,
          }}
        >
          <TeamMark abbr={data.abbr} />
          <span>{data.name}</span>
        </div>

        <div style={{ display: "flex", gap: 32, marginTop: 20, flexWrap: "wrap" }}>
          <ProfileCell label="Rating" value={data.rating?.toFixed(0) ?? "—"} />
          <ProfileCell label="Rank" value={data.rank?.toString() ?? "—"} />
          <ProfileCell
            label="Record"
            value={
              data.record
                ? `${data.record.wins}-${data.record.losses}${
                    data.record.ties > 0 ? `-${data.record.ties}` : ""
                  }`
                : "—"
            }
          />
          <ProfileCell
            label="Off Rating"
            value={<InlineFieldStatus status={fieldStatus?.off_rating as FieldStatus | undefined} />}
          />
          <ProfileCell
            label="Def Rating"
            value={<InlineFieldStatus status={fieldStatus?.def_rating as FieldStatus | undefined} />}
          />
          <ProfileCell
            label="Trend"
            value={<InlineFieldStatus status={fieldStatus?.trend as FieldStatus | undefined} />}
          />
          <ProfileCell
            label="Sched. Difficulty"
            value={<InlineFieldStatus status={fieldStatus?.schedule_difficulty as FieldStatus | undefined} />}
          />
          <ProfileCell
            label="Playoff Prob."
            value={<InlineFieldStatus status={fieldStatus?.playoff_probability as FieldStatus | undefined} />}
          />
        </div>
      </div>

      {/* Rating history */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Rating Trajectory ({data.season})
        </div>
        <Spark
          data={data.rating_history?.map((p) => p.rating) ?? []}
          width={480}
          height={60}
        />
      </div>

      {/* Recent results */}
      <div className="hm-card" style={{ padding: 24 }}>
        <div className="upper dim" style={{ fontSize: 10, marginBottom: 12 }}>
          Recent Results
        </div>
        <RecentResultsStrip results={data.recent_results} />
      </div>

      {/* Scaffolded cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <ComingSoonCard
          title="Top Players"
          status={fieldStatus?.top_players as FieldStatus | undefined}
        />
        <ComingSoonCard
          title="Situational Splits"
          status={fieldStatus?.situational_splits as FieldStatus | undefined}
        />
      </div>
    </div>
  );
}

function ProfileCell({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div style={{ minWidth: 100 }}>
      <div className="upper dim2" style={{ fontSize: 10, marginBottom: 6 }}>
        {label}
      </div>
      <div className="mono tnum" style={{ fontSize: 14 }}>
        {value}
      </div>
    </div>
  );
}

function InlineFieldStatus({ status }: { status: FieldStatus | undefined }) {
  if (!status) return <span className="mono tnum dim2">—</span>;
  if (status === "pending") return <PendingField />;
  return <BlockedField blocker={status.blocker} roadmap={status.roadmap} />;
}

function ComingSoonCard({
  title,
  status,
}: {
  title: string;
  status: FieldStatus | undefined;
}) {
  return (
    <div className="hm-card" style={{ padding: 20 }}>
      <div
        className="upper dim"
        style={{
          fontSize: 10,
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span>{title}</span>
        {status === "pending" && <PendingField placeholder="" />}
        {status && status !== "pending" && (
          <BlockedField
            blocker={status.blocker}
            roadmap={status.roadmap}
            placeholder=""
          />
        )}
      </div>
      <div
        style={{
          padding: 20,
          textAlign: "center",
          color: "var(--ink-4)",
          fontSize: 12,
        }}
      >
        Not yet available
      </div>
    </div>
  );
}
