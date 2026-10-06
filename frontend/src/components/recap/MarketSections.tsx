import type { components } from "../../api/schema";
import {
  recapCloseoutStatusLabel,
  recapClvKindLabel,
  recapEdgeStatusMessage,
  recapEdgeStatusShort,
  recapRecommendationStateLabel,
} from "../field-status/recapStatus";
import { SectionHead } from "../performance/SectionHead";
import { marketLabel, money, record, roi, signedMoney } from "./recapFormat";

type Edges = components["schemas"]["RecapEdges"];
type Chain = components["schemas"]["RecapProductionChain"];
type Family = components["schemas"]["RecapCandidateFamily"];
type Receipt = components["schemas"]["RecapEdgeReceipt"];
type Wagers = components["schemas"]["RecapWagers"];
type WagerSummary = components["schemas"]["RecapWagerSummary"];

/** What the week published as edges: the production chain, or why there is none. */
export function EdgesSection({ edges }: { edges: Edges }) {
  const published = edges.receipts.some((receipt) => receipt.outcome === "done");
  return (
    <section className="hm-card pf-card" data-recap-section="edges" aria-labelledby="recap-edges">
      <SectionHead eyebrow="Published edges" title="Edges and recommendations" id="recap-edges" />
      <div data-edge-status={edges.status}>
        <span className="pf-badge pf-badge-shadow" style={{ background: "var(--bg-3)", color: "var(--ink-2)" }}>{recapEdgeStatusShort(edges.status)}</span>
        <div className="pf-muted" style={{ fontSize: 12, marginTop: 8, lineHeight: 1.5 }}>
          {recapEdgeStatusMessage(edges.status)}.
          {edges.status !== "production_chain" && published &&
            " Earlier runs listed below published edges; their rows were not kept."}
        </div>
      </div>
      {edges.chain && <ChainSummary chain={edges.chain} />}
      <ReceiptTable receipts={edges.receipts} through={edges.receipts_through} />
    </section>
  );
}

function ChainSummary({ chain }: { chain: Chain }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }} data-production-chain>
      <div className="mono dim" style={{ fontSize: 11 }}>
        Issuance {chain.issuance_id.slice(0, 12)} · issued {formatTimestamp(chain.issued_at)} · {chain.rows} rows:{" "}
        {chain.candidates} candidates, {chain.not_candidates} not candidates, {chain.unavailable} unavailable
      </div>
      <div className="mono dim" style={{ fontSize: 11 }} data-recommendation-states>
        Recommendations:{" "}
        {chain.recommendations.length === 0
          ? "no candidate rows"
          : chain.recommendations
              .map((item) => `${item.count} ${recapRecommendationStateLabel(item.status)}`)
              .join(" · ")}
        {chain.evaluation_ids.length > 0 &&
          ` (evaluation ${chain.evaluation_ids.map((id) => id.slice(0, 8)).join(", ")})`}
      </div>
      <div className="pf-note">
        Candidates were never recommendations: closing-line value and outcomes only, no dollars. Rows repeat a side
        across books and snapshots, so rows are correlated.
      </div>
      <div className="market-table-scroll" role="region" aria-label="Candidates by market" tabIndex={0} style={{ marginTop: 6 }}>
        <table className="recap-table mono tnum">
          <caption className="visually-hidden">Candidate rows by market: closeout, closing-line value and graded outcomes</caption>
          <thead>
            <tr>
              <th scope="col">Market</th>
              <th scope="col">Rows</th>
              <th scope="col">Sides</th>
              <th scope="col">Closeout</th>
              <th scope="col">Median CLV</th>
              <th scope="col">CLV &gt; 0</th>
              <th scope="col">Won-lost-push</th>
            </tr>
          </thead>
          <tbody>
            {chain.families.map((family) => (
              <FamilyRow key={family.market} family={family} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FamilyRow({ family }: { family: Family }) {
  if (family.candidates === 0) {
    return (
      <tr data-candidate-market={family.market}>
        <th scope="row">{marketLabel(family.market)}</th>
        <td colSpan={6}>{unavailable("no_candidates", "No candidate rows")}</td>
      </tr>
    );
  }
  return (
    <tr data-candidate-market={family.market}>
      <th scope="row">{marketLabel(family.market)}</th>
      <td>{family.candidates}</td>
      <td>{family.sides}</td>
      <td>
        {family.closeout.map((item) => (
          <div key={item.status} data-closeout-status={item.status} style={{ fontSize: 10.5 }}>
            {item.count} {recapCloseoutStatusLabel(item.status)}
          </div>
        ))}
      </td>
      <td>
        {family.median_clv != null && family.clv_kind
          ? <>
              {signed(family.median_clv, family.clv_kind === "moneyline_price" ? 3 : 1)}
              <div className="pf-sub">{recapClvKindLabel(family.clv_kind)} · {family.clv_count} closed</div>
            </>
          : unavailable("clv_unavailable", "No candidate closed")}
      </td>
      <td>{family.share_positive_clv != null ? percent(family.share_positive_clv) : "—"}</td>
      <td>
        {family.wins}-{family.losses}-{family.pushes}
        {family.outcome_unavailable > 0 && (
          <div className="pf-sub">{family.outcome_unavailable} not graded</div>
        )}
      </td>
    </tr>
  );
}

function ReceiptTable({ receipts, through }: { receipts: Receipt[]; through: string }) {
  return (
    <div>
      <div className="pf-eyebrow">Edge checks through the last kickoff ({formatTimestamp(through)})</div>
      {receipts.length === 0
        ? <div style={{ marginTop: 8, fontSize: 12 }}>{unavailable("no_receipts", "No weekly-cycle edge check before the last kickoff")}</div>
        : (
          <div className="market-table-scroll" role="region" aria-label="Edge checks" tabIndex={0} style={{ marginTop: 8 }}>
            <table className="recap-table mono">
              <caption className="visually-hidden">Every weekly-cycle edge check decided at or before the week's last kickoff</caption>
              <thead>
                <tr>
                  <th scope="col">Checked</th>
                  <th scope="col">Outcome</th>
                  <th scope="col">Detail</th>
                  <th scope="col">Quotes fetched</th>
                </tr>
              </thead>
              <tbody>
                {receipts.map((receipt) => (
                  <tr key={receipt.decided_at} data-receipt-outcome={receipt.outcome}>
                    <th scope="row">{formatTimestamp(receipt.decided_at)}</th>
                    <td>{receipt.outcome === "done" ? "published" : receipt.outcome}</td>
                    <td>{receipt.detail}</td>
                    <td>{receipt.market_fetched_at ? formatTimestamp(receipt.market_fetched_at) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
    </div>
  );
}

/** The week's recorded wagers: a headline row, then each market type. */
export function WagersSection({ wagers }: { wagers: Wagers }) {
  const { total } = wagers;
  return (
    <section className="hm-card pf-card" data-recap-section="wagers" aria-labelledby="recap-wagers">
      <SectionHead eyebrow="Your bets" title="Recorded wagers" id="recap-wagers" />
      {total.bets === 0
        ? <div style={{ fontSize: 12 }}>{unavailable("no_wagers", "No recorded wagers on this week's games")}</div>
        : (
          <>
            <WagerTiles summary={total} />
            <WagerNotes summary={total} />
            <div className="market-table-scroll" role="region" aria-label="Wagers by market" tabIndex={0}>
              <table className="recap-table mono tnum">
                <caption className="visually-hidden">Recorded wagers by market type</caption>
                <thead>
                  <tr>
                    <th scope="col">Market</th>
                    <th scope="col">Bets</th>
                    <th scope="col">Won-lost-push</th>
                    <th scope="col">Staked</th>
                    <th scope="col">P&amp;L</th>
                    <th scope="col">ROI</th>
                    <th scope="col">CLV</th>
                  </tr>
                </thead>
                <tbody>
                  {wagers.markets.map((market) => (
                    <tr key={market.market_type} data-wager-market={market.market_type}>
                      <th scope="row">{marketLabel(market.market_type)}</th>
                      <td>{market.summary.bets}</td>
                      <td>{record(market.summary)}</td>
                      <td>{money(market.summary.staked)}</td>
                      <td>{signedMoney(market.summary.pnl)}</td>
                      <td>{roi(market.summary.roi_pct)}</td>
                      <td>
                        {market.median_clv != null && market.clv_kind
                          ? `${signed(market.median_clv, 3)} median (${market.summary.clv_available})`
                          : unavailable("clv_unavailable", "CLV unavailable")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
    </section>
  );
}

const TILES: [string, (summary: WagerSummary) => string][] = [
  ["Bets", (summary) => String(summary.bets)],
  ["Won-lost-push", record],
  ["Staked", (summary) => money(summary.staked)],
  ["P&L", (summary) => signedMoney(summary.pnl)],
  ["ROI", (summary) => roi(summary.roi_pct)],
];

function WagerTiles({ summary }: { summary: WagerSummary }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: 12 }} aria-label="Recorded wager totals">
      {TILES.map(([label, value]) => (
        <div key={label} data-wager-tile={label}>
          <div className="pf-stat-label">{label}</div>
          <div className="mono tnum" style={{ fontSize: 20, fontWeight: 500, marginTop: 4 }}>{value(summary)}</div>
        </div>
      ))}
    </div>
  );
}

/** Everything the headline leaves out, stated rather than hidden. */
function WagerNotes({ summary }: { summary: WagerSummary }) {
  const notes: [string, string][] = [];
  if (summary.unsettled > 0) {
    notes.push(["unsettled", `${summary.unsettled} still open although the game has finished; ROI covers settled bets only`]);
  }
  if (summary.clv_unavailable > 0) {
    notes.push(["clv_unavailable", `CLV unavailable for ${summary.clv_unavailable}: no quote reference was recorded with the bet, or the market has no single quote`]);
  }
  if (summary.bonus_funded > 0) notes.push(["bonus_funded", `${summary.bonus_funded} bonus-funded`]);
  notes.push(["linked", summary.linked_recommendations > 0
    ? `${summary.linked_recommendations} linked to a recommendation`
    : "None linked to a recommendation"]);
  return (
    <div className="mono dim" style={{ fontSize: 10.5, display: "flex", flexDirection: "column", gap: 2 }}>
      {notes.map(([key, text]) => <div key={key} data-wager-note={key}>{text}</div>)}
    </div>
  );
}

function signed(value: number, digits: number): string { return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`; }
function percent(value: number): string { return `${(value * 100).toFixed(1)}%`; }
function formatTimestamp(value: string): string { return new Date(value).toLocaleString(); }

function unavailable(status: string, message: string) {
  return (
    <span data-weekly-status={status} data-weekly-value-state="unavailable" title={message} style={{ color: "var(--ink-4)" }}>
      {message}
    </span>
  );
}
