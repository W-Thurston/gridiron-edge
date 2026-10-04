import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarketAsOf } from "./MarketAsOf";
import { formatMarketAge, latestTimestamp, marketAge } from "./marketAge";

const freshness = {
  evaluated_at: "2026-10-04T12:00:00Z",
  max_age_seconds: 24 * 3600,
};

describe("marketAge", () => {
  it("is fresh within the limit and stale beyond it", () => {
    expect(marketAge("2026-10-04T09:00:00Z", freshness).state).toBe("fresh");
    expect(marketAge("2026-10-03T12:00:00Z", freshness).state).toBe("fresh");
    expect(marketAge("2026-10-03T11:59:00Z", freshness).state).toBe("stale");
  });

  it("uses the newest timestamp and ignores invalid ones", () => {
    expect(
      latestTimestamp(["2026-10-01T00:00:00Z", "nonsense", "2026-10-03T00:00:00Z"]),
    ).toBe("2026-10-03T00:00:00Z");
    expect(marketAge(["nonsense"], freshness).state).toBe("unknown");
    expect(marketAge([], freshness).state).toBe("unknown");
    expect(marketAge(null, freshness).state).toBe("unknown");
  });

  it("never reports a negative age", () => {
    expect(marketAge("2026-10-05T00:00:00Z", freshness).ageMs).toBe(0);
  });

  it("formats minutes, hours and days", () => {
    expect(formatMarketAge(5 * 60000)).toBe("5 min");
    expect(formatMarketAge(3.5 * 3600000)).toBe("3.5 h");
    expect(formatMarketAge(76.8 * 3600000)).toBe("3.2 days");
  });
});

describe("MarketAsOf", () => {
  it("states a fresh snapshot without a warning", () => {
    render(<MarketAsOf fetchedAt="2026-10-04T09:00:00Z" freshness={freshness} />);

    const status = screen.getByRole("note");
    expect(status).toHaveAttribute("data-market-freshness", "fresh");
    expect(status).toHaveTextContent("3.0 h old");
    expect(status).not.toHaveTextContent("withheld");
  });

  it("states that a stale snapshot withholds edges and guidance", () => {
    render(
      <MarketAsOf
        fetchedAt={["2026-09-30T12:38:29Z"]}
        freshness={freshness}
      />,
    );

    const status = screen.getByRole("note");
    expect(status).toHaveAttribute("data-market-freshness", "stale");
    expect(status).toHaveTextContent("4.0 days old");
    expect(status).toHaveTextContent("older than the 24 h limit");
    expect(status).toHaveTextContent("edges and price guidance are withheld");
  });

  it("states an unavailable snapshot time instead of hiding it", () => {
    render(<MarketAsOf fetchedAt={[]} freshness={freshness} />);

    expect(screen.getByRole("note")).toHaveAttribute(
      "data-market-freshness",
      "unknown",
    );
    expect(screen.getByText("Market snapshot time unavailable")).toBeInTheDocument();
  });
});
