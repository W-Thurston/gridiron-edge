import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ModelPerformanceRail } from "./ModelPerformanceRail";
import { TestWrapper } from "../../test/testWrapper";

vi.mock("../../api/hooks", () => ({
  usePortfolioSummary: vi.fn(),
  usePortfolioCurve: vi.fn(),
}));

import { usePortfolioCurve, usePortfolioSummary } from "../../api/hooks";

describe("ModelPerformanceRail", () => {
  it("renders header", () => {
    vi.mocked(usePortfolioSummary).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);
    vi.mocked(usePortfolioCurve).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelPerformanceRail />
      </TestWrapper>,
    );
    expect(screen.getByText("Model Performance")).toBeInTheDocument();
  });

  it("renders loading state", () => {
    vi.mocked(usePortfolioSummary).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);
    vi.mocked(usePortfolioCurve).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelPerformanceRail />
      </TestWrapper>,
    );
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders empty state when no bets", () => {
    vi.mocked(usePortfolioSummary).mockReturnValue({
      data: {
        bankroll: 0,
        total_pnl: 0,
        roi_pct: 0,
        wins: 0,
        losses: 0,
        pushes: 0,
        total_bets: 0,
        _meta: {},
      },
      isLoading: false,
      error: null,
    } as never);
    vi.mocked(usePortfolioCurve).mockReturnValue({
      data: { items: [], _meta: {} },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelPerformanceRail />
      </TestWrapper>,
    );
    expect(screen.getByText(/No performance data yet/)).toBeInTheDocument();
  });

  it("renders ROI and stats when data available", () => {
    vi.mocked(usePortfolioSummary).mockReturnValue({
      data: {
        bankroll: 10000,
        total_pnl: 500,
        roi_pct: 24.3,
        wins: 15,
        losses: 8,
        pushes: 1,
        total_bets: 24,
        _meta: {},
      },
      isLoading: false,
      error: null,
    } as never);
    vi.mocked(usePortfolioCurve).mockReturnValue({
      data: {
        items: [
            { timestamp: "2026-01-01", bankroll: 9500 },
            { timestamp: "2026-01-02", bankroll: 10000 },
            { timestamp: "2026-01-03", bankroll: 10500 },
        ],
        _meta: {},
        },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelPerformanceRail />
      </TestWrapper>,
    );
    expect(screen.getByText("+24.3%")).toBeInTheDocument();
    expect(screen.getByText("All-Time ROI")).toBeInTheDocument();
    expect(screen.getByText(/15-8-1/)).toBeInTheDocument();
    expect(screen.getByText(/24 bets/)).toBeInTheDocument();
  });

  it("renders 'Open bankroll →' button", () => {
    vi.mocked(usePortfolioSummary).mockReturnValue({
      data: {
        bankroll: 10000,
        total_pnl: 500,
        roi_pct: 24.3,
        wins: 15,
        losses: 8,
        pushes: 1,
        total_bets: 24,
        _meta: {},
      },
      isLoading: false,
      error: null,
    } as never);
    vi.mocked(usePortfolioCurve).mockReturnValue({
      data: { items: [], _meta: {} },
      isLoading: false,
      error: null,
    } as never);

    render(
      <TestWrapper>
        <ModelPerformanceRail />
      </TestWrapper>,
    );
    expect(screen.getByText("Open bankroll →")).toBeInTheDocument();
  });
});
