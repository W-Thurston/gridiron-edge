import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LineShopping } from "./LineShopping";
import { TestWrapper } from "../test/testWrapper";

describe("LineShopping", () => {
  it("renders as a blocked screen with correct blocker context", () => {
    render(
      <TestWrapper>
        <LineShopping />
      </TestWrapper>,
    );
    expect(screen.getByText("Line Shopping")).toBeInTheDocument();
    expect(screen.getByText(/multi_book_ingest/)).toBeInTheDocument();
    expect(screen.getByText(/W7/)).toBeInTheDocument();
    expect(
      screen.getByText(
        /current game markets use the nflverse schedule source/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/currently only DraftKings/i),
    ).not.toBeInTheDocument();
  });
});
