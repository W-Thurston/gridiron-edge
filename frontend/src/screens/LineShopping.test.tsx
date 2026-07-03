import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LineShopping } from "./LineShopping";

describe("LineShopping", () => {
  it("renders as a blocked screen with correct blocker context", () => {
    render(<LineShopping />);
    expect(screen.getByText("Line Shopping")).toBeInTheDocument();
    expect(screen.getByText(/multi_book_ingest/)).toBeInTheDocument();
    expect(screen.getByText(/W7/)).toBeInTheDocument();
  });
});
