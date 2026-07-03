import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FieldValue } from "./FieldValue";

describe("FieldValue", () => {
  it("renders the value when populated", () => {
    render(<FieldValue value="hello" />);
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders em dash for null with no status", () => {
    render(<FieldValue value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders pending badge for null with pending status", () => {
    render(<FieldValue value={null} status="pending" />);
    const badge = screen.getByTitle("Coming soon");
    expect(badge).toBeInTheDocument();
  });

  it("renders blocked badge for null with blocked status", () => {
    render(
      <FieldValue
        value={null}
        status={{
          status: "blocked",
          blocker: "test_slug",
          roadmap: "W99",
        }}
      />,
    );
    expect(screen.getByTitle(/test_slug.*W99/)).toBeInTheDocument();
  });
});
