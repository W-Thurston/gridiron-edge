import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FieldValue } from "./FieldValue";
import { TestWrapper } from "../../test/testWrapper";

describe("FieldValue", () => {
  it("renders the value when populated", () => {
    render(
      <TestWrapper>
        <FieldValue value="hello" />
      </TestWrapper>,
    );
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders em dash for null with no status", () => {
    render(
      <TestWrapper>
        <FieldValue value={null} />
      </TestWrapper>,
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders pending badge for null with pending status", () => {
    render(
      <TestWrapper>
        <FieldValue value={null} status="pending" />
      </TestWrapper>,
    );
    const badge = screen.getByTitle("Coming soon");
    expect(badge).toBeInTheDocument();
  });

  it("renders blocked badge for null with blocked status", () => {
    render(
      <TestWrapper>
        <FieldValue
          value={null}
          status={{
            status: "blocked",
            blocker: "test_slug",
            roadmap: "future capability",
          }}
        />
      </TestWrapper>,
    );
    expect(screen.getByTitle(/test_slug.*future capability/)).toBeInTheDocument();
  });
});
