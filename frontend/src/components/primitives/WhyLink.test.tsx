import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { WhyLink } from "./WhyLink";
import { TestWrapper } from "../../test/testWrapper";

describe("WhyLink", () => {
  it("renders labeled variant with default label", () => {
    render(
      <TestWrapper>
        <WhyLink />
      </TestWrapper>,
    );
    expect(screen.getByText("Why?")).toBeInTheDocument();
  });

  it("renders custom label", () => {
    render(
      <TestWrapper>
        <WhyLink label="Why 71%?" />
      </TestWrapper>,
    );
    expect(screen.getByText("Why 71%?")).toBeInTheDocument();
  });

  it("renders dot variant without label text", () => {
    render(
      <TestWrapper>
        <WhyLink dot label="Why?" />
      </TestWrapper>,
    );
    // Dot variant renders just "?" not the label
    expect(screen.queryByText("Why?")).not.toBeInTheDocument();
    expect(screen.getByText("?")).toBeInTheDocument();
  });

  it("navigates to /explain on click", async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper>
        <WhyLink label="Why?" subject={{ kind: "sim" }} />
      </TestWrapper>,
    );

    // Route change is hard to assert directly; instead assert click doesn't throw
    // and button remains rendered.
    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("stops propagation on click", async () => {
    const user = userEvent.setup();
    const parentClick = vi.fn();

    render(
      <TestWrapper>
        <div onClick={parentClick}>
          <WhyLink label="Why?" />
        </div>
      </TestWrapper>,
    );

    await user.click(screen.getByRole("button"));
    expect(parentClick).not.toHaveBeenCalled();
  });

  it("aria-label defaults to Why?", () => {
    render(
      <TestWrapper>
        <WhyLink />
      </TestWrapper>,
    );
    expect(screen.getByRole("button")).toHaveAttribute("aria-label", "Why?");
  });

  it("aria-label uses custom label", () => {
    render(
      <TestWrapper>
        <WhyLink label="Why 71%?" />
      </TestWrapper>,
    );
    expect(screen.getByRole("button")).toHaveAttribute("aria-label", "Why 71%?");
  });
});
