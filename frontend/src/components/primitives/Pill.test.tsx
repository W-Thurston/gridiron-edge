import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Pill } from "./Pill";

describe("Pill", () => {
  it("renders label", () => {
    render(<Pill active={false} onClick={() => {}}>Season</Pill>);
    expect(screen.getByText("Season")).toBeInTheDocument();
  });

  it("aria-pressed reflects active state", () => {
    const { rerender } = render(
      <Pill active={true} onClick={() => {}}>Active</Pill>,
    );
    expect(screen.getByRole("button")).toHaveAttribute("aria-pressed", "true");

    rerender(<Pill active={false} onClick={() => {}}>Inactive</Pill>);
    expect(screen.getByRole("button")).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onClick when clicked", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Pill active={false} onClick={onClick}>Click me</Pill>);

    await user.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("does not call onClick when disabled", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <Pill active={false} onClick={onClick} disabled>Disabled</Pill>,
    );

    await user.click(screen.getByRole("button"));
    expect(onClick).not.toHaveBeenCalled();
  });
});
