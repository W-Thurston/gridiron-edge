import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SortableHeader } from "./SortableHeader";

function renderHeader(
  props: Partial<React.ComponentProps<typeof SortableHeader>> = {},
) {
  const onClick = props.onClick ?? vi.fn();

  render(
    <table>
      <thead>
        <tr>
          <SortableHeader
            label="Win SB"
            active={false}
            direction="desc"
            onClick={onClick}
            {...props}
          />
        </tr>
      </thead>
    </table>,
  );

  return { onClick };
}

describe("SortableHeader", () => {
  it("renders an inactive sortable column header", () => {
    renderHeader();

    expect(screen.getByRole("columnheader")).toHaveAttribute(
      "aria-sort",
      "none",
    );
    expect(
      screen.getByRole("button", { name: "Sort by Win SB" }),
    ).toBeInTheDocument();
  });

  it("exposes ascending state", () => {
    renderHeader({
      active: true,
      direction: "asc",
    });

    expect(screen.getByRole("columnheader")).toHaveAttribute(
      "aria-sort",
      "ascending",
    );
    expect(screen.getByText("↑")).toBeInTheDocument();
  });

  it("exposes descending state", () => {
    renderHeader({
      active: true,
      direction: "desc",
    });

    expect(screen.getByRole("columnheader")).toHaveAttribute(
      "aria-sort",
      "descending",
    );
    expect(screen.getByText("↓")).toBeInTheDocument();
  });

  it("calls onClick when activated", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();

    renderHeader({ onClick });

    await user.click(
      screen.getByRole("button", { name: "Sort by Win SB" }),
    );

    expect(onClick).toHaveBeenCalledOnce();
  });

  it("supports right-aligned numeric columns", () => {
    renderHeader({ align: "right" });

    expect(screen.getByRole("columnheader")).toHaveStyle({
      textAlign: "right",
    });
  });
});
