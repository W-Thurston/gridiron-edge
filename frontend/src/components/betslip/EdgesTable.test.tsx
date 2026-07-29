import {
  render,
  screen,
} from "@testing-library/react";
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { EdgesTable } from "./EdgesTable";
import { TestWrapper } from "../../test/testWrapper";
import { useEdges } from "../../api/hooks";

vi.mock("../../api/hooks", () => ({
  useEdges: vi.fn(),
}));

vi.mock(
  "../../api/team_metadata_hook",
  () => ({
    useTeamByAbbr: vi.fn(
      () => null,
    ),
  }),
);

function mockEmptyEdges() {
  vi.mocked(
    useEdges,
  ).mockReturnValue({
    data: {
      items: [],
      total: 0,
      bankroll: null,
      kelly_multiplier: 0.25,
      _meta: null,
    },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  } as never);
}

describe("EdgesTable sizing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockEmptyEdges();
  });

  it(
    "passes an explicit bankroll and multiplier",
    () => {
      render(
        <TestWrapper>
          <EdgesTable
            bankroll={2500}
            kellyMultiplier={0.1}
          />
        </TestWrapper>,
      );

      expect(
        useEdges,
      ).toHaveBeenCalledWith({
        bankroll: 2500,
        kelly_multiplier: 0.1,
      });

      expect(
        screen.getByText(
          "No edges available.",
        ),
      ).toBeInTheDocument();
    },
  );

  it(
    "omits bankroll when the sizing basis is unavailable",
    () => {
      render(
        <TestWrapper>
          <EdgesTable
            bankroll={null}
            kellyMultiplier={0.25}
          />
        </TestWrapper>,
      );

      expect(
        useEdges,
      ).toHaveBeenCalledWith({
        kelly_multiplier: 0.25,
      });
    },
  );

  it(
    "preserves zero as an explicit bankroll",
    () => {
      render(
        <TestWrapper>
          <EdgesTable
            bankroll={0}
            kellyMultiplier={0.25}
          />
        </TestWrapper>,
      );

      expect(
        useEdges,
      ).toHaveBeenCalledWith({
        bankroll: 0,
        kelly_multiplier: 0.25,
      });
    },
  );
});
