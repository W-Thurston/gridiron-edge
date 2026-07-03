type WinProbBandProps = {
  homeWinProb: number | null | undefined;
  homeWinLo: number | null | undefined;
  homeWinHi: number | null | undefined;
};

export function WinProbBand({
  homeWinProb,
  homeWinLo,
  homeWinHi,
}: WinProbBandProps) {
  if (
    homeWinProb == null ||
    homeWinLo == null ||
    homeWinHi == null
  ) {
    return <span className="dim mono">—</span>;
  }

  const loPct = homeWinLo * 100;
  const hiPct = homeWinHi * 100;
  const pointPct = homeWinProb * 100;

  return (
    <div style={{ width: 120 }}>
      <div className="prob-band">
        <div
          className="range"
          style={{
            left: `${loPct}%`,
            right: `${100 - hiPct}%`,
          }}
        />
        <div
          className="point"
          style={{
            left: `calc(${pointPct}% - 1px)`,
          }}
        />
      </div>
    </div>
  );
}
