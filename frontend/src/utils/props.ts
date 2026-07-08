/**
 * Format prop stat_type slug for display.
 *
 * "qb_pass_yards" → "Passing Yards"
 * "qb_rush_yards" → "Rushing Yards"
 * "rb_rush_yards" → "Rushing Yards"
 * "wr_rec_yards" → "Receiving Yards"
 * "te_rec_yards" → "Receiving Yards"
 *
 * Falls back to the slug itself if not recognized.
 */
export function formatStatType(statType: string): string {
  const map: Record<string, string> = {
    qb_pass_yards: "Passing Yards",
    qb_rush_yards: "Rushing Yards",
    rb_rush_yards: "Rushing Yards",
    wr_rec_yards: "Receiving Yards",
    te_rec_yards: "Receiving Yards",
  };
  return map[statType] ?? statType;
}

/**
 * Short version — for compact contexts like tables.
 *
 * "qb_pass_yards" → "Pass Yds"
 */
export function formatStatTypeShort(statType: string): string {
  const map: Record<string, string> = {
    qb_pass_yards: "Pass Yds",
    qb_rush_yards: "Rush Yds",
    rb_rush_yards: "Rush Yds",
    wr_rec_yards: "Rec Yds",
    te_rec_yards: "Rec Yds",
  };
  return map[statType] ?? statType;
}
