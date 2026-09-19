# src/gridiron_edge/cli/production_chain.py

"""CLI reporting and immutable persistence for production-chain preflight."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

# pyrefly: ignore [missing-import]
import typer

production_chain_app = typer.Typer(
    help="Assess or verify full production recommendation-chain readiness.",
    no_args_is_help=True,
)


def _render(preflight: object) -> None:
    from gridiron_edge.market.production_chain_preflight import ProductionChainPreflight

    if not isinstance(preflight, ProductionChainPreflight):
        raise TypeError("Expected ProductionChainPreflight.")
    typer.echo(f"production-chain  {preflight.season} week {preflight.week}")
    typer.echo(f"assessed at       {preflight.assessed_at.isoformat()}")
    typer.echo("")
    for family in (preflight.moneyline, preflight.spread, preflight.total):
        typer.echo(family.market.value.upper())
        for component in family.components:
            state = component.state.value.replace("_", " ").upper()
            typer.echo(f"  {component.component_id:<27} {state}")
            typer.echo(f"    {component.reason}")
        typer.echo("")
    typer.echo("ALL FAMILIES PROVEN" if preflight.all_families_proven else "PROOF INCOMPLETE")


@production_chain_app.command("assess")
def assess_cmd(
    season: str = typer.Option(..., "--season"),
    week: int = typer.Option(..., "--week", min=1, max=22),
    assessed_at: str | None = typer.Option(None, "--assessed-at"),
    write: bool = typer.Option(False, "--write/--no-write"),
    require_ready: bool = typer.Option(False, "--require-ready/--allow-incomplete"),
) -> None:
    """Assess current evidence once and optionally persist that exact assessment."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.production_chain_preflight import assess_production_chain_preflight
    from gridiron_edge.market.production_chain_preflight_store import (
        production_chain_preflight_id,
        write_production_chain_preflight,
    )

    try:
        timestamp = (
            datetime.now(UTC) if assessed_at is None else datetime.fromisoformat(assessed_at)
        )
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError("assessed-at must be timezone-aware UTC.")
        settings = get_settings()
        preflight = assess_production_chain_preflight(
            repo=settings.repo_root,
            season=season,
            week=week,
            assessed_at=timestamp,
        )
        _render(preflight)
        typer.echo(f"preflight id      {production_chain_preflight_id(preflight)}")
        if write:
            path = write_production_chain_preflight(preflight, repo=settings.repo_root)
            typer.echo(f"stored            {path}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if require_ready and not preflight.all_families_proven:
        raise typer.Exit(code=1)


def _utc_option(value: str, *, label: str) -> datetime:
    """Parse one explicit timezone-aware UTC CLI timestamp."""
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != UTC.utcoffset(result):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return result


def _render_candidate_issuance(issuance: object) -> None:
    """Render immutable candidate issuance counts independently by family."""
    from collections import Counter

    from gridiron_edge.market.candidate_issuance import CandidateIssuance

    if not isinstance(issuance, CandidateIssuance):
        raise TypeError("Expected CandidateIssuance.")

    typer.echo(f"candidate issuance  {issuance.season} week {issuance.week}")
    typer.echo(f"product id          {issuance.product_id}")
    typer.echo(f"product run         {issuance.product_run_id}")
    typer.echo(f"evaluated at        {issuance.evaluated_at.isoformat()}")
    typer.echo(f"observations        {len(issuance.rows)}")
    typer.echo("")

    for market in ("moneyline", "spread", "total"):
        counts = Counter(row.state.value for row in issuance.rows if row.market == market)
        typer.echo(market.upper())
        typer.echo(f"  candidate         {counts['candidate']}")
        typer.echo(f"  not candidate     {counts['not_candidate']}")
        typer.echo(f"  unavailable       {counts['unavailable']}")
        typer.echo("")

    typer.echo(f"issuance id         {issuance.issuance_id}")


def _render_governance(version: object) -> None:
    from gridiron_edge.market.recommendation_governance import (
        RecommendationGovernanceVersion,
    )

    if not isinstance(version, RecommendationGovernanceVersion):
        raise TypeError("Expected RecommendationGovernanceVersion.")
    governance = version.governance
    typer.echo("recommendation governance")
    typer.echo(f"governance id       {version.governance_id}")
    typer.echo(f"created at          {version.created_at.isoformat()}")
    typer.echo(f"fractional Kelly    {governance.fractional_kelly_multiplier}")
    typer.echo(f"minimum stake       {governance.minimum_actionable_stake}")
    typer.echo(f"stake increment     {governance.stake_increment}")
    typer.echo(f"stake rounding      {governance.stake_rounding.value}")
    typer.echo(f"candidate exposure  {governance.maximum_candidate_bankroll_fraction}")
    typer.echo(f"game exposure       {governance.maximum_game_bankroll_fraction}")
    typer.echo(f"portfolio exposure  {governance.maximum_portfolio_bankroll_fraction}")
    typer.echo(f"prohibit opposing   {governance.prohibit_opposing_positions}")
    typer.echo(f"correlation needed  {governance.correlation_check_mandatory}")
    typer.echo(f"eligible statuses   {', '.join(governance.exposure_eligible_statuses)}")


@production_chain_app.command("create-governance")
def create_governance_cmd(
    *,
    created_at: str = typer.Option(..., "--created-at"),
    fractional_kelly_multiplier: float = typer.Option(..., "--fractional-kelly-multiplier"),
    minimum_actionable_stake: float = typer.Option(..., "--minimum-actionable-stake"),
    stake_increment: float = typer.Option(..., "--stake-increment"),
    stake_rounding: str = typer.Option(..., "--stake-rounding"),
    maximum_candidate_bankroll_fraction: float = typer.Option(
        ..., "--maximum-candidate-bankroll-fraction"
    ),
    maximum_game_bankroll_fraction: float = typer.Option(..., "--maximum-game-bankroll-fraction"),
    maximum_portfolio_bankroll_fraction: float = typer.Option(
        ..., "--maximum-portfolio-bankroll-fraction"
    ),
    prohibit_opposing_positions: bool = typer.Option(
        ..., "--prohibit-opposing-positions/--allow-opposing-positions"
    ),
    correlation_check_mandatory: bool = typer.Option(
        ..., "--correlation-check-mandatory/--correlation-check-optional"
    ),
    exposure_eligible_status: Annotated[
        list[str] | None,
        typer.Option(
            "--exposure-eligible-status",
            help=("Wager status included in exposure accounting. Repeat for multiple statuses."),
        ),
    ] = None,
    write: bool = typer.Option(False, "--write/--no-write"),
) -> None:
    """Create governance from explicit values and optionally persist it."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.recommendation_governance import (
        create_recommendation_governance,
    )
    from gridiron_edge.market.recommendation_governance_store import (
        write_recommendation_governance,
    )
    from gridiron_edge.market.recommendation_policy import StakeRoundingMode

    try:
        if exposure_eligible_status is None:
            raise ValueError("At least one exposure-eligible status is required.")
        statuses = tuple(sorted(set(exposure_eligible_status)))
        if len(statuses) != len(exposure_eligible_status):
            raise ValueError("Exposure-eligible statuses must be unique.")
        version = create_recommendation_governance(
            created_at=_utc_option(created_at, label="created-at"),
            fractional_kelly_multiplier=fractional_kelly_multiplier,
            minimum_actionable_stake=minimum_actionable_stake,
            stake_increment=stake_increment,
            stake_rounding=StakeRoundingMode(stake_rounding),
            maximum_candidate_bankroll_fraction=maximum_candidate_bankroll_fraction,
            maximum_game_bankroll_fraction=maximum_game_bankroll_fraction,
            maximum_portfolio_bankroll_fraction=maximum_portfolio_bankroll_fraction,
            prohibit_opposing_positions=prohibit_opposing_positions,
            correlation_check_mandatory=correlation_check_mandatory,
            exposure_eligible_statuses=statuses,
        )
        _render_governance(version)
        if write:
            path = write_recommendation_governance(
                version,
                repo=get_settings().repo_root,
            )
            typer.echo(f"stored              {path}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@production_chain_app.command("verify-governance")
def verify_governance_cmd(
    governance_id: str = typer.Option(..., "--governance-id"),
) -> None:
    """Read and display one exact stored governance version."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.recommendation_governance import (
        RECOMMENDATION_GOVERNANCE_SCHEMA_VERSION,
    )
    from gridiron_edge.market.recommendation_governance_store import (
        read_recommendation_governance,
        recommendation_governance_path,
    )

    try:
        settings = get_settings()
        path = recommendation_governance_path(
            RECOMMENDATION_GOVERNANCE_SCHEMA_VERSION,
            governance_id,
            repo=settings.repo_root,
        )
        version = read_recommendation_governance(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _render_governance(version)


def _render_policy(policy: object) -> None:
    from gridiron_edge.market.recommendation_policy import RecommendationPolicy

    if not isinstance(policy, RecommendationPolicy):
        raise TypeError("Expected RecommendationPolicy.")
    typer.echo("recommendation policy")
    typer.echo(f"policy id           {policy.policy_id}")
    typer.echo(f"created at          {policy.created_at.isoformat()}")
    typer.echo(f"governance          {policy.governance_fingerprint}")
    typer.echo("")
    for family in (policy.moneyline, policy.spread, policy.total):
        typer.echo(family.market.upper())
        typer.echo(f"  status            {family.status.value}")
        typer.echo(f"  reason            {family.reason.value}")
        typer.echo(f"  candidates        {family.candidate_count}")
        typer.echo(f"  outcomes          {family.outcome_available_count}")
        typer.echo(f"  closeouts         {family.clv_available_count}")
        typer.echo(f"  returns           {family.return_available_count}")
        typer.echo("")


def _render_recommendation_evaluation(evaluation: object) -> None:
    from collections import Counter

    from gridiron_edge.market.recommended_bet_result import RecommendedBetEvaluation

    if not isinstance(evaluation, RecommendedBetEvaluation):
        raise TypeError("Expected RecommendedBetEvaluation.")
    counts = Counter(result.result_state.value for result in evaluation.results)
    typer.echo("recommended-bet evaluation")
    typer.echo(f"evaluation id       {evaluation.evaluation_id}")
    typer.echo(f"issuance id         {evaluation.issuance_id}")
    typer.echo(f"policy id           {evaluation.policy_id}")
    typer.echo(f"evaluated at        {evaluation.evaluated_at.isoformat()}")
    typer.echo(f"results             {len(evaluation.results)}")
    for state in ("qualified", "recommended", "failed", "unavailable", "conflicting"):
        typer.echo(f"  {state:<18} {counts[state]}")


@production_chain_app.command("derive-policy")
def derive_policy_cmd(
    issuance_id: str = typer.Option(..., "--issuance-id"),
    governance_id: str = typer.Option(..., "--governance-id"),
    created_at: str = typer.Option(..., "--created-at"),
    write: bool = typer.Option(False, "--write/--no-write"),
) -> None:
    """Derive an exact policy from persisted issuance and empirical evidence."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_games
    from gridiron_edge.ingest.odds.store import load_odds_ledger
    from gridiron_edge.market.candidate_issuance_store import (
        candidate_issuance_path,
        read_candidate_issuance,
    )
    from gridiron_edge.market.history_boundaries import select_quote_history_boundaries
    from gridiron_edge.market.market_family_evaluation import evaluate_market_families
    from gridiron_edge.market.recommendation_governance import (
        RECOMMENDATION_GOVERNANCE_SCHEMA_VERSION,
    )
    from gridiron_edge.market.recommendation_governance_store import (
        read_recommendation_governance,
        recommendation_governance_path,
    )
    from gridiron_edge.market.recommendation_policy import derive_recommendation_policy
    from gridiron_edge.market.recommendation_policy_store import write_recommendation_policy

    try:
        settings = get_settings()
        issuance = read_candidate_issuance(
            candidate_issuance_path(issuance_id, repo=settings.repo_root)
        )
        governance = read_recommendation_governance(
            recommendation_governance_path(
                RECOMMENDATION_GOVERNANCE_SCHEMA_VERSION,
                governance_id,
                repo=settings.repo_root,
            )
        )
        history = load_odds_ledger(
            season=issuance.season,
            week=issuance.week,
            repo=settings.repo_root,
        )
        empirical = evaluate_market_families(
            issuance=issuance,
            closeouts=(),
            games=load_games(settings.repo_root),
            history_boundaries=select_quote_history_boundaries(history),
        )
        policy = derive_recommendation_policy(
            evaluation=empirical,
            governance=governance.governance,
            created_at=_utc_option(created_at, label="created-at"),
        )
        _render_policy(policy)
        if write:
            path = write_recommendation_policy(policy, repo=settings.repo_root)
            typer.echo(f"stored              {path}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@production_chain_app.command("evaluate-recommendations")
def evaluate_recommendations_cmd(
    issuance_id: str = typer.Option(..., "--issuance-id"),
    policy_id: str = typer.Option(..., "--policy-id"),
    decision_at: str = typer.Option(..., "--decision-at"),
    write: bool = typer.Option(False, "--write/--no-write"),
) -> None:
    """Evaluate exact persisted candidates against one exact persisted policy."""
    from gridiron_edge.betting.bankroll import bankroll_snapshot_as_of
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.candidate_issuance_store import (
        candidate_issuance_path,
        read_candidate_issuance,
    )
    from gridiron_edge.market.recommendation_policy import (
        RECOMMENDATION_POLICY_SCHEMA_VERSION,
        BankrollBasis,
        portfolio_exposure_snapshot,
    )
    from gridiron_edge.market.recommendation_policy_store import (
        read_recommendation_policy,
        recommendation_policy_path,
    )
    from gridiron_edge.market.recommended_bet_result import (
        evaluate_recommendation_issuance,
    )
    from gridiron_edge.market.recommended_bet_result_store import (
        write_recommended_bet_evaluation,
    )

    try:
        settings = get_settings()
        issuance = read_candidate_issuance(
            candidate_issuance_path(issuance_id, repo=settings.repo_root)
        )
        policy = read_recommendation_policy(
            recommendation_policy_path(
                RECOMMENDATION_POLICY_SCHEMA_VERSION, policy_id, repo=settings.repo_root
            )
        )
        decision = _utc_option(decision_at, label="decision-at")
        snapshot = bankroll_snapshot_as_of(decision, repo=settings.repo_root)
        bankroll = (
            None
            if snapshot is None
            else BankrollBasis(
                amount=snapshot.amount,
                observed_at=snapshot.observed_at,
                source_kind=snapshot.source_kind,
                source_id=snapshot.source_id,
            )
        )
        evaluation = evaluate_recommendation_issuance(
            policy=policy,
            issuance=issuance,
            decision_at=decision,
            bankroll=bankroll,
            portfolio=portfolio_exposure_snapshot(observed_at=decision, rows=()),
        )
        _render_recommendation_evaluation(evaluation)
        if write:
            path = write_recommended_bet_evaluation(
                evaluation,
                repo=settings.repo_root,
            )
            typer.echo(f"stored              {path}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@production_chain_app.command("issue-candidates")
def issue_candidates_cmd(
    season: str = typer.Option(..., "--season"),
    week: int = typer.Option(..., "--week", min=1, max=22),
    evaluated_at: str = typer.Option(..., "--evaluated-at"),
    write: bool = typer.Option(False, "--write/--no-write"),
) -> None:
    """Issue immutable candidates from selected forecasts and quote history."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_current_weekly_product
    from gridiron_edge.evaluation.forecast_evidence_disposition import (
        require_operational_weekly_product,
    )
    from gridiron_edge.evaluation.forecast_evidence_disposition_store import (
        list_forecast_evidence_dispositions,
    )
    from gridiron_edge.evaluation.forecast_store import load_forecast_events
    from gridiron_edge.ingest.odds.as_known import as_known_at
    from gridiron_edge.ingest.odds.store import load_odds_ledger
    from gridiron_edge.market.candidate_issuance import issue_pregame_candidates
    from gridiron_edge.market.candidate_issuance_store import write_candidate_issuance

    try:
        timestamp = _utc_option(evaluated_at, label="evaluated-at")
        settings = get_settings()
        product = load_current_weekly_product(
            settings.repo_root,
            season=season,
            week=week,
        )
        if "product_id" not in product.columns:
            raise ValueError("Selected weekly product is missing required column: product_id")

        product_ids = tuple(sorted(product["product_id"].dropna().astype(str).str.strip().unique()))
        if len(product_ids) != 1 or product_ids == ("",):
            raise ValueError("Selected weekly product must contain one nonempty product_id.")

        dispositions = list_forecast_evidence_dispositions(
            season=season,
            week=week,
            product_id=product_ids[0],
            repo=settings.repo_root,
        )
        require_operational_weekly_product(
            product,
            dispositions,
        )
        run_ids = product["product_run_id"].dropna().astype(str).unique().tolist()
        if len(run_ids) != 1:
            raise ValueError("Selected weekly product must contain one product run identity.")
        events = load_forecast_events(
            season=season,
            week=week,
            run_id=run_ids[0],
            repo=settings.repo_root,
        )
        quotes = load_odds_ledger(
            season=season,
            week=week,
            repo=settings.repo_root,
        )
        visible_quotes = as_known_at(quotes, timestamp)
        issuance = issue_pregame_candidates(
            product=product,
            forecast_events=events,
            quotes=visible_quotes,
            evaluated_at=timestamp,
        )

        _render_candidate_issuance(issuance)
        if write:
            path = write_candidate_issuance(issuance, repo=settings.repo_root)
            typer.echo(f"stored              {path}")
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@production_chain_app.command("verify")
def verify_cmd(
    preflight_id: str = typer.Option(..., "--preflight-id"),
) -> None:
    """Read and render one exact stored assessment without reassessing evidence."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.production_chain_preflight_store import (
        PREFLIGHT_STORE_SCHEMA_VERSION,
        production_chain_preflight_path,
        read_production_chain_preflight,
    )

    try:
        settings = get_settings()
        path = production_chain_preflight_path(
            PREFLIGHT_STORE_SCHEMA_VERSION,
            preflight_id,
            repo=settings.repo_root,
        )
        preflight = read_production_chain_preflight(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _render(preflight)
    typer.echo(f"preflight id      {preflight_id}")
