import requests
import pprint
import random
import itertools
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from sparklines import sparklines
import tabulate
from functools import total_ordering
from typing import Optional, Callable

from _rebbl_predictor import SeasonScore, random_score

tabulate.PRESERVE_WHITESPACE = True


class SeasonScores(defaultdict):
    def __init__(self, default_factory=None, *args):
        super().__init__(None, *args)

    def __missing__(self, key):
        score = SeasonScore(key)
        self[key] = score
        return score


def score_season(games):
    scores = SeasonScores()
    for game in games:
        scores[game["homeTeamIndex"]].add_game(game, True)
        scores[game["awayTeamIndex"]].add_game(game, False)

    return scores


def predict_games(predictor, games):
    for game in games:
        game["homeScore"] = predictor(game["homeTeamName"])
        game["awayScore"] = predictor(game["awayTeamName"])


def sum_stats(old, new):
    return {
        team: SeasonScore(
            team,
            old[team].points + new[team].points,
            old[team].tdd + new[team].tdd,
            old[team].losses + new[team].losses,
            old[team].head_to_head | new[team].head_to_head,
        )
        for team in old.keys() | new.keys()
    }


def sort_stats(stats):
    return sorted(list(stats.items()), key=lambda x: x[1], reverse=True)


def season_teams(games):
    return {game["homeTeamName"] for game in games} | {
        game["awayTeamName"] for game in games
    }


def predict_season(
    league: str,
    season: str,
    division: str,
    *,
    as_of: Optional[int] = None,
    iterations: int = 10000,
    predictor: Callable = random_score,
):
    resp = requests.get(
        f"https://rebbl.net/api/v2/division/{league}/{season}/{division}/slim"
    )

    season = resp.json()
    teams = sorted(season_teams(season))
    team_indexes = {team: index for index, team in enumerate(teams)}
    for game in season:
        game["homeTeamIndex"] = team_indexes[game["homeTeamName"]]
        game["awayTeamIndex"] = team_indexes[game["awayTeamName"]]

    played_games = [
        game
        for game in season
        if game["homeScore"] is not None
        and game["awayScore"] is not None
        and (as_of is None or game["round"] <= as_of)
    ]
    remaining_games = [
        game
        for game in season
        if game["homeScore"] is None
        or game["awayScore"] is None
        or (as_of is not None and game["round"] > as_of)
    ]

    current_stats = score_season(played_games)

    outcomes = []
    for _ in range(iterations):
        predict_games(predictor, remaining_games)
        remaining_stats = score_season(remaining_games)
        total_stats = sum_stats(current_stats, remaining_stats)
        sorted_stats = sort_stats(total_stats)
        assert sorted_stats == sorted(sorted_stats, key=lambda x: x[1], reverse=True)
        outcomes.append(sorted_stats)

    outcome_positions = (
        enumerate((team_name for team_name, _ in outcome)) for outcome in outcomes
    )
    outcome_counts = Counter(itertools.chain.from_iterable(outcome_positions))

    results_table = {
        team: [
            outcome_counts[(i, team_indexes[team])] / iterations
            for i in range(len(teams))
        ]
        for team in teams
    }
    return results_table


if __name__ == "__main__":
    league = "REBBL - REL"
    season = "season 15"
    division = "Season 15 - Division 3B"
    results_table = predict_season(league, season, division, iterations=10000, as_of=5)
    team_order = [
        team
        for (team, _) in sorted(results_table.items(), key=lambda x: x[1], reverse=True)
    ]

    print(
        tabulate.tabulate(
            [results_table[team] for team in team_order],
            showindex=team_order,
            floatfmt="2.0%",
        )
    )
