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

tabulate.PRESERVE_WHITESPACE = True


@total_ordering
class SeasonScore:
    team: str
    points: int
    tdd: int
    losses: int
    head_to_head: int

    __slots__ = ("team", "points", "tdd", "losses", "head_to_head")

    def __init__(self, team, points=0, tdd=0, losses=0, head_to_head=0):
        self.team = team
        self.points = points
        self.tdd = tdd
        self.losses = losses
        self.head_to_head = head_to_head

    def add_game(self, game, home):
        homeScore = game["homeScore"]
        awayScore = game["awayScore"]
        if home:
            if homeScore > awayScore:
                self.points += 3
                self.head_to_head |= 1 << game["awayTeamIndex"]
            elif homeScore == awayScore:
                self.points += 1
            else:
                self.losses += 1
            self.tdd += homeScore - awayScore
        else:
            if homeScore < awayScore:
                self.points += 3
                self.head_to_head |= 1 << game["awayTeamIndex"]
            elif homeScore == awayScore:
                self.points += 1
            else:
                self.losses += 1
            self.tdd += awayScore - homeScore

    def copy(self):
        return SeasonScore(
            self.team, self.points, self.tdd, self.losses, self.head_to_head.copy()
        )

    def __lt__(self, other):
        if self.points != other.points:
            return self.points < other.points
        if self.tdd != other.tdd:
            return self.tdd < other.tdd
        if self.losses != other.losses:
            return self.losses > other.losses
        return 1 << self.team & other.head_to_head

    def __eq__(self, other):
        return (
            self.points == other.points
            and self.tdd == other.tdd
            and self.losses == other.losses
            and not (1 << self.team & other.head_to_head)
            and not (1 << other.team & self.head_to_head)
        )

    def __repr__(self):
        return f"SeasonScore({self.team!r}, {self.points!r}, {self.tdd!r}, {self.losses!r}, {self.head_to_head!r})"


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
        game["homeScore"] = predictor(game["homeTeamIndex"])
        game["awayScore"] = predictor(game["awayTeamIndex"])


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


def random_score(team):
    choice = random.random()
    if choice < 0.25:
        return 0
    elif choice < 0.55:
        return 1
    elif choice < 0.85:
        return 2
    elif choice < 0.95:
        return 3
    elif choice < 0.98:
        return 4
    else:
        return 5


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
