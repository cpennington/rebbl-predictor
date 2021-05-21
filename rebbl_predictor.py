#!/usr/bin/env python

import itertools
import pprint
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Callable, Optional

import challonge
import dataset
import requests
import tabulate
import typer
from numpy.random import default_rng
from sparklines import sparklines

from _rebbl_predictor import Game, SeasonScore, random_score

predictions = dataset.connect("sqlite:///predictions.db")
rebbl_stats = predictions["rebbl_stats"]

Competition = Enum(
    "Competition",
    {
        row["competition"]: row["competition"]
        for row in rebbl_stats.distinct("competition")
    },
)

RAND = default_rng()

if False:
    import logging

    # Enabling debugging at http.client level (requests->urllib3->http.client)
    # you will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
    # the only thing missing will be the response.body which is not logged.
    try:  # for Python 3
        from http.client import HTTPConnection
    except ImportError:
        from httplib import HTTPConnection

    HTTPConnection.debuglevel = 1

    logging.basicConfig()  # you need to initialize logging, otherwise you will not see anything from requests
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


tabulate.PRESERVE_WHITESPACE = True


app = typer.Typer()


class SeasonScores(defaultdict):
    def __init__(self, *args):
        super().__init__(None, *args)

    def __missing__(self, key):
        score = SeasonScore(key)
        self[key] = score
        return score

    def copy(self):
        return type(self)((key, score.copy()) for key, score in self.items())


def score_season(games, initial_scores=None):
    if initial_scores:
        scores = initial_scores.copy()
    else:
        scores = SeasonScores()
    for game in games:
        scores[game.homeTeamIndex].add_game(game, True)
        scores[game.awayTeamIndex].add_game(game, False)

    return scores


def predict_games(predictor, games):
    for game in games:
        winner, td = predictor(game.homeTeam, game.awayTeam)
        if winner == game.homeTeam:
            game.homeScore = td
            game.awayScore = -td
        else:
            game.homeScore = -td
            game.awayScore = td


def sum_stats(teams, old, new):
    return {
        teamIdx: SeasonScore(
            teamIdx,
            old[teamIdx].points + new[teamIdx].points,
            old[teamIdx].tdd + new[teamIdx].tdd,
            old[teamIdx].losses + new[teamIdx].losses,
            old[teamIdx].head_to_head | new[teamIdx].head_to_head,
        )
        for teamIdx in range(len(teams))
    }


def sort_stats(stats):
    return sorted(list(stats.items()), key=lambda x: x[1], reverse=True)


def season_teams(games):
    return {game["homeTeamName"] for game in games} | {
        game["awayTeamName"] for game in games
    }


def _predict_season(
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

    season = games(league, season, division)
    teams = sorted(season_teams(season))
    team_indexes = {team: index for index, team in enumerate(teams)}
    for game in season:
        if "Playin" in division or "TG" in division:
            game["round"] += 8
        game["homeTeamIndex"] = team_indexes[game["homeTeamName"]]
        game["awayTeamIndex"] = team_indexes[game["awayTeamName"]]

    season = [
        Game(
            game["homeTeamName"],
            -1 if game["homeScore"] is None else game["homeScore"],
            game["homeTeamIndex"],
            game["awayTeamName"],
            -1 if game["awayScore"] is None else game["awayScore"],
            game["awayTeamIndex"],
            game["round"],
        )
        for game in season
    ]

    played_games = [
        game
        for game in season
        if game.homeScore >= 0
        and game.awayScore >= 0
        and (as_of is None or game.round <= as_of)
    ]
    remaining_games = [
        game
        for game in season
        if game.homeScore == -1
        or game.awayScore == -1
        or (as_of is not None and game.round > as_of)
    ]

    current_stats = score_season(played_games)

    outcomes = []
    for _ in range(iterations):
        predict_games(predictor, remaining_games)
        total_stats = score_season(remaining_games, initial_scores=current_stats)
        # total_stats = sum_stats(teams, current_stats, remaining_stats)
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


RACES = [
    "Amazon",
    "Bretonnian",
    "Chaos",
    "ChaosDwarf",
    "DarkElf",
    "Dwarf",
    "ElvenUnion",
    "Goblin",
    "Halfling",
    "HighElf",
    "Human",
    "Khemri",
    "Kislev",
    "Lizardman",
    "Necromantic",
    "Norse",
    "Nurgle",
    "Orc",
    "Ogre",
    "Skaven",
    "Undead",
    "UnderworldDenizens",
    "Vampire",
    "WoodElf",
]


def win_chance_from_stats(stats, opp_race):
    win_p = 0.5
    win_p = (win_p * stats["S15_p"]) / (
        win_p * stats["S15_p"] + (1 - win_p) * (1 - stats["S15_p"])
    )
    win_p = (win_p * stats["Total_p"]) / (
        win_p * stats["Total_p"] + (1 - win_p) * (1 - stats["Total_p"])
    )
    win_p = win_p * stats[f"{opp_race}_p"]

    return win_p


def prep_row(row):
    if row["race"] == "ProElf":
        row["race"] = "ElvenUnion"
    if row["race"] == "Bretonnia":
        row["race"] = "Bretonnian"
    for col, weight in [
        ("S15", 50),
        ("Total", 100),
    ] + [(race, 30) for race in RACES]:
        if row[col]:
            win, tie, loss = (int(val) for val in row[col].split("/"))
        else:
            win = tie = loss = 0
        row[f"{col}_w"] = win
        row[f"{col}_t"] = tie
        row[f"{col}_l"] = loss
        total = win + tie + loss + weight
        row[f"{col}_p"] = (win + tie / 2 + weight / 2) / total

    for race in RACES:
        row[race] = win_chance_from_stats(row, race)

    return row


coach_stats = {
    **{row["team"]: prep_row(row) for row in predictions["rebbl_playoff_stats"].find()},
    **{row["team"]: prep_row(row) for row in predictions["rebbl_cc_stats"].find()},
}

_beta_params = {}


def beta_params(home, away):
    if (home, away) not in _beta_params:
        home_stats = coach_stats[home]
        away_stats = coach_stats[away]
        alpha = 1
        beta = 1
        home_race = home_stats["race"]
        away_race = away_stats["race"]
        for col in ("S15", "Total"):
            alpha += (
                home_stats[f"{col}_w"]
                + home_stats[f"{col}_t"] / 2
                + away_stats[f"{col}_l"]
                + away_stats[f"{col}_t"] / 2
            )
            beta += (
                home_stats[f"{col}_l"]
                + home_stats[f"{col}_t"] / 2
                + away_stats[f"{col}_w"]
                + away_stats[f"{col}_t"] / 2
            )
        alpha += (
            home_stats[f"{away_race}_w"]
            + home_stats[f"{away_race}_t"] / 2
            + away_stats[f"{home_race}_l"]
            + away_stats[f"{home_race}_t"] / 2
        )
        beta += (
            home_stats[f"{away_race}_l"]
            + home_stats[f"{away_race}_t"] / 2
            + away_stats[f"{home_race}_w"]
            + away_stats[f"{home_race}_t"] / 2
        )
        _beta_params[(home, away)] = (alpha, beta)
    return _beta_params[(home, away)]


def predict_from_beta_dist(home, away):
    (alpha, beta) = beta_params(home, away)
    p = RAND.beta(alpha, beta)
    if p > 0.5:
        return home, 1
    else:
        return away, -1


def predict_from_stats(home, away):
    home_stats = coach_stats[home]
    away_stats = coach_stats[away]
    home_win_p = home_stats[away_stats["race"]]
    away_win_p = home_stats[home_stats["race"]]

    home_win_frac = home_win_p / (home_win_p + away_win_p)
    home_win = random.random() < home_win_frac
    return home if home_win else away, 1 if home_win else -1


def predict_playoffs(
    name,
    round_one_matches,
    slot_odds=None,
    iterations=1000,
    as_of=None,
    predictor: Callable = random_score,
):
    pending_rows = []

    def predict_match(home, away, round, iteration):
        if home is None:
            winner = away
        elif away is None:
            winner = home
        else:
            winner, tdd = predictor(home, away)

        pending_rows.append(
            dict(
                home=home,
                away=away,
                winner=winner,
                round=round,
                name=name,
                iteration=iteration,
                predictor=predictor.__name__,
            )
        )
        return winner

    playoffs = predictions["playoffs"]

    prev_results = None
    while True:
        results = {
            (row["round"], row["winner"]): row["count"]
            for row in predictions.query(
                """
                    SELECT
                        round,
                        winner,
                        count(*) AS count
                    FROM playoffs
                    WHERE name=:name
                    AND predictor=:predictor
                    GROUP BY 1, 2
                """,
                name=name,
                predictor=predictor.__name__,
            )
        }
        if prev_results is not None:
            pct_change = [
                (
                    (results[(round, winner)] - prev_results.get((round, winner), 1))
                    / (prev_results.get((round, winner), 1)),
                    round,
                    winner,
                )
                for (round, winner) in results.keys()
            ]
            print(max(pct_change))
            if max(pct_change)[0] < 0.05:
                break

        prev_results = results
        for i in range(iterations):
            round = 1
            round_matches = round_one_matches.copy()
            next_round_matches = []
            while len(round_matches) > 1:
                for (home1, away1), (home2, away2) in zip(
                    round_matches[0::2], round_matches[1::2]
                ):
                    winner1 = predict_match(home1, away1, round, i)
                    winner2 = predict_match(home2, away2, round, i)
                    next_round_matches.append((winner1, winner2))

                round_matches = next_round_matches
                next_round_matches = []
                round += 1

            (home, away) = round_matches[0]
            winner = predict_match(home, away, round, i)
            playoffs.insert_many(pending_rows)
            pending_rows = []


def leagues():
    return requests.get(f"https://rebbl.net/api/v2/league/").json()


def seasons(league):
    return requests.get(f"https://rebbl.net/api/v2/league/{league}/Seasons").json()


def divisions(league, season):
    resp = requests.get(f"https://rebbl.net/api/v2/division/{league}/{season}")
    return resp.json()


def tickets(league, season):
    return requests.get(
        f"https://rebbl.net/api/v2/standings/{league}/{season}/tickets"
    ).json()


def games(league, season, division):
    return requests.get(
        f"https://rebbl.net/api/v2/division/{league}/{season}/{division}/slim"
    ).json()


def playoff(playoff):
    return requests.get(f"https://rebbl.net/api/v1/playoffs/{playoff}").json()

def _random_score(home, away):
    homeScore = random_score(home)
    awayScore = random_score(away)
    if homeScore > awayScore:
        return home, homeScore - awayScore
    else:
        return away, homeScore - awayScore


PREDICTORS = {
    'predict_from_beta_dist': predict_from_beta_dist,
    'predict_from_stats': predict_from_stats,
    'random_score': _random_score,
}

Predictor = Enum("Predictor", {f: f for f in PREDICTORS.keys()})


@app.command()
def predict_season(
    season: str = "season 15",
    as_of: Optional[int] = None,
    predictor: Predictor = Predictor.random_score,
    iterations: int = 1000,
):
    playoff_slots = []
    challenger_slots = []

    for league in (
        "REBBL - REL",
        "REBBL - GMan",
        "REBBL - Big O",
    ):
        tkts = tickets(league, season)
        for division, playoff_line, challenger_line in zip(
            divisions(league, season), tkts["cutoff"], tkts["challenger"]
        ):
            actual_league = league
            actual_division = division
            if ("REL" in league or "GMan" in league) and (
                ("Division 5" in division)
                or ("Playins" in division)
                or ("TG" in division)
            ):
                actual_league = f"{league} 2"
            if "Big O" in league and "Div 3" in division:
                actual_division = f"{division} "
            results_table = _predict_season(
                actual_league,
                season,
                actual_division,
                iterations=iterations,
                as_of=as_of,
                predictor=PREDICTORS[predictor.value],
            )
            team_order = [
                team
                for (team, _) in sorted(
                    results_table.items(), key=lambda x: x[1], reverse=True
                )
            ]

            for index in range(playoff_line):
                playoff_slots.append(
                    [
                        (team, results_table[team][index])
                        for team in team_order
                        if results_table[team][index] > 0
                    ]
                )

            if challenger_line < len(team_order):
                for index in range(playoff_line, challenger_line):
                    challenger_slots.append(
                        [
                            (team, results_table[team][index])
                            for team in team_order
                            if results_table[team][index] > 0
                        ]
                    )

            print(
                tabulate.tabulate(
                    [results_table[team] for team in team_order],
                    showindex=team_order,
                    floatfmt="2.0%",
                    headers=[f"{actual_league} - {actual_division}"]
                    + [
                        f"#{place+1} PO"
                        if place < playoff_line
                        else f"#{place+1} CC"
                        if place < challenger_line and challenger_line < len(team_order)
                        else f"#{place+1}"
                        for place in range(len(team_order))
                    ],
                )
            )
            print()

    pprint.pprint(playoff_slots)
    pprint.pprint(challenger_slots)


@app.command()
def predict_offseason(
    name: Competition,
    iterations: int = 1000,
    predictor: Predictor = Predictor.random_score,
):
    playoff_index = {
        int(row["index"]): row["team"]
        for row in rebbl_stats.find(competition=name.value)
    }
    playoff_matches = [
        (playoff_index.get(idx), playoff_index.get(idx + 1))
        for idx in range(1, max(playoff_index.keys()), 2)
    ]
    predict_playoffs(
        name.value,
        playoff_matches,
        iterations=iterations,
        predictor=PREDICTORS[predictor],
    )
    results = list(
        predictions.query(
            """
                SELECT
                    round,
                    "group",
                    FIRST_VALUE(winner) OVER (
                        PARTITION BY round, "group"
                        ORDER BY count DESC
                    ) AS winner,
                    FIRST_VALUE(count) OVER (
                        PARTITION BY round, "group"
                        ORDER BY count DESC
                    ) AS count
                FROM (
                    SELECT
                        round,
                        ("index"-1)/(1 << round) as "group",
                        winner,
                        count(*) AS count
                    FROM playoffs
                    JOIN rebbl_stats ps
                    ON winner = team
                    AND competition = name
                    WHERE name=:name
                    AND predictor=:predictor
                    GROUP BY 1, 3
                    ORDER BY 1 DESC, 2 asc, 4 DESC
                )
                GROUP BY 1, 2
                ORDER BY 1 DESC, 2 ASC
            """,
            name=name.value,
            predictor=predictor.__name__,
        )
    )
    print(tabulate.tabulate(results))


class Tournament(Enum):
    Playoff15 = "po15"
    ChalCup15 = "cc15"
    Playoff16 = "po16"
    ChalCup16 = "cc16"


REBBL_IDS = {
    Tournament.Playoff15: "REBBL Playoffs Season 15",
    Tournament.ChalCup15: "Challenger's Cup XV",
    Tournament.Playoff16: "REBBL Playoffs Season 16",
    Tournament.ChalCup16: "Challenger's Cup XVI",
}

CHALLONGE_IDS = {
    Tournament.ChalCup15: 9289905,
    Tournament.Playoff15: 9289820,
    Tournament.Playoff16: 9823214,
    Tournament.ChalCup16: 9823264,
}


@app.command()
def sync_challonge(
    tournament: Tournament,
    username: str = typer.Option(None, prompt=True),
    api_key: str = typer.Option(None, prompt=True),
):
    print(f"Updating {tournament}")
    challonge.set_credentials(username, api_key)
    chal_matches = challonge.matches.index(CHALLONGE_IDS[tournament])
    chal_participants = challonge.participants.index(CHALLONGE_IDS[tournament])
    participant_ids = {
        participant["name"].split(" - ")[0].lower(): participant["id"]
        for participant in chal_participants
    }
    chal_match_ids = {
        (match["player1_id"], match["player2_id"]): (
            match["id"],
            match["scores_csv"],
            match["winner_id"],
        )
        for match in chal_matches
    }

    playoff_games = playoff(REBBL_IDS[tournament])
    completed_rebbl_matches = [
        (
            (
                match["opponents"][0]["team"]["name"],
                match["opponents"][0]["team"]["score"],
            ),
            (
                match["opponents"][1]["team"]["name"],
                match["opponents"][1]["team"]["score"],
            ),
            match["winner"]["team"]["name"],
        )
        for round in playoff_games["matches"].values()
        for match in round
        if match.get("winner")
    ]

    print(f"Found {len(completed_rebbl_matches)} matches")
    for ((home, home_score), (away, away_score), winner) in completed_rebbl_matches:
        match_id, scores, winner_id = chal_match_ids[
            (participant_ids[home.lower()], participant_ids[away.lower()])
        ]
        if (
            not scores
            or [int(score) for score in scores.split("-")] != [home_score, away_score]
            or participant_ids[winner.lower()] != winner_id
        ):
            print(
                home, home_score, away, away_score, winner, match_id, scores, winner_id
            )
            print(f"Updating {home}/{away} to {home_score}-{away_score}")
            challonge.matches.update(
                CHALLONGE_IDS[tournament],
                match_id,
                scores_csv=f'{home_score}-{away_score}',
                winner_id=participant_ids[winner.lower()],
            )


if __name__ == "__main__":
    app()
