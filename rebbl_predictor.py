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
import dataset

from _rebbl_predictor import SeasonScore, random_score, Game

predictions = dataset.connect("sqlite:///predictions.db")

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
        game.homeScore = predictor(game.homeTeam)
        game.awayScore = predictor(game.awayTeam)


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


def prep_row(row):
    if row['race'] == 'ProElf':
        row['race'] = 'ElvenUnion'
    if row['race'] == 'Bretonnia':
        row['race'] = 'Bretonnian'
    for col, weight in (
        ("S15", 50),
        ("Total", 100),
        ("Amazon", 30),
        ("Bretonnian", 30),
        ("Chaos", 30),
        ("ChaosDwarf", 30),
        ("DarkElf", 30),
        ("Dwarf", 30),
        ("ElvenUnion", 30),
        ("Goblin", 30),
        ("Halfling", 30),
        ("HighElf", 30),
        ("Human", 30),
        ("Khemri", 30),
        ("Kislev", 30),
        ("Lizardman", 30),
        ("Necromantic", 30),
        ("Norse", 30),
        ("Nurgle", 30),
        ("Orc", 30),
        ("Ogre", 30),
        ("Skaven", 30),
        ("Undead", 30),
        ("UnderworldDenizens", 30),
        ("Vampire", 30),
        ("WoodElf", 30),
    ):
        if row[col]:
            win, tie, loss = (int(val) for val in row[col].split("/"))
        else:
            win = tie = loss = 0
        total = win + tie + loss + weight
        row[f"{col}_p"] = (win + tie/2 + weight/2)/total
    return row


coach_stats = {
    **{row["team"]: prep_row(row) for row in predictions["rebbl_playoff_stats"].find()},
    **{row["team"]: prep_row(row) for row in predictions["rebbl_cc_stats"].find()},
}


def win_chance_from_stats(stats, opp_race):
    win_p = 0.5
    win_p = (win_p * stats['S15_p']) / (win_p * stats['S15_p'] + (1 - win_p) * (1 - stats['S15_p']))
    win_p = (win_p * stats['Total_p']) / (
        win_p * stats['Total_p'] + (1 - win_p) * (1 - stats['Total_p'])
    )
    win_p = (win_p * stats[f'{opp_race}_p'])

    return win_p


def predict_from_stats(home, away):
    home_stats = coach_stats[home]
    away_stats = coach_stats[away]
    home_win_p = win_chance_from_stats(home_stats, away_stats['race'])
    away_win_p = win_chance_from_stats(away_stats, home_stats['race'])

    home_win_frac = home_win_p / (home_win_p + away_win_p)
    home_win = random.random() < home_win_frac
    return home if home_win else away, 1 if home_win else -1


def predict_playoffs(
    name, round_one_matches, slot_odds=None, iterations=1000, as_of=None
):
    predictor = predict_from_stats
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


def main_season():
    season = "season 15"
    as_of = 6

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
            results_table = predict_season(
                actual_league, season, actual_division, iterations=1000, as_of=as_of
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


def main_playoffs():

    playoff_matches = [
        ("Damage Incorporated.", None),
        ("Return of The Phobias!", "Glart DeathGrips Jr."),
        ("Blue-tongued Bruisers", "The Spanish lnquisition"),
        ("Aves of a Feder", "Fecal Fantasy VII"),
        ("Scaled Actors Guild", "Skull Raisers"),
        ("Kingslayerzs", "Casters' Curse"),
        ("Salty Wounds", "The Lubricated Lawmen"),
        ("Bourbon Street Brawlers", "Ordinary Elves"),
        ("The Voltrex Vanguard", "The Nec-Romancers"),
        ("The Descecrators", "Rature Time"),
        ("Moist Owlettes", "Very High Heels"),
        ("Funky Flowers", "Deadbeat Ex's"),
        ("The Tanking Generals", "The Bigger Flex"),
        ("High Barnet Raiders", "Ratt Utd Legends"),
        ("Warc Machine III", "Red Bull Chorfing"),
        ("Smoked and Cured", "The Rat REBBL"),
        ("Bullcanoe!", None),
        ("Baltigore Ravens", "Disn'Orc"),
        ("Cold Cutz", "Taintburglars"),
        ("Made in Chernobyl", "Jungler Beats"),
        ("The  Greenhorns", "Entheogenisis"),
        ("Charlestown Chiefs", "New Romantic XI"),
        ("Talk Show Terror", "Predatory Pocket Monsters"),
        ("Knights Saying Ni", "New Yorc Pilanders"),
        ("Knight Juggler is back", None),
        ("Putilinoids", "Zoot's Oddjobs"),
        ("Dreams of Golden Streams", "Rowdy Vanity Passers"),
        ("The Jumpy Sproingers", "Practice Makes Permanent"),
        ("Bare Gills", "Chaos Goes Forth"),
        ("Major Annoyance", "Marvelous Creatures"),
        ("Felwithe's Koada'Dal", "Wood United"),
        ("CHEESE + CAKE ", "Trump's Chumps"),
    ]
    predict_playoffs("Season 15 Playoffs", playoff_matches, iterations=100000)
    print(tabulate.tabulate(list(
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
                    JOIN rebbl_playoff_stats ps
                    ON winner = team
                    WHERE name='Season 15 Playoffs'
                    GROUP BY 1, 3
                    ORDER BY 1 DESC, 2 asc, 4 DESC
				)
				GROUP BY 1, 2
                ORDER BY 2 DESC, 1 ASC
            """
        )
    )))


def main_cc():

    playoff_matches = [
        ("From Beyond", "Rave'n Skaven!"),
        ("Get To The Choppah", "High Sea Surfriders"),
        ("Mønster Bash", "FootBulldogs"),
        ("Delicious Elves 3.0", "Black Creek Buccaneers"),
        ("Mass Extinction Event", "Nob Goblins"),
        ("Toronto Trash Pandas", "Dance Monkeys"),
        ("Southern Wildlings", "High Toon Hustlers"),
        ("Crawling Croks", "Chicago Bolshoit"),
        ("Schnitz and Giggles", "Never Die Twice"),
        ("The Floating Toads", "The Big Zuccs"),
        ("West Easton Skittlers", "Scales of Beauty"),
        ("Khorne Worshipping Doom", "Zoltans Zwords"),
        ("Da Brute Skwadd", "Flyboys"),
        ("Melbourne Marauders", "Crawl Contenders"),
        ("The Horn of Gor'thanc", "Kicker of Elves"),
        ("The Killer Kings", "Scooby and the Gang"),
        ("Gorgoth All Grays", "Life After Love"),
        ("Dark Mode FTW", "WrackleMania"),
        ("Spooky Action Atavistic", "The Cathartic Carnivores"),
        ("Apocalypse of Death", "Orcanized Krime"),
        ("The Lizzardblizzard", "Straight Crooked Wood"),
        ("The Breton Brawlers", "Unquiet Vertabrae"),
        ("All rats must DIE again", "ReBBL Klobba Klubb"),
        ("Skinkin' Ain't Eazy", "Gorgoth Golden Guards"),
        ("Dull Fangs II", "Noobtown Smasherz"),
        ("From Tou Till Can", "Dakimakuras"),
        ("-= Les Fléaux =-", "The Knives of Khaine"),
        ("Marry Me Bloody Mary X", "Sabertooth Vag 3.0"),
        ("Pesedjet", "Mongrels of the Empire"),
        ("Children of Dune", "Dazed N Confused"),
        ("We're Slaying 'Em", "We See Dead People, Again"),
        ("The Tiger Lizards", "Gods Of Fate"),
    ]
    predict_playoffs("Season 15 Challenger Cup", playoff_matches, iterations=100)
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
                    JOIN rebbl_cc_stats ps
                    ON winner = team
                    WHERE name='Season 15 Challenger Cup'
                    GROUP BY 1, 3
                    ORDER BY 1 DESC, 2 asc, 4 DESC
				)
				GROUP BY 1, 2
                ORDER BY 1 DESC, 2 ASC
            """
        )
    )
    print(tabulate.tabulate(results))


if __name__ == "__main__":
    main_cc()
