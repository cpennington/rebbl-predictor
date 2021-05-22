#!/usr/bin/env python
import pprint
from enum import Enum

import challonge
import requests
import typer

session = requests.Session()
session.headers.pop('User-Agent')

challonge.api.request = session.request

app = typer.Typer()


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


def playoff(playoff):
    return requests.get(f"https://rebbl.net/api/v1/playoffs/{playoff}").json()


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
    participant_names = {
        id: name for (name, id) in participant_ids.items()
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

    print(f"Found {len(completed_rebbl_matches)} rebbl matches, {len(chal_match_ids)} challonge matches")
    pprint.pprint({
        (participant_names.get(home), participant_names.get(away)): (match_id, scores, participant_names.get(winner))
        for ((home, away), (match_id, scores, winner)) in chal_match_ids.items()
    })
    pprint.pprint(completed_rebbl_matches)
    for ((home, home_score), (away, away_score), winner) in completed_rebbl_matches:
        if (participant_ids.get(home.lower()), participant_ids.get(away.lower())) not in chal_match_ids:
            continue
        match_id, scores, winner_id = chal_match_ids[
            (participant_ids.get(home.lower()), participant_ids.get(away.lower()))
        ]
        if (
            not scores
            or [int(score) for score in scores.split("-")] != [home_score, away_score]
            or participant_ids.get(winner.lower()) != winner_id
        ):
            print(
                home, home_score, away, away_score, winner, match_id, scores, winner_id
            )
            print(f"Updating {home}/{away} to {home_score}-{away_score}")
            challonge.matches.update(
                CHALLONGE_IDS[tournament],
                match_id,
                scores_csv=f'{home_score}-{away_score}',
                winner_id=participant_ids.get(winner.lower()),
            )


if __name__ == "__main__":
    app()
