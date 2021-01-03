
cdef class SeasonScore:
    cdef public int team
    cdef public int points
    cdef public int tdd
    cdef public int losses
    cdef public int head_to_head

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

