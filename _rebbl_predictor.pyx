from libc.stdlib cimport rand, RAND_MAX

cdef class SeasonScore:
    cdef public int team
    cdef public int points
    cdef public int tdd
    cdef public int losses
    cdef public int head_to_head

    def __cinit__(self, team, points=0, tdd=0, losses=0, head_to_head=0):
        self.team = team
        self.points = points
        self.tdd = tdd
        self.losses = losses
        self.head_to_head = head_to_head

    cpdef add_game(self, game, bint home):
        cdef int homeScore = game["homeScore"]
        cdef int awayScore = game["awayScore"]
        cdef int homeIndex = game["homeTeamIndex"]
        cdef int awayIndex = game["awayTeamIndex"]
        if home:
            if homeScore > awayScore:
                self.points += 3
                self.head_to_head |= 1 << awayIndex
            elif homeScore == awayScore:
                self.points += 1
            else:
                self.losses += 1
            self.tdd += homeScore - awayScore
        else:
            if homeScore < awayScore:
                self.points += 3
                self.head_to_head |= 1 << homeIndex
            elif homeScore == awayScore:
                self.points += 1
            else:
                self.losses += 1
            self.tdd += awayScore - homeScore

    cdef bint _lt(self, SeasonScore other):
        if self.points != other.points:
            return self.points < other.points
        if self.tdd != other.tdd:
            return self.tdd < other.tdd
        if self.losses != other.losses:
            return self.losses > other.losses
        return 1 << self.team & other.head_to_head

    def __lt__(self, other):
        other  = <SeasonScore?>other
        return self._lt(other)

    cdef bint _eq(self, SeasonScore other):
        return (
            self.points == other.points
            and self.tdd == other.tdd
            and self.losses == other.losses
            and not (1 << self.team & other.head_to_head)
            and not (1 << other.team & self.head_to_head)
        )

    def __eq__(self, other):
        other  = <SeasonScore?>other
        return self._eq(other)

    def __repr__(self):
        return f"SeasonScore({self.team!r}, {self.points!r}, {self.tdd!r}, {self.losses!r}, {self.head_to_head!r})"


cpdef int random_score(str team):
    choice = rand()/RAND_MAX
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
