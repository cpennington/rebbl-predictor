from libc.stdlib cimport rand, RAND_MAX

cdef class Game:
    cdef public str homeTeam
    cdef public int homeScore
    cdef public int homeTeamIndex
    cdef public str awayTeam
    cdef public int awayScore
    cdef public int awayTeamIndex
    cdef public int round

    def __cinit__(self, str homeTeam, int homeScore, int homeTeamIndex, str awayTeam, int awayScore, int awayTeamIndex, int round):
        self.homeTeam = homeTeam
        self.homeScore = homeScore
        self.homeTeamIndex = homeTeamIndex
        self.awayTeam = awayTeam
        self.awayScore = awayScore
        self.awayTeamIndex = awayTeamIndex
        self.round = round
        

cdef class SeasonScore:
    cdef public int teamIdx
    cdef public int points
    cdef public int tdd
    cdef public int losses
    cdef public int head_to_head

    def __cinit__(self, int teamIdx, int points=0, int tdd=0, int losses=0, int head_to_head=0):
        self.teamIdx = teamIdx
        self.points = points
        self.tdd = tdd
        self.losses = losses
        self.head_to_head = head_to_head

    cpdef add_game(self, Game game, bint home):
        cdef int homeScore = game.homeScore
        cdef int awayScore = game.awayScore
        cdef int homeIndex = game.homeTeamIndex
        cdef int awayIndex = game.awayTeamIndex
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
        return 1 << self.teamIdx & other.head_to_head

    def __lt__(self, other):
        other  = <SeasonScore?>other
        return self._lt(other)

    cdef bint _eq(self, SeasonScore other):
        return (
            self.points == other.points
            and self.tdd == other.tdd
            and self.losses == other.losses
            and not (1 << self.teamIdx & other.head_to_head)
            and not (1 << other.teamIdx & self.head_to_head)
        )

    def __eq__(self, other):
        other  = <SeasonScore?>other
        return self._eq(other)

    cpdef copy(self):
        return SeasonScore(
            self.teamIdx, self.points, self.tdd, self.losses, self.head_to_head
        )

    def __repr__(self):
        return f"SeasonScore({self.teamIdx!r}, {self.points!r}, {self.tdd!r}, {self.losses!r}, {self.head_to_head!r})"


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
