import heapq
from typing import List, Tuple


class VotingRule:
    def __init__(self, num_candidates: int, preferences: List[List[int]], log=False):
        self.num_candidates = num_candidates
        # candidate indices are assumed to be 0-indexed in preferences and are in lexicographic ordering
        self.preferences = preferences
        self.log = log

    def update_preferences(self, preferences: List[List[int]]):
        self.preferences = preferences

    def get_plurality_scores(
        self, num_candidates: int, preferences: List[List[int]]
    ) -> List[List[int]]:
        plurality_scores = [0 for i in range(num_candidates)]

        for preference in preferences:
            plurality_scores[preference[0]] += 1

        return plurality_scores

    def head_to_head_scores(
        self, candidate_1: int, candidate_2: int, preferences: List[List[int]]
    ) -> Tuple[int, int]:
        score_1, score_2 = 0, 0
        for preference in preferences:
            if preference.index(candidate_1) < preference.index(candidate_2):
                score_1 += 1
            else:
                score_2 += 1
        return score_1, score_2

    def head_to_head_winner(
        self, candidate_1: int, candidate_2: int, preferences: List[List[int]]
    ) -> int:
        score_1, score_2 = self.head_to_head_scores(
            candidate_1=candidate_1, candidate_2=candidate_2, preferences=preferences
        )

        if self.log:
            print("Head to head scores")
            print(
                str(candidate_1)
                + ": "
                + str(score_1)
                + ", "
                + str(candidate_2)
                + ": "
                + str(score_2)
            )

        if score_1 > score_2:
            return candidate_1
        elif score_1 < score_2:
            return candidate_2
        else:
            if candidate_1 < candidate_2:
                return candidate_1
            else:
                return candidate_2

    def get_winner(self) -> int:
        return 0


class Plurality(VotingRule):
    def get_winner(self) -> int:
        plurality_scores = self.get_plurality_scores(
            num_candidates=self.num_candidates, preferences=self.preferences
        )
        if self.log:
            print("Plurality Scores")
            print(plurality_scores)

        winner = plurality_scores.index(max(plurality_scores))
        return winner


class Borda(VotingRule):
    def get_winner(self) -> int:
        borda_count = [0 for i in range(self.num_candidates)]

        for preference in self.preferences:
            for rank, candidate in enumerate(preference):
                borda_count[candidate] += self.num_candidates - rank - 1

        if self.log:
            print("Borda Counts")
            print(borda_count)

        winner = borda_count.index(max(borda_count))
        return winner


class PluralityWithRunoff(VotingRule):
    def get_winner(self) -> int:
        plurality_scores = self.get_plurality_scores(
            num_candidates=self.num_candidates, preferences=self.preferences
        )

        if self.log:
            print("Plurality Scores")
            print(plurality_scores)

        max_1, max_2 = heapq.nlargest(2, plurality_scores)

        candidate_1 = plurality_scores.index(max_1)
        candidate_2 = -1
        for candidate, score in enumerate(plurality_scores):
            if (score == max_2) and (candidate != candidate_1):
                candidate_2 = candidate
                break

        if self.log:
            print("Top 2 candidates")
            print(candidate_1, candidate_2)

        winner = self.head_to_head_winner(
            candidate_1, candidate_2, preferences=self.preferences
        )

        return winner


class SingleTransferableVote(VotingRule):
    def get_winner(self) -> int:
        round_num = 0
        removed_candidates = set()
        new_preferences = self.preferences[:][:]
        while self.num_candidates - len(removed_candidates) > 1:
            round_num += 1
            if self.log:
                print("Round " + str(round_num))
            plurality_scores = self.get_plurality_scores(
                num_candidates=self.num_candidates, preferences=new_preferences
            )

            if self.log:
                print("Plurality Scores")
                print(plurality_scores)

            # assign large scores to already eliminated candidates
            for candidate in removed_candidates:
                plurality_scores[candidate] = len(new_preferences) + 1

            min_score = min(plurality_scores)
            candidate_elim = -1

            for candidate, score in reversed(list(enumerate(plurality_scores))):
                if score == min_score:
                    candidate_elim = candidate
                    break

            if self.log:
                print("Eliminated Candidate: " + str(candidate_elim))

            removed_candidates.add(candidate_elim)
            for preference in new_preferences:
                preference.remove(candidate_elim)

            if self.log:
                print("New preferences")
                for preference in new_preferences:
                    print(preference)

        winner = -1
        for candidate in range(0, self.num_candidates):
            if candidate not in removed_candidates:
                winner = candidate

        return winner


class Copeland(VotingRule):
    def get_winner(self) -> int:
        copeland_scores = [0 for i in range(self.num_candidates)]
        for candidate_1 in range(0, self.num_candidates):
            for candidate_2 in range(candidate_1 + 1, self.num_candidates):
                if self.log:
                    print(
                        "Head to head between "
                        + str(candidate_1)
                        + " and "
                        + str(candidate_2)
                    )
                score_1, score_2 = self.head_to_head_scores(
                    candidate_1=candidate_1,
                    candidate_2=candidate_2,
                    preferences=self.preferences,
                )

                if self.log:
                    print(
                        str(candidate_1)
                        + ": "
                        + str(score_1)
                        + ", "
                        + str(candidate_2)
                        + ": "
                        + str(score_2)
                    )

                copeland_score_1, copeland_score_2 = 0, 0
                if score_1 > score_2:
                    copeland_scores[candidate_1] += 2
                    copeland_score_1 = 1
                elif score_2 > score_1:
                    copeland_scores[candidate_2] += 2
                    copeland_score_2 = 1
                else:
                    copeland_scores[candidate_1] += 1
                    copeland_scores[candidate_2] += 1
                    copeland_score_1 = 0.5
                    copeland_score_2 = 0.5

                if self.log:
                    print(
                        str(candidate_1)
                        + ": "
                        + str(copeland_score_1)
                        + ", "
                        + str(candidate_2)
                        + ": "
                        + str(copeland_score_2)
                    )

        if self.log:
            print([s / 2 for s in copeland_scores])

        winner = copeland_scores.index(max(copeland_scores))

        return winner


class Schulze(VotingRule):
    def get_strongest_paths(self, adjacency_matrix: List[List[int]]) -> List[List[int]]:
        strongest_paths = adjacency_matrix[:][:]
        for candidate in range(self.num_candidates):
            for candidate_1 in range(self.num_candidates):
                for candidate_2 in range(self.num_candidates):
                    if candidate_1 == candidate_2:
                        continue
                    strongest_paths[candidate_1][candidate_2] = max(
                        strongest_paths[candidate_1][candidate_2],
                        min(
                            strongest_paths[candidate_1][candidate],
                            strongest_paths[candidate][candidate_2],
                        ),
                    )
        return strongest_paths

    def get_winner(self) -> int:
        adjacency_matrix = [
            [0 for j in range(self.num_candidates)] for i in range(self.num_candidates)
        ]
        for candidate_1 in range(self.num_candidates):
            for candidate_2 in range(candidate_1 + 1, self.num_candidates):
                score_1, score_2 = self.head_to_head_scores(
                    candidate_1=candidate_1,
                    candidate_2=candidate_2,
                    preferences=self.preferences,
                )
                adjacency_matrix[candidate_1][candidate_2] = score_1 - score_2
                adjacency_matrix[candidate_2][candidate_1] = score_2 - score_1

        if self.log:
            for candidate, adj in enumerate(adjacency_matrix):
                print("Candidate: " + str(candidate))
                print(adj)

        strongest_paths = self.get_strongest_paths(adjacency_matrix)
        if self.log:
            for candidate, paths in enumerate(strongest_paths):
                print("Candidate: " + str(candidate))
                print("Strongest Paths: " + str(paths))

        chain_beat_winners = [
            [True for i in range(self.num_candidates)]
            for j in range(self.num_candidates)
        ]

        for candidate_1 in range(self.num_candidates):
            for candidate_2 in range(candidate_1 + 1, self.num_candidates):
                if (
                    strongest_paths[candidate_1][candidate_2]
                    > strongest_paths[candidate_2][candidate_1]
                ):
                    chain_beat_winner = candidate_1
                elif (
                    strongest_paths[candidate_1][candidate_2]
                    < strongest_paths[candidate_2][candidate_1]
                ):
                    chain_beat_winner = candidate_2
                else:
                    chain_beat_winner = -1

                if chain_beat_winner == -1:
                    if self.log:
                        print(
                            "Candidate "
                            + str(candidate_1)
                            + " chain beats Candidate "
                            + str(candidate_2)
                        )
                    if self.log:
                        print(
                            "Candidate "
                            + str(candidate_2)
                            + " chain beats Candidate "
                            + str(candidate_1)
                        )
                elif chain_beat_winner == candidate_1:
                    chain_beat_winners[candidate_2][candidate_1] = False
                    if self.log:
                        print(
                            "Candidate "
                            + str(candidate_1)
                            + " chain beats Candidate "
                            + str(candidate_2)
                        )

                else:
                    chain_beat_winners[candidate_1][candidate_2] = False

                    if self.log:
                        print(
                            "Candidate "
                            + str(candidate_2)
                            + " chain beats Candidate "
                            + str(candidate_1)
                        )

        winner = -1
        for candidate in range(self.num_candidates):
            beats_all = True
            for x in chain_beat_winners[candidate]:
                if not x:
                    beats_all = False
                    break
            if beats_all:
                winner = candidate
                break

        return winner


def get_preferecnes(
    preference_list: List[List[int]], cnts: List[int]
) -> List[List[int]]:
    preferences = []
    for i in range(len(cnts)):
        for cnt in range(cnts[i]):
            preferences.append(preference_list[i][:])
    return preferences


def main():

    # Schulze
    print("\nSchulze")
    num_candidates = 4
    # R = 0, G = 1, Y = 2, B = 3
    preference_list = [
        [0, 1, 2, 3],
        [3, 0, 2, 1],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [2, 1, 3, 0],
    ]
    cnts = [8, 2, 4, 4, 3]
    preferences = get_preferecnes(preference_list=preference_list, cnts=cnts)

    votingRule = Schulze(
        num_candidates=num_candidates, preferences=preferences, log=True
    )
    print("Winner: " + str(votingRule.get_winner()))

    # Copeland
    print("\nCopeland")
    num_candidates = 4
    # R = 0, G = 1, Y = 2, B = 3
    preference_list = [
        [0, 3, 2, 1],
        [0, 2, 1, 3],
        [0, 1, 2, 3],
        [3, 0, 2, 1],
        [1, 2, 3, 0],
    ]
    cnts = [3, 1, 1, 4, 4]
    preferences = get_preferecnes(preference_list=preference_list, cnts=cnts)

    votingRule = Copeland(
        num_candidates=num_candidates, preferences=preferences, log=True
    )
    print("Winner: " + str(votingRule.get_winner()))

    # STV
    print("\nSTV")
    num_candidates = 4
    # R = 0, G = 1, Y = 2, B = 3
    preference_list = [
        [0, 3, 2, 1],
        [0, 1, 2, 3],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [2, 1, 0, 3],
    ]
    cnts = [3, 3, 2, 4, 2, 2, 1]
    preferences = get_preferecnes(preference_list=preference_list, cnts=cnts)

    votingRule = SingleTransferableVote(
        num_candidates=num_candidates, preferences=preferences, log=True
    )
    print("Winner: " + str(votingRule.get_winner()))

    # Plurality with runoff
    print("\nPlurality with runoff")
    num_candidates = 4
    # R = 0, G = 1, Y = 2, B = 3
    preference_list = [
        [0, 3, 1, 2],
        [0, 1, 2, 3],
        [1, 3, 2, 0],
        [2, 1, 3, 0],
        [3, 2, 1, 0],
    ]
    cnts = [2, 1, 2, 1, 1]
    preferences = get_preferecnes(preference_list=preference_list, cnts=cnts)

    votingRule = PluralityWithRunoff(
        num_candidates=num_candidates, preferences=preferences, log=True
    )
    print("Winner: " + str(votingRule.get_winner()))

    # Borda
    print("\nBorda")
    num_candidates = 4
    # R = 0, G = 1, Y = 2, B = 3
    preference_list = [
        [0, 3, 1, 2],
        [0, 1, 2, 3],
        [1, 3, 2, 0],
        [2, 1, 3, 0],
        [3, 2, 1, 0],
    ]
    cnts = [1, 1, 1, 1, 1]
    preferences = get_preferecnes(preference_list=preference_list, cnts=cnts)

    votingRule = Borda(num_candidates=num_candidates, preferences=preferences, log=True)
    print("Winner: " + str(votingRule.get_winner()))

    # Plurality
    print("\nPlurality")
    num_candidates = 4
    # R = 0, G = 1, Y = 2, B = 3
    preference_list = [
        [0, 3, 1, 2],
        [0, 1, 2, 3],
        [1, 3, 2, 0],
        [2, 1, 3, 0],
        [3, 2, 1, 0],
    ]
    cnts = [1, 1, 1, 1, 1]
    preferences = get_preferecnes(preference_list=preference_list, cnts=cnts)

    votingRule = Plurality(num_candidates=num_candidates, preferences=preferences, log=True)
    print("Winner: " + str(votingRule.get_winner()))


if __name__ == "__main__":
    main()
