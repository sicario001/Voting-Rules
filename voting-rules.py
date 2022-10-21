import copy
import heapq
import math
from typing import List, Tuple
import numpy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


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


class ScoreBased(VotingRule):
    def get_scores(self) -> List[int]:
        scores = [0 for i in range(self.num_candidates)]
        return scores

    def is_valid_next_candidate(
        self,
        next_candidate: int,
        next_rank: int,
        voter: int,
        candidate: int,
        new_preferences: List[List[int]],
    ) -> bool:
        new_preferences[voter][next_rank] = next_candidate
        remaining_candidates = set([i for i in range(self.num_candidates)])
        for c in new_preferences[voter][: next_rank + 1]:
            remaining_candidates.remove(c)
        for ind, c in enumerate(remaining_candidates):
            new_preferences[voter][next_rank + 1 + ind] = c

        self.update_preferences(new_preferences)

        log = self.log
        self.log = False

        scores = self.get_scores()

        self.log = log

        if scores[candidate] > scores[next_candidate]:
            return True
        elif (scores[candidate] == scores[next_candidate]) and (
            candidate < next_candidate
        ):
            return True
        else:
            return False

    def can_win_by_misreporting(
        self, voter: int, candidate: int, original_preferences: List[List[int]]
    ) -> bool:
        new_preferences = copy.deepcopy(original_preferences)
        valid_candidates = set([i for i in range(self.num_candidates)])

        new_preferences[voter][0] = candidate
        valid_candidates.remove(candidate)
        next_rank = 1

        while next_rank < self.num_candidates:
            valid = False
            for next_candidate in valid_candidates:
                if self.is_valid_next_candidate(
                    next_candidate, next_rank, voter, candidate, new_preferences
                ):
                    valid = True

                    new_preferences[voter][next_rank] = next_candidate
                    valid_candidates.remove(next_candidate)
                    next_rank += 1

                    break
            if not valid:
                return False
        self.update_preferences(new_preferences)

        log = self.log
        # self.log = False

        new_winner = self.get_winner()

        self.log = log
        assert new_winner == candidate

        if self.log:
            print("Voter " + str(voter) + " can improve by misreporting")
            print("Original preference: " + str(original_preferences[voter]))
            print("New preference: " + str(new_preferences[voter]))
            print("New winner: " + str(new_winner))

        return True

    def is_manipulable(self) -> bool:
        original_preferences = copy.deepcopy(self.preferences)

        log = self.log
        self.log = False

        original_winner = self.get_winner()

        self.log = log

        for voter in range(len(original_preferences)):
            better_candidates = original_preferences[voter][
                : (original_preferences[voter].index(original_winner))
            ]
            for candidate in better_candidates:
                if self.can_win_by_misreporting(voter, candidate, original_preferences):
                    return True
        return False

    def get_winner(self) -> bool:
        scores = self.get_scores()
        winner = scores.index(max(scores))
        return winner


class Plurality(ScoreBased):
    def get_scores(self) -> List[int]:
        plurality_scores = self.get_plurality_scores(
            num_candidates=self.num_candidates, preferences=self.preferences
        )
        if self.log:
            print("Plurality Scores")
            print(plurality_scores)
        return plurality_scores


class Borda(ScoreBased):
    def get_scores(self) -> List[int]:
        borda_count = [0 for i in range(self.num_candidates)]

        for preference in self.preferences:
            for rank, candidate in enumerate(preference):
                borda_count[candidate] += self.num_candidates - rank - 1

        if self.log:
            print("Borda Counts")
            print(borda_count)
        return borda_count


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
        new_preferences = copy.deepcopy(self.preferences)
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


class Copeland(ScoreBased):
    def get_scores(self) -> List[int]:
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

        return copeland_scores


class Schulze(VotingRule):
    def get_strongest_paths(self, adjacency_matrix: List[List[int]]) -> List[List[int]]:
        strongest_paths = copy.deepcopy(adjacency_matrix)
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


def get_preferences(
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
    preferences = get_preferences(preference_list=preference_list, cnts=cnts)

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
    preferences = get_preferences(preference_list=preference_list, cnts=cnts)

    votingRule = Copeland(
        num_candidates=num_candidates, preferences=preferences, log=True
    )
    print("Winner: " + str(votingRule.get_winner()))
    print("Manipulable: " + str(votingRule.is_manipulable()))

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
    preferences = get_preferences(preference_list=preference_list, cnts=cnts)

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
    preferences = get_preferences(preference_list=preference_list, cnts=cnts)

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
    preferences = get_preferences(preference_list=preference_list, cnts=cnts)

    votingRule = Borda(num_candidates=num_candidates, preferences=preferences, log=True)
    print("Winner: " + str(votingRule.get_winner()))
    print("Manipulable: " + str(votingRule.is_manipulable()))

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
    preferences = get_preferences(preference_list=preference_list, cnts=cnts)

    votingRule = Plurality(
        num_candidates=num_candidates, preferences=preferences, log=True
    )
    print("Winner: " + str(votingRule.get_winner()))
    print("Manipulable: " + str(votingRule.is_manipulable()))


def process_sample(voters: int, candidates: int):
    copeland = Copeland(num_candidates=candidates, preferences=None, log=False)
    borda = Borda(num_candidates=candidates, preferences=None, log=False)
    plurality = Plurality(num_candidates=candidates, preferences=None, log=False)

    preferences = []
    for j in range(voters):
        preferences.append(list(numpy.random.permutation(candidates)))

    copeland.update_preferences(preferences)
    borda.update_preferences(preferences)
    plurality.update_preferences(preferences)

    manipulable_copeland = 0
    manipulable_borda = 0
    manipulable_plurality = 0

    if copeland.is_manipulable():
        manipulable_copeland += 1
    if borda.is_manipulable():
        manipulable_borda += 1
    if plurality.is_manipulable():
        manipulable_plurality += 1

    return manipulable_copeland, manipulable_borda, manipulable_plurality


def experiment(voters: int, candidates: int, num_samples: int):
    num_manipulable_copeland = 0
    num_manipulable_borda = 0
    num_manipulable_plurality = 0

    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(
        delayed(process_sample)(voters, candidates) for i in range(num_samples)
    )

    for result in results:
        num_manipulable_copeland += result[0]
        num_manipulable_borda += result[1]
        num_manipulable_plurality += result[2]

    f_manipulable_copeland = num_manipulable_copeland / num_samples
    f_manipulable_borda = num_manipulable_borda / num_samples
    f_manipulable_plurality = num_manipulable_plurality / num_samples

    print("Fraction of manipulable preferences")
    print("Copeland: " + str(f_manipulable_copeland))
    print("Borda: " + str(f_manipulable_borda))
    print("Plurality: " + str(f_manipulable_plurality))

    return f_manipulable_copeland, f_manipulable_borda, f_manipulable_plurality


def exp1():
    # f-manipulable vs number of candidates
    f_copeland = []
    f_borda = []
    f_plurality = []

    exp_candidates: List[int] = [2, 3, 4, 5, 6]
    exp_voters = 100

    samples = 5000

    for candidates in exp_candidates:
        copeland, borda, plurality = experiment(exp_voters, candidates, samples)
        f_copeland.append(copeland)
        f_borda.append(borda)
        f_plurality.append(plurality)

    f_copeland = numpy.array(f_copeland)
    f_borda = numpy.array(f_borda)
    f_plurality = numpy.array(f_plurality)

    exp_candidates = numpy.array(exp_candidates)

    plt.plot(exp_candidates, f_copeland, color="r", label="Copeland")
    plt.plot(exp_candidates, f_borda, color="g", label="Borda")
    plt.plot(exp_candidates, f_plurality, color="b", label="Plurality")

    plt.xticks(exp_candidates)

    plt.xlabel("Number of candidates")
    plt.ylabel("Fraction of manipulable preferences")
    plt.title("Fraction of manipulable preferences with changing candidate count")
    plt.legend()
    # plt.show()
    plt.savefig("f_manipulable_vs_candidates.png")


def exp2():
    # f-manipulable vs sample sizes
    f_copeland = []
    f_borda = []
    f_plurality = []

    exp_candidates = 5
    exp_voters = 100

    samples = [100, 500, 1000, 2000, 5000, 10000]

    for sample in samples:
        copeland, borda, plurality = experiment(exp_voters, exp_candidates, sample)
        f_copeland.append(copeland)
        f_borda.append(borda)
        f_plurality.append(plurality)

    f_copeland = numpy.array(f_copeland)
    f_borda = numpy.array(f_borda)
    f_plurality = numpy.array(f_plurality)

    samples = numpy.array(samples)

    plt.plot(samples, f_copeland, color="r", label="Copeland")
    plt.plot(samples, f_borda, color="g", label="Borda")
    plt.plot(samples, f_plurality, color="b", label="Plurality")

    plt.xticks(samples, fontsize=6)

    plt.xlabel("Number of samples")
    plt.ylabel("Fraction of manipulable preferences")
    plt.title("Fraction of manipulable preferences with changing sample size")
    plt.legend()
    # plt.show()
    plt.savefig("f_manipulable_vs_samples.png")


def exp3():
    # f-manipulable vs number of voters
    f_copeland = []
    f_borda = []
    f_plurality = []

    exp_candidates = 5
    exp_voters = [1, 2, 5, 10, 20, 50, 100, 200]

    samples = 5000

    for exp_voter in exp_voters:
        copeland, borda, plurality = experiment(exp_voter, exp_candidates, samples)
        f_copeland.append(copeland)
        f_borda.append(borda)
        f_plurality.append(plurality)

    f_copeland = numpy.array(f_copeland)
    f_borda = numpy.array(f_borda)
    f_plurality = numpy.array(f_plurality)

    exp_voters = numpy.array(exp_voters)

    plt.plot(exp_voters, f_copeland, color="r", label="Copeland")
    plt.plot(exp_voters, f_borda, color="g", label="Borda")
    plt.plot(exp_voters, f_plurality, color="b", label="Plurality")

    plt.xticks(exp_voters, fontsize=6)

    plt.xlabel("Number of voters")
    plt.ylabel("Fraction of manipulable preferences")
    plt.title("Fraction of manipulable preferences with changing voter count")
    plt.legend()
    # plt.show()
    plt.savefig("f_manipulable_vs_voters.png")


if __name__ == "__main__":
    exp1()
    exp2()
    exp3()
