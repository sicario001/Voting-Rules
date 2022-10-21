# Voting-Rules

## Hypothesis
1. For each of Copeland, Borda and Plurality, the winner is the candidate securing the highest score.
2. In Copeland, modifying the preferecne of a single voter, say `v`, might not necessarily lead to an increase in score of `v's` preferred candidate. For instance, consider an example where the head to head scores for every pair of candidates differ by at least 2 without considering the vote of`v`. In such a scenario, vote of `v` won't have any effect on the final scores of candidates.
3. In Borda and Plurality, on the other hand, the vote of `v` can increase the score of `v's` preferred candidates (other than `v's` most preferred candidate). Thus, the extent of manipulability appears higher in Borda and Plurality.
4. In Borda, there is a possibility to increase the maximum score of `v's` more preferred candidates (more preferred than original winner) and at the same time decrease the maximum score of `v's` lesser preferred candidates (original winner and lesser preferred candidates). However, in Plurality, the vote of `v` can only increase the maximum score of `v's` more preferred candidates. So, it appears that Borda in more manipulable than Plurality.

## Experimental Setup
### We did three different experiments
1. Analyzing the convergence of fraction of manipulable preferences with changing sample size. We fixed the number of candidates to `5`, number of voters to `100` and varied the sample size as `[100, 500, 1000, 2000, 5000, 10000]`.
2. Analyzing the fraction of manipulable preferences with changing candidate count. We fixed the number of voters to `100` and number of samples to `5000` and varied the candidate count as `[2, 3, 4, 5, 6]`
3. Analyzing the fraction of manipulable preferences with changing voter count. We fixed the number of candidates to `5` and number of samples to `5000` and varied the voter count as `[1, 2, 5, 10, 20, 50, 100, 200]`

### Justifications

### Algorithm for checking manipulability



## Experimental Results


## Inference