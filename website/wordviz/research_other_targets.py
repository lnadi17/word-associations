"""Graph-compatible selections derived from two published nomination datasets.

Regenerate the word sets with scripts/derive_orientation_party_targets.py.
"""

from .research_design import PairedStudy


def _orientation(first, second, primes, nominations, first_words, second_words):
    counts = f"{nominations[0]} nominations for {first.lower()} and {nominations[1]} for {second.lower()}"
    return PairedStudy(
        primes=primes,
        labels=(first, second),
        targets=(tuple(first_words.split()), tuple(second_words.split())),
        citation="McGorray & Petsko (2024), Experiment 2",
        url="https://doi.org/10.1177/19485506231183467",
        data_url="https://osf.io/download/c3sv2/?version=1",
        evidence_url="/assets/evidence/orientation-targets.csv",
        context=(
            f"US participants named stereotypes they believed society associated with each group ({counts}). "
            "This explorer ranks each term by the difference in its share of all coded nominations for the two "
            "groups. Each side retains up to 25 terms with a positive difference and at least two nominations "
            "on that side. Ties favor more nominations, then alphabetical order. We use single-word codes "
            "present in the six original analysis graphs, taking the first label when all slash-separated alternatives are "
            "single words. Orientation labels are excluded. The study asked about gender-specific groups. "
            "The graph primes are single words without that gender qualifier. These terms describe reported "
            "social stereotypes, including stigmatizing ones."
        ),
    )


STUDIES = {
    "orientation": {
        "Gay men vs. straight men": _orientation(
            "Gay men", "Straight men", ("gay", "straight"), (342, 349),
            "feminine fashionable flamboyant promiscuous loud weak dramatic emotional annoying "
            "friendly colorful happy immoral kind sassy sensitive social funny thin creative disease "
            "energetic shallow dance domestic",
            "masculine strong aggressive sexual boring dominant muscular tough arrogant intelligent "
            "sexist unintelligent mean average conservative driven homophobic insensitive messy "
            "biased brave confident disloyal independent tall",
        ),
        "Lesbian women vs. straight women": _orientation(
            "Lesbian women", "Straight women", ("lesbian", "straight"), (343, 312),
            "masculine fat mean unattractive athletic aggressive tough feminist assertive strong "
            "dominant muscular confused different angry blunt arrogant big cool friendly opinionated "
            "outspoken sexist unusual loud",
            "feminine kind domestic weak attractive emotional intelligent bossy unintelligent white "
            "traditional deceptive needy boring christian disloyal educated hardworking independent "
            "jealous privileged sensitive thin entitled sexual",
        ),
        "Bisexual men vs. straight men": _orientation(
            "Bisexual men", "Straight men", ("bisexual", "straight"), (301, 349),
            "feminine promiscuous confused indecisive fashionable weak flamboyant emotional disloyal "
            "unusual disease attractive deceptive insecure secretive submissive unreliable colorful "
            "dramatic friendly greedy kind lisp lonely open-minded",
            "masculine strong aggressive boring dominant muscular tough unintelligent arrogant "
            "athletic mean sexist average conservative driven egotistical hardworking homophobic "
            "insensitive lazy messy intelligent biased independent tall",
        ),
        "Bisexual women vs. straight women": _orientation(
            "Bisexual women", "Straight women", ("bisexual", "straight"), (294, 312),
            "promiscuous masculine confused sexual strong indecisive unattractive adventurous immoral "
            "selfish deceptive aggressive ambitious arrogant colorful creative dirty fat happy "
            "immature insecure open-minded opinionated stubborn unusual",
            "feminine kind domestic emotional weak intelligent bossy white needy traditional boring "
            "christian educated entitled hardworking jealous privileged sensitive thin average basic "
            "caring dramatic gentle greedy",
        ),
        "Bisexual men vs. gay men": _orientation(
            "Bisexual men", "Gay men", ("bisexual", "gay"), (301, 342),
            "confused indecisive disloyal sexual unusual attractive deceptive secretive selfish "
            "submissive unreliable careless caring greedy lonely open-minded disease insecure lisp "
            "bad wealthy",
            "feminine loud fashionable flamboyant dramatic annoying funny happy immoral sensitive "
            "thin athletic weak promiscuous energetic shallow social friendly aggressive angry dance "
            "disgusting glittery liberal outlandish",
        ),
        "Bisexual women vs. lesbian women": _orientation(
            "Bisexual women", "Lesbian women", ("bisexual", "lesbian"), (294, 343),
            "promiscuous sexual confused indecisive attractive deceptive strong adventurous disloyal "
            "independent selfish unintelligent immoral ambitious colorful creative happy immature "
            "open-minded stubborn dirty insecure liberal arrogant fun",
            "masculine fat mean athletic feminist dominant assertive tough aggressive unattractive "
            "different muscular big caring cool friendly lazy outspoken sexist angry blunt loud",
        ),
    },
    "politics": {
        "US party supporters · 2016 and 2021": PairedStudy(
            primes=("democrat", "republican"),
            labels=("Democratic supporters", "Republican supporters"),
            targets=(
                tuple((
                    "liberal caring open-minded poor youth equality lazy friendly unrealistic generous "
                    "educated weak diversity supportive gullible dependent radical emotional understanding "
                    "urban hopeful social thoughtful peaceful donkey"
                ).split()),
                tuple((
                    "conservative rich white religious patriotism mean tradition trump right rigid angry "
                    "american hardworking economy elitist strong rural business crazy responsible red "
                    "southern ignorant realistic morality"
                ).split()),
            ),
            citation="Busby, Howat & Myers (2024), Appendix A replication data",
            url="https://doi.org/10.1017/psrm.2023.30",
            data_url="https://dataverse.harvard.edu/api/access/datafile/7234329",
            evidence_url="/assets/evidence/party-targets.csv",
            context=(
                "US respondents listed descriptors for supporters of each party. This adaptation uses the "
                "596 respondents in the 2016 file and 1,342 in the 2021 file who gave usable responses to both "
                "prompts. Each person counts once per word per party. A term must favor the same party in "
                "both waves and have at least ten nominators on that side overall. Each list takes the 25 "
                "largest pooled differences in respondent counts, breaking ties by number of nominators on "
                "that side, then alphabetically. Terms must occur in the six original analysis graphs. Party labels are excluded. "
                "Descriptors can refer to traits, demographics, symbols or issues. The rankings summarize "
                "respondents’ associations in these two surveys."
            ),
        ),
    },
}
