"""Survey-derived ethnic and religious target sets.

Reproduce the selections and frequencies with scripts/derive_identity_targets.py.
"""

from .research_design import PairedStudy


def _study(primes, labels, targets, sample, religion=False):
    if religion:
        design = "Studies 2–3"
        procedure = (
            "Study 2 elicited one fast response twice per group. Study 3 elicited three responses. "
            "For each word we calculated the difference in respondent shares between the groups within "
            "each study, then averaged the two differences with equal weight. "
        )
    else:
        design = "Study 1"
        procedure = (
            "Participants could give six descriptions per group. We calculated the difference in the "
            "share of respondents naming each word for the two groups. "
        )
    context = (
        f"Participants described social groups in American society. {sample} {procedure}"
        "Repeated words count once per respondent and group. These explorer lists take up to 25 "
        "single-word responses on each side with a positive difference and at least two respondents "
        "on that side. Ties favor more respondents on that side, then alphabetical order. "
        "Spelling and common plurals are normalized. Both groups’ labels and ties between groups "
        "are excluded. Unavailable graph words are reported in coverage. These are respondents’ "
        "associations, which include derogatory terms. Some words were named by only two people."
    )
    if not religion:
        ambiguous = "black and white can both" if "black" in primes else "white can"
        context += f" In these graphs, {ambiguous} evoke colors as well as social groups."
    return PairedStudy(
        primes=primes,
        labels=labels,
        targets=tuple(tuple(words.split()) for words in targets),
        citation=f"Nicolas, Bai & Fiske (2022), {design}",
        url="https://www.nicolaslab.org/files/SSCM.pdf",
        data_url="https://osf.io/74rax/",
        evidence_url="/assets/evidence/identity-targets.csv",
        context=context,
    )


STUDIES = {
    "religion": {
        "Christians vs. Muslims · US survey": _study(
            ("christian", "muslim"), ("Christians", "Muslims"),
            (
                "good hypocritical caring judgmental dumb conservative faithful generous holy "
                "trusting fearful giving happy helpful rigid devout closed-minded strong uptight "
                "follower honest loyal god jesus naive",
                "terrorist religious foreign different misunderstood violent peaceful scary "
                "dangerous brown human kind dedicated radical arab people friendly allah angry "
                "backwards killer praying worrisome nice hateful",
            ),
            "Study 2 had 88 respondents for Christians and 98 for Muslims. "
            "Study 3 had 25 and 19, respectively.",
            religion=True,
        ),
        "Christians vs. Atheists · US survey": _study(
            ("christian", "atheist"), ("Christians", "Atheists"),
            (
                "religious hypocritical caring judgmental good conservative faithful devout kind "
                "generous holy trusting dumb fearful giving helpful rigid happy loving closed-minded "
                "compassionate follower loyal nice uptight",
                "smart intelligent normal logical intellectual questioning scientist thoughtful "
                "people intolerant rational bad independent mean reasonable skeptical nonbeliever "
                "thinker evil angry arrogant different focused godless open",
            ),
            "Study 2 had 88 respondents for Christians and 86 for atheists. "
            "Study 3 had 25 and 22, respectively.",
            religion=True,
        ),
        "Christians vs. Jewish people · US survey": _study(
            ("christian", "jewish"), ("Christians", "Jewish people"),
            (
                "good hypocritical caring judgmental dumb devout generous holy trusting fearful "
                "giving happy helpful rigid conservative faithful kind closed-minded compassionate "
                "follower friendly uptight loyal nice idiot",
                "rich smart cheap religious greedy wealthy frugal stingy dedicated people funny "
                "intelligent interesting successful thrifty normal selfish banker cool holocaust "
                "money resilient religion",
            ),
            "Study 2 had 88 respondents for Christians and 88 for Jewish people. "
            "Study 3 had 25 and 25, respectively.",
            religion=True,
        ),
        "Christians vs. Buddhists · US survey": _study(
            ("christian", "buddhist"), ("Christians", "Buddhists"),
            (
                "good hypocritical caring conservative dumb faithful judgmental generous trusting "
                "devout fearful giving helpful rigid holy happy kind closed-minded compassionate "
                "follower honest loyal strong uptight loving",
                "peaceful calm quiet spiritual asian patient religious peace serene smart zen "
                "disciplined monk wise centered enlightened meditative pacifist relaxed nice "
                "religion focused",
            ),
            "Study 2 had 88 respondents for Christians and 81 for Buddhists. "
            "Study 3 had 25 and 27, respectively.",
            religion=True,
        ),
    },
    "ethnicity": {
        "Black vs. White · US survey": _study(
            ("black", "white"), ("Black people", "White people"),
            (
                "loud athletic strong dark tall funny beautiful oppressed poor aggressive basketball "
                "human musical sad violent active religious hardworking angry bad boisterous city "
                "crime criminal dangerous",
                "nice privileged boring educated average proud racist rich christian entitled "
                "powerful honest generous selfish conceited dominant driven dumb equal free majority "
                "motivated plain polite power",
            ),
            "The survey had 61 respondents for Black people and 47 for White people.",
        ),
        "Asian vs. White · US survey": _study(
            ("asian", "white"), ("Asian people", "White people"),
            (
                "smart short intelligent small rude skinny friendly strict quiet kind eastern fast "
                "industrious interesting minority pressured studious thin young healthy successful "
                "traditional creative wise driven",
                "nice privileged average boring racist honest proud christian entitled generous lazy "
                "powerful selfish american arrogant conceited dominant dumb equal fine free fun "
                "majority motivated plain",
            ),
            "The survey had 51 respondents for Asian people and 47 for White people.",
        ),
        "Hispanic vs. White · US survey": _study(
            ("hispanic", "white"), ("Hispanic people", "White people"),
            (
                "brown dark friendly family poor loud religious fun diligent illegal immigrant "
                "mexico spanish beans catholic dancer dirty fast loving mexican strict thug tough "
                "violent intelligent",
                "nice smart privileged average boring educated rich honest proud racist christian "
                "entitled powerful selfish generous american arrogant conceited dominant driven dumb "
                "equal fine free majority",
            ),
            "The survey had 48 respondents for Hispanic people and 47 for White people.",
        ),
    },
}
