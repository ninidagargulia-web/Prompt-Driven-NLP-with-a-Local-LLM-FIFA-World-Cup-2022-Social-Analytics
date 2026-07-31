# Task 1 — Error Analysis

## Sentiment — Zero-shot failures: 25 / 100 (25.0%)

- True=**positive**, Pred=**neutral** — "Tomorrow is the beginning of the World Cup in #Qatar2022 this might be the last time we see these two icons on the biggest football ⚽️ stage"
- True=**positive**, Pred=**neutral** — "FIFA World Cup is one of the most-watched sporting events in the world—in 2018, nearly 3.6 billion people tuned in to watch the tournament. "
- True=**neutral**, Pred=**negative** — "Everyone's* been asking me for my #WorldCup2022 predictions so here they are! Winner: Brazil Final Four: Argentina/Brazil/Uruguay/England Go"
- True=**positive**, Pred=**neutral** — "Today's FIFA World Cup game: 🇶🇦 Qatar vs Ecuador 🇪🇨 It is finally the opening day of the World Cup! Will the hosts get the win, or will Ecua"
- True=**positive**, Pred=**neutral** — "So- the World Cup is here! What’s the first World Cup you remember watching? I remember Mexico 86 🇲🇽-especially the England Vs Argentina gam"
- True=**positive**, Pred=**neutral** — "2 things start today our new online booking for our Christmas camps and the #WorldCup2022 Can't wait to see you all #christmas #camps #activ"
- True=**positive**, Pred=**neutral** — "Expect a strong challenge from the South America favourites at the World Cup. Combined them for a series of name the finalists bets. Gone fo"
- True=**neutral**, Pred=**negative** — "France has adopted a different approach to the #WorldCup2022 in #Qatar than most of its European rivals. While the likes of England, Denmark"

## Sentiment — Few-shot failures: 37 / 100 (37.0%)

- True=**positive**, Pred=**neutral** — "My favorites to win The World Cup 2022 are: 1- Brazil. 2- France. 3- Spain. #WorldCup #WorldCup2022 #Qatar2022 #QatarWorldCup2022"
- True=**negative**, Pred=**positive** — "Ballon dor winner Karim Benzema is set to miss the world up after picking up a thigh injury❌ in Training. He is out of France 🇫🇷 26 man squa"
- True=**neutral**, Pred=**positive** — "Everyone's* been asking me for my #WorldCup2022 predictions so here they are! Winner: Brazil Final Four: Argentina/Brazil/Uruguay/England Go"
- True=**positive**, Pred=**neutral** — "Today's FIFA World Cup game: 🇶🇦 Qatar vs Ecuador 🇪🇨 It is finally the opening day of the World Cup! Will the hosts get the win, or will Ecua"
- True=**neutral**, Pred=**positive** — "Most ill disciplined teams leading into #WorldCup2022? Top 5 🇪🇨Ecuador (avg 2.46 cards per game) 🏴󠁧󠁢󠁷󠁬󠁳󠁿Wales (avg 2.38) 🇵🇱Poland (avg 2.38)"
- True=**neutral**, Pred=**negative** — "World Cup 2022 Prediction Semi finalists Argentina Senegal Uruguay Spain Winner Argentina Golden Boot Argentinian Dark Horse Denmark Surpris"
- True=**neutral**, Pred=**positive** — "🇶🇦 #FIFAWorldCupQatar2022, who will be crowned 🏆 Champion - Argentina 🥈 Second - Belgium 🥉 Third - Netherlands 🏅 Best Player - Lionel Messi "
- True=**positive**, Pred=**neutral** — "Expect a strong challenge from the South America favourites at the World Cup. Combined them for a series of name the finalists bets. Gone fo"

## Entity Extraction — structured-output JSON parse failures: 0 / 100 (0.0%)


## Observed Failure Patterns (fill in after reviewing the examples above)
- Sarcasm / banter often misread as positive when it is actually critical.
- Tweets mixing praise and criticism in one sentence confuse zero-shot more than few-shot.
- Very short tweets (<70 characters) have a higher unparsed/ambiguous rate.
- Structured-output prompt occasionally wraps JSON in commentary text, lowering the parse-valid rate versus a pure JSON-only instruction.

## Limitation Note
Topic-classification accuracy is measured against a keyword-based heuristic label, not human annotation — treat that accuracy figure as indicative only, and mention this explicitly in the Prompt Catalogue and final report.