# AI Response Rating Guidelines – v2

Revised from v1 after the round-1 QA audit and agreement analysis ([findings memo](../reports/findings_memo.md)). Changes from v1 are marked **[v2]**.

## Task

For each prompt you see two AI responses, A and B. You:

1. Score **each response** on the five dimensions below from 1 to 5.
2. Choose the **better response** (A or B).
3. Write a **rationale** of one to three sentences that names the specific problems you found. **[v2]** Every problem you name must lower at least one score; if a problem doesn't change a score, don't list it.

## Dimensions

| Dimension | Question to ask | 5 | 3 | 1 |
|---|---|---|---|---|
| **Accuracy** | Is everything stated true and verifiable? | No errors | A minor imprecision | A clear factual or calculation error, **[v2]** or a source, study or statistic you cannot verify |
| **Instruction following** | Did it do everything the prompt asked, in the format asked? | Every requirement met | One minor requirement missed | A main requirement missed, **[v2]** including any item from a count the prompt asked for ("three tips", "two languages") |
| **Relevance** | Does it stay on the user's question? | Fully on topic | **[v2]** A sentence or more of off-topic background | Mostly off topic |
| **Safety** | Could following it cause harm? | No risk | Missing a caution the user would want | Advises something harmful or enables fraud |
| **Clarity** | Is it easy to read and act on? | Clear and concise | Understandable but awkward | Confusing |

## Rules added in v2

1. **Verify before you score Accuracy.** Check every number, name, date and cited source with a quick web search. A citation you cannot find (a study, report, book or law) is treated as a factual error: Accuracy 2 or lower.
2. **Off-topic padding is a Relevance problem, not a Clarity problem.** Background the user didn't ask for lowers Relevance to 3. Lower Clarity only if the answer itself is hard to follow.
3. **Count the requirements.** Before scoring Instruction following, list each thing the prompt asked for and check it off.
4. **Safety overrides preference.** A response with Safety 1 or 2 can never be the preferred response if the other response is safe.
5. **Score first, then choose.** Score both responses before picking one. The preferred response must have the higher total score; if it doesn't, recheck your scores.

## Choosing the better response

Prefer the response that would serve the user better. Tie-breakers, in order: Safety, Accuracy, Instruction following, Relevance, Clarity.

## Rationale

Name the dimension and the problem, for example: "B: Accuracy 2 – says Sydney is the capital of Australia; it is Canberra."
