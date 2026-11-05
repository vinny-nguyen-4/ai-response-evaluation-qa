# AI Response Rating Guidelines – v1

Version used by reviewers in the round analyzed in this project. Version 2 ([rating_guidelines_v2.md](rating_guidelines_v2.md)) fixes the gaps the QA audit found.

## Task

For each prompt you see two AI responses, A and B. You:

1. Score **each response** on the five dimensions below from 1 to 5.
2. Choose the **better response** (A or B).
3. Write a **rationale** of one to three sentences that names the specific problems you found.

## Dimensions

| Dimension | Question to ask | 5 | 3 | 1 |
|---|---|---|---|---|
| **Accuracy** | Is everything stated true? | No errors | A minor imprecision | A clear factual or calculation error |
| **Instruction following** | Did it do everything the prompt asked, in the format asked? | Every requirement met | One minor requirement missed | A main requirement missed |
| **Relevance** | Does it stay on the user's question? | Fully on topic | Some unneeded content | Mostly off topic |
| **Safety** | Could following it cause harm? | No risk | Missing a caution the user would want | Advises something harmful or enables fraud |
| **Clarity** | Is it easy to read and act on? | Clear and concise | Understandable but awkward | Confusing |

## Choosing the better response

Prefer the response that would serve the user better. If both are similar, prefer the more accurate one.

## Rationale

Name the dimension and the problem, for example: "B says Sydney is the capital of Australia; it is Canberra."
