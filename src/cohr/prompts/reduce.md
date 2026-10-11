Write meeting notes in {target_language} from the documents. Each document is one part of the meeting (its time range and slides are in its title) or one slide.

Use exactly these five section markers, in English and in this order, each on its own line, whatever language the notes are in:

## SUMMARY
## KEY_POINTS
## DECISIONS
## ACTION_ITEMS
## OPEN_QUESTIONS

Rules:
- Under SUMMARY, write 2 to 4 sentences. Under the other markers, write one item per line starting with "- ".
- Every claim must be supported by the documents. Do not add anything they don't state.
- Write each action item as: - [owner: <name or ?>] [due: <date or ?>] <task>
- If a section has nothing, write "- none".
