Write meeting notes in {target_language} from the documents. Each document is either one part of the meeting (its time range and slides are in its title) or one slide.

Use exactly these five section markers, in English and in this order, each on its own line, whatever language the notes are in:

## SUMMARY
## KEY_POINTS
## DECISIONS
## ACTION_ITEMS
## OPEN_QUESTIONS

Rules:
- Under SUMMARY, write 2 to 4 sentences. Under the other markers, write one item per line starting with "- ".
- Every claim must be supported by the documents. Do not add anything they don't state.
- Keep names and numbers exactly as they appear in the documents. When a slide and the transcript disagree on a name or number, use the slide.
- Write each action item on one line, in exactly this form, with the labels "owner" and "due" in English and "?" when it is not stated. Copy a due date as it was said; never work out a calendar date:
  - [owner: Priya] [due: 20 October] Prepare the store screenshots
  - [owner: ?] [due: ?] Update the release notes
- If a section has nothing, write "- none" in English.

Write the notes in {target_language}, keeping the five English section markers.
