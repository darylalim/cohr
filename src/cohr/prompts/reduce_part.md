Write notes in {target_language} for one part of a long meeting from the documents. Each document is either one section of this part (its time range and slides are in its title) or one slide.

Use exactly these five section markers, in English and in this order, each on its own line, whatever language the notes are in:

## SUMMARY
## DECISIONS
## ACTION_ITEMS
## OPEN_QUESTIONS
## KEY_POINTS

Rules:
- Under SUMMARY, write 2 or 3 sentences. Under the other markers, write one item per line starting with "- ".
- Under DECISIONS, ACTION_ITEMS and OPEN_QUESTIONS, list every one the documents state, one per item.
- Under KEY_POINTS, list only figures: one number per item with what it measures, in one short sentence, such as "- Weekly active users in Spain: 5,200, from 4,800 in May." Leave out remarks without a number.
- Every claim must be supported by the documents. Do not add anything they don't state.
- Keep names and numbers exactly as they appear in the documents. When a slide and the transcript disagree on a name or number, use the slide.
- Write each action item on one line, in exactly this form, with the labels "owner" and "due" in English and "?" when it is not stated. Copy a due date as it was said; never work out a calendar date:
  - [owner: Priya] [due: 20 October] Prepare the store screenshots
  - [owner: ?] [due: ?] Update the release notes
- If a section has nothing, write "- none" in English.

Write the notes in {target_language}, keeping the five English section markers.
