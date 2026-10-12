Write meeting notes in {target_language} for a long meeting from the documents. Each document is either the notes already written for one part of the meeting (its time range and slides are in its title) or one slide. In the part notes, every statement ends with markers such as [seg-3] or [slide-4] that say where it comes from.

Use exactly these five section markers, in English and in this order, each on its own line, whatever language the notes are in:

## SUMMARY
## DECISIONS
## ACTION_ITEMS
## OPEN_QUESTIONS
## KEY_POINTS

Rules:
- Under SUMMARY, write 2 to 4 sentences about the whole meeting. Under the other markers, write one item per line starting with "- ".
- Keep every decision, action item, open question and figure from the parts, one per item. Merge items only when two parts state the same thing.
- Under KEY_POINTS, list only figures: one number per item with what it measures, in one short sentence. Leave out remarks without a number.
- After each statement you take from a part, copy its markers, such as [seg-3], exactly as they appear.
- Every claim must be supported by the documents. Do not add anything they don't state.
- Keep names and numbers exactly as they appear in the documents. When a slide and a part disagree on a name or number, use the slide.
- Write each action item on one line, in exactly this form, with the labels "owner" and "due" in English and "?" when it is not stated. Copy a due date as it was said; never work out a calendar date:
  - [owner: Priya] [due: 20 October] Prepare the store screenshots
  - [owner: ?] [due: ?] Update the release notes
- If a section has nothing, write "- none" in English.

Write the notes in {target_language}, keeping the five English section markers.
