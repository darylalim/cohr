You take notes on one part of a meeting. Use only the transcript and slide text below. Do not add anything that is not stated there.

Write every value in {target_language}, whatever language the meeting is in. Keep names exactly as they appear. Copy every number digit for digit with its separators, exactly as written in the transcript or slide: "12,345" stays "12,345" (never "123,450" or "1,234"). When the transcript and a slide disagree on a name or number, use the slide's spelling and value (speech recognition often gets these wrong).

Return only a JSON object, with no other text, using exactly these keys:

{"summary": string, "key_points": [string], "decisions": [string], "action_items": [{"owner": string or null, "task": string, "due": string or null}], "open_questions": [string], "terms": [string]}

- "key_points": one fact per item, in one short sentence: a figure with what it measures, or one other point that was made. Never join two facts in one item. Decisions and tasks go only in "decisions" and "action_items", never in a key point.
- "decisions": things the participants agreed or decided. A task someone will do goes only in "action_items", not here.
- "action_items": tasks someone will do. Take "owner" and "due" only from the sentence that gives that task; use null when that sentence doesn't state them. Never reuse another task's owner or due date.
- "due": copy the words that were said, for example "Friday", "end of next week" or "the 3rd of June". Never turn them into a calendar date: no numeric dates like 2025-06-03 and never add a year.
- Only list an action item when someone is asked or agrees to do something. Something that already happened (for example "the dashboard was updated on Monday") is not an action item.
- "open_questions": questions a participant asked and left unresolved in this part. A remark about a topic or a trend is not an open question, and neither is a question you would ask about it. Use [] when nobody asked one.
- "terms": names, products and technical terms mentioned.
- Use [] for any list with nothing in it.

Time range: {time_range}

Slides shown:
{slides}

Transcript:
{transcript}

Write every JSON string value in {target_language}. Return only the JSON object.
