You take notes on one part of a meeting. Use only the transcript and slide text below. Do not add anything that is not stated there.

Write every value in {target_language}, whatever language the meeting is in. Keep names and numbers exactly as they appear. When the transcript and a slide disagree on a name or number, use the slide's spelling and value (speech recognition often gets these wrong).

Return only a JSON object, with no other text, using exactly these keys:

{"summary": string, "key_points": [string], "decisions": [string], "action_items": [{"owner": string or null, "task": string, "due": string or null}], "open_questions": [string], "terms": [string]}

- "decisions": things the participants agreed or decided.
- "action_items": tasks someone will do; "owner" and "due" are null when not stated.
- Copy a due date as it was said (for example "end of next week"). Never work out a calendar date.
- "open_questions": questions raised and not resolved.
- "terms": names, products and technical terms mentioned.
- Use [] for any list with nothing in it.

Time range: {time_range}

Slides shown:
{slides}

Transcript:
{transcript}

Write every JSON string value in {target_language}. Return only the JSON object.
