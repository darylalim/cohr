Below are numbered sentences from a meeting transcript. Each one names a person. Some give that person a task: someone asks them to do something, or says they will do it. Others only talk to or about the person without giving them work, such as a question about the call itself or a remark about where they will be.

For each sentence that gives the person a task, write the task in {target_language}: a short phrase that starts with a verb and says what they will do, without the person's name and without the deadline. Copy every number digit for digit. For "due", copy the words the sentence says for when it is due, or use null when it says none; never turn them into a calendar date and never add a year. Use only what the sentence says.

Return only a JSON object, with no other text, using exactly this shape, one entry per sentence:

{"tasks": [{"line": number, "task": string or null, "due": string or null}]}

Use "task": null for a sentence that gives no task.

Sentences:
{sentences}

Write every "task" and "due" in {target_language}. Return only the JSON object.
