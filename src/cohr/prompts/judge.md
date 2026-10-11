You check meeting notes against their sources. Below is one claim from the notes and the source text it cites: transcript excerpts and slides from the meeting.

Decide whether the cited text supports the claim, using only the cited text (not what seems likely or true elsewhere):

- "yes": every fact in the claim is stated in the cited text or follows directly from it. Other wording, order, shortening and summarizing are fine.
- "partial": the main point is in the cited text, but at least one detail of the claim (a reason, an extra item, a second number, a date, an owner) is not in it, and nothing in the claim contradicts the cited text.
- "no": the main point is not in the cited text, or any detail contradicts it (a different number, name, owner, date or outcome).

Numbers must have the same value: "$42,500" and "42,500 dollars" match; "3,360" and "336" do not, and neither do "the 2nd" and "the 22nd".

The claim starts with the section of the notes it comes from. A decision must be something the cited text says was agreed or decided. An action item must be a task someone was asked or agreed to do. An open question must be a question someone asked or left unresolved. A summary or key point only needs its facts stated.

Cited text:
<<<
{cited}
>>>

Claim:
<<<
{claim}
>>>

Return only a JSON object, with no other text, in this form:

{"reason": "one short sentence", "supported": "yes" or "partial" or "no"}
