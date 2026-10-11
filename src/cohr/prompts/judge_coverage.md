You check whether meeting notes contain a fact. Below is one fact from the meeting and the claims from one section of the notes. The notes may be in a different language from the fact.

Answer true if at least one claim states the fact, in any language and in any wording. Answer false if no claim does, including when a claim states something different (another decision, another owner, another date or number).

Fact:
<<<
{fact}
>>>

Claims:
<<<
{claims}
>>>

Return only a JSON object, with no other text, in this form:

{"reason": "one short sentence", "covered": true or false}
