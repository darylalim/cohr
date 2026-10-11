You are reading one presentation slide. Return only a JSON object, with no other text, using exactly these keys:

{"title": string, "text_md": string, "tables_md": [string], "chart_notes": string, "visual_notes": string}

- "title": the slide title, verbatim.
- "text_md": all other text on the slide, verbatim, in reading order, as Markdown (bullets as "- ").
- "tables_md": each table as a Markdown table, with every cell copied exactly (keep numbers, units and symbols).
- "chart_notes": what any chart shows (axes, series, values); "" if there is no chart.
- "visual_notes": anything else important that isn't text; "" if nothing.
