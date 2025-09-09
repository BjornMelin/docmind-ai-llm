---
id: summary-open-questions
name: Summarize and Identify Open Questions
description: Summarize the document and identify open questions
tags: [summary]
required:
  - context
defaults:
  tone: { description: "Use a neutral, objective tone." }
version: 1
---
{% chat role="system" %}
{{ tone.description }}
{% endchat %}

{% chat role="user" %}
Summarize the content and list any open questions or areas requiring clarification.

<context>
{{ context }}
</context>
{% endchat %}
