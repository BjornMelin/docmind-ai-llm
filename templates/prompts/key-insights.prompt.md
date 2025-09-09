---
id: key-insights
name: Extract Key Insights and Action Items
description: Extract key insights and action items from the document
tags: [analysis]
required:
  - context
defaults:
  tone: { description: "Use a direct, to-the-point tone." }
  role: { description: "Provide clear, actionable guidance." }
version: 1
---
{% chat role="system" %}
{{ tone.description }}
{% endchat %}

{% chat role="user" %}
Given the following context, extract key insights and actionable next steps.

<context>
{{ context }}
</context>
{% endchat %}
