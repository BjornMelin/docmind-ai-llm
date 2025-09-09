---
id: comprehensive-analysis
name: Comprehensive Document Analysis
description: Summary, key insights, action items, open questions
tags: [analysis, default]
required:
  - context
  - tone
  - role
defaults:
  tone: { description: "Use a professional, objective tone." }
  role: { description: "Act as a helpful assistant." }
version: 1
---
{% chat role="system" %}
You are {{ role.description }}. {{ tone.description }}
{% endchat %}

{% chat role="user" %}
Context:
{{ context }}

Tasks:
- Provide a concise summary
- Extract key insights
- List action items
- Raise open questions
{% endchat %}
