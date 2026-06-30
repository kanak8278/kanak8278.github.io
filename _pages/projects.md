---
title: "Projects"
permalink: /projects/
author_profile: false
layout: single
classes:
  - content-page
---

{% for project in site.data.projects %}
<div class="proj-entry">
  <div class="proj-entry__header">
    <div class="proj-entry__title-block">
      <h2 class="proj-entry__title">{% if project.url %}<a href="{{ project.url }}" target="_blank" rel="noopener">{{ project.title }}</a>{% else %}{{ project.title }}{% endif %}</h2>
      {% if project.context %}<span class="proj-entry__context">{{ project.context }}</span>{% endif %}
    </div>
    {% if project.period %}<span class="exp-entry__period">{{ project.period }}</span>{% endif %}
  </div>

  <p class="proj-entry__desc">{{ project.description | markdownify | remove: '<p>' | remove: '</p>' | strip }}</p>

  <div class="proj-entry__meta">
    {% if project.links %}
    <div class="proj-entry__links">
      {% for link in project.links %}<a href="{{ link.url }}" class="res-link" target="_blank" rel="noopener">↗ {{ link.label }}</a>{% endfor %}
    </div>
    {% endif %}
    {% if project.tech %}
    <div class="tag-list">
      {% for tag in project.tech %}<span class="tag">{{ tag }}</span>{% endfor %}
    </div>
    {% endif %}
  </div>
</div>
{% unless forloop.last %}<hr class="section-divider">{% endunless %}
{% endfor %}
