---
title: "Experience"
permalink: /experience/
author_profile: false
layout: single
classes:
  - experience-page
  - content-page
---

{% for entry in site.data.experience %}
<div class="exp-entry">
  <div class="exp-entry__header">
    <div class="exp-entry__title-block">
      <span class="exp-entry__role">{{ entry.role }}</span>
      <span class="exp-entry__org">
        {% if entry.org_url %}<a href="{{ entry.org_url }}">{{ entry.org }}</a>{% else %}{{ entry.org }}{% endif %}{% if entry.location %} &middot; {{ entry.location }}{% endif %}
      </span>
    </div>
    <span class="exp-entry__period">{{ entry.period }}</span>
  </div>

  <div class="exp-entry__body">
    {% for project in entry.projects %}
    <h3 class="exp-entry__project">{{ project.name }}</h3>
    <ul class="exp-entry__bullets">
      {% for bullet in project.bullets %}
      <li>{{ bullet | markdownify | remove: '<p>' | remove: '</p>' | strip }}</li>
      {% endfor %}
    </ul>
    {% endfor %}
  </div>

  <div class="exp-entry__meta">
    {% if entry.open_source %}
    <div class="exp-entry__links">
      {% for link in entry.open_source %}<a href="{{ link.url }}" class="res-link" target="_blank" rel="noopener">↗ {{ link.label }}</a>{% endfor %}
    </div>
    {% endif %}
    {% if entry.tech %}
    <div class="tag-list">
      {% for tag in entry.tech %}<span class="tag">{{ tag }}</span>{% endfor %}
    </div>
    {% endif %}
    {% if entry.mentors %}
    <p class="exp-entry__mentors">Mentors: {% for m in entry.mentors %}{% if m.url %}<a href="{{ m.url }}">{{ m.name }}</a>{% else %}{{ m.name }}{% endif %}{% unless forloop.last %}, {% endunless %}{% endfor %}</p>
    {% endif %}
  </div>
</div>
{% unless forloop.last %}<hr class="section-divider">{% endunless %}
{% endfor %}
