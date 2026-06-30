---
title: "Publications"
permalink: /publications/
author_profile: false
layout: single
classes:
  - content-page
---

{% if site.data.publications.preprints %}
<h2 class="pub-section-heading">Preprints</h2>
{% for pub in site.data.publications.preprints %}
<div class="pub-entry">
  <span class="pub-entry__year">{{ pub.year }}</span>
  <div class="pub-entry__content">
    <div class="pub-entry__title">
      {% if pub.url %}<a href="{{ pub.url }}" target="_blank" rel="noopener">{{ pub.title }}</a>{% else %}{{ pub.title }}{% endif %}
      <span class="pub-entry__venue">{{ pub.venue }}</span>
    </div>
    <div class="pub-entry__authors">{{ pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip }}</div>
    {% if pub.links %}
    <div class="pub-entry__links">
      {% for link in pub.links %}<a href="{{ link.url }}" class="res-link" target="_blank" rel="noopener">↗ {{ link.label }}</a>{% endfor %}
    </div>
    {% endif %}
  </div>
</div>
{% endfor %}
{% endif %}

{% if site.data.publications.thesis %}
<h2 class="pub-section-heading">Thesis</h2>
{% for pub in site.data.publications.thesis %}
<div class="pub-entry">
  <span class="pub-entry__year">{{ pub.year }}</span>
  <div class="pub-entry__content">
    <div class="pub-entry__title">
      {% if pub.url %}<a href="{{ pub.url }}" target="_blank" rel="noopener">{{ pub.title }}</a>{% else %}{{ pub.title }}{% endif %}
      <span class="pub-entry__venue">{{ pub.venue }}</span>
    </div>
    <div class="pub-entry__authors">{{ pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip }}</div>
    {% if pub.links %}
    <div class="pub-entry__links">
      {% for link in pub.links %}<a href="{{ link.url }}" class="res-link" target="_blank" rel="noopener">↗ {{ link.label }}</a>{% endfor %}
    </div>
    {% endif %}
  </div>
</div>
{% endfor %}
{% endif %}

{% if site.data.publications.conferences %}
<h2 class="pub-section-heading">Conferences</h2>
{% for pub in site.data.publications.conferences %}
<div class="pub-entry">
  <span class="pub-entry__year">{{ pub.year }}</span>
  <div class="pub-entry__content">
    <div class="pub-entry__title">
      {% if pub.url %}<a href="{{ pub.url }}" target="_blank" rel="noopener">{{ pub.title }}</a>{% else %}{{ pub.title }}{% endif %}
      <span class="pub-entry__venue">{{ pub.venue }}</span>
    </div>
    <div class="pub-entry__authors">{{ pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip }}</div>
    {% if pub.links %}
    <div class="pub-entry__links">
      {% for link in pub.links %}<a href="{{ link.url }}" class="res-link" target="_blank" rel="noopener">↗ {{ link.label }}</a>{% endfor %}
    </div>
    {% endif %}
  </div>
</div>
{% endfor %}
{% endif %}
