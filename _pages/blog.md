---
title: "Blog"
permalink: /blog/
layout: single
author_profile: false
classes:
  - content-page
---

{% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% for year_group in posts_by_year %}
<h2 class="pub-section-heading">{{ year_group.name }}</h2>

{% for post in year_group.items %}
{% assign post_href = post.link | default: post.url %}
{% assign is_external = false %}
{% if post.link %}{% assign is_external = true %}{% endif %}
<div class="blog-entry">
  <div class="blog-entry__header">
    <h3 class="blog-entry__title">
      <a href="{% if is_external %}{{ post.link }}{% else %}{{ post.url | relative_url }}{% endif %}"{% if is_external %} target="_blank" rel="noopener"{% endif %}>{{ post.title }}{% if is_external %} <span style="font-size:0.75em;opacity:0.6;">↗</span>{% endif %}</a>
    </h3>
    <span class="exp-entry__period">{{ post.date | date: "%b %-d, %Y" }}</span>
  </div>
  {% if post.excerpt %}
  <p class="blog-entry__excerpt">{{ post.excerpt | strip_html | truncatewords: 45 }}</p>
  {% endif %}
  {% if post.tags %}
  <div class="tag-list">{% for tag in post.tags %}<span class="tag">{{ tag }}</span>{% endfor %}</div>
  {% endif %}
</div>
{% unless forloop.last %}<hr class="section-divider">{% endunless %}
{% endfor %}

{% endfor %}
