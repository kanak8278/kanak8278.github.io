---
title: "Mech Interp Log"
permalink: /mechinterp-log/
layout: single
author_profile: false
classes:
  - content-page
---

A running log of self-study in AI alignment and mechanistic interpretability, working through [learnmechinterp.com](https://learnmechinterp.com/)'s curriculum. Entries are dated to when the work actually happened, not polished after the fact — read in order, oldest first.

{% assign log_posts = site.posts | where_exp: "post", "post.categories contains 'mechinterp-log'" | sort: "date" %}

{% for post in log_posts %}
<div class="blog-entry">
  <div class="blog-entry__header">
    <h3 class="blog-entry__title">
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
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
