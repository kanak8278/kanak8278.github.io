---
title: "Personal"
permalink: /personal/
author_profile: true
classes:
  - personal-hub-page
---
{% assign hub = site.data.personal-hub %}

<section class="personal-feed">
  {% for item in hub.feed %}
  <article class="personal-feed-item">
    <div class="personal-feed-meta">
      <code>{{ item.date }}</code>
      <span class="personal-chip">{{ item.kind }}</span>
      {% if item.city %}
        <span class="personal-muted">{{ item.city }}</span>
        {% if item.map_url %}
          <a class="personal-location-link" href="{{ item.map_url }}" target="_blank" rel="noopener noreferrer" aria-label="Open {{ item.city }} in Google Maps">
            <i class="fas fa-map-marker-alt" aria-hidden="true"></i>
          </a>
        {% elsif item.lat and item.lng %}
          <a class="personal-location-link" href="https://www.google.com/maps/search/?api=1&query={{ item.lat }},{{ item.lng }}" target="_blank" rel="noopener noreferrer" aria-label="Open {{ item.city }} in Google Maps">
            <i class="fas fa-map-marker-alt" aria-hidden="true"></i>
          </a>
        {% endif %}
      {% endif %}
    </div>

    <h2 class="personal-feed-title">{{ item.title }}</h2>
    <p>{{ item.text }}</p>

    {% if item.images and item.images.size > 0 %}
      <div class="personal-feed-slideshow" data-personal-slideshow>
        <div class="personal-feed-slides">
          {% for feed_image in item.images %}
            <figure class="personal-feed-slide{% if forloop.first %} is-active{% endif %}" data-slide-item>
              <img class="personal-feed-image" src="{{ feed_image | relative_url }}" alt="{{ item.title }} photo {{ forloop.index }}">
            </figure>
          {% endfor %}
        </div>

        {% if item.images.size > 1 %}
          <button type="button" class="personal-slide-control personal-slide-prev" data-slide-prev aria-label="Previous photo">
            &#8249;
          </button>
          <button type="button" class="personal-slide-control personal-slide-next" data-slide-next aria-label="Next photo">
            &#8250;
          </button>

          <div class="personal-slide-dots" aria-label="Photo selection">
            {% for feed_image in item.images %}
              <button type="button" class="personal-slide-dot{% if forloop.first %} is-active{% endif %}" data-slide-to="{{ forloop.index0 }}" aria-label="Show photo {{ forloop.index }}"></button>
            {% endfor %}
          </div>
        {% endif %}
      </div>
    {% elsif item.image %}
      <img class="personal-feed-image" src="{{ item.image | relative_url }}" alt="{{ item.title }}">
    {% endif %}

  </article>
  {% endfor %}
</section>

<script defer src="{{ '/assets/js/personal-slideshow.js' | relative_url }}"></script>
