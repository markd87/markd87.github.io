---
layout: default
title: Snippets
---

{% assign sorted_snippets = site.snippets | reverse %}

<h2>Snippets</h2>
{%- for snippet in sorted_snippets -%}
<div class="snippet">
  <h3><a href="{{ snippet.url }}">{{ snippet.title }}</a></h3>
          {%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
          <time class="dt-published" datetime="{{ snippet.date | date_to_xmlschema }}" itemprop="datePublished">
              {{ snippet.date | date: date_format }}
          </time>  
</div>
{% endfor %}
