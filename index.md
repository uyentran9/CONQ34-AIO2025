---  

layout: page
title: "🎓 CONQ34 – AIO 2025 Blog"
permalink: /
---  

{:options parse_block_html="true" /}

<!-- Hero Banner -->
<div class="hero-landing">
  <img src="{{ '/assets/module6-week1/assets/img/brand/header-ai.jpg' | relative_url }}" alt="CONQ34 – AIO 2025 banner">
  <div class="hero-overlay">
    <h1>CONQ34 – AIO 2025</h1>
    <p>Học AI & hơn thế nữa 🚀</p>
  </div>
</div>

---  

Chào mừng đến với blog học AI của nhóm **CONQ34 – AIO 2025** 🌱  
Đây là nơi chúng mình chia sẻ kiến thức AI và các dự án của nhóm.   
Các bài viết mới nhất sẽ hiện ngay bên dưới.  

---  

## Posts

<div class="post-grid">
  {% assign posts = site.posts | where_exp: "p", "p.draft != true" %}
  {% for post in posts %}
  <a class="post-card" href="{{ post.url | relative_url }}">
    <div class="thumb-wrap">
      {% if post.image %}
        <img src="{{ post.image | relative_url }}" alt="{{ post.title | escape }}">
      {% else %}
        <!-- fallback nếu chưa có image -->
        <img src="{{ '/assets/module6-week1/BCE.png' | relative_url }}" alt="{{ post.title | escape }}">
      {% endif %}
    </div>

    <div class="meta">
      <div class="date">{{ post.date | date: "%b %d, %Y" }}</div>
      <h3 class="title">{{ post.title }}</h3>
      {% if post.excerpt %}
        <p class="excerpt">{{ post.excerpt | strip_html | truncate: 120 }}</p>
      {% endif %}
    </div>
  </a>
  {% endfor %}
</div>

