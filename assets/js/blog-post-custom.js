// Enhanced Blog Post JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Add Back to Top button
  const backToTopButton = document.createElement('a');
  backToTopButton.href = '#';
  backToTopButton.className = 'back-to-top';
  backToTopButton.setAttribute('aria-label', 'Back to top');
  backToTopButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
  document.body.appendChild(backToTopButton);

  // Show/hide Back to Top button based on scroll position
  window.addEventListener('scroll', function() {
    if (window.pageYOffset > 300) {
      backToTopButton.classList.add('visible');
    } else {
      backToTopButton.classList.remove('visible');
    }
  });

  // Smooth scroll to top when Back to Top button is clicked
  backToTopButton.addEventListener('click', function(e) {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });

  // Add heading anchor links hover effect
  const headings = document.querySelectorAll('.page__content h2, .page__content h3, .page__content h4, .page__content h5, .page__content h6');
  headings.forEach(function(heading) {
    heading.addEventListener('mouseenter', function() {
      const headerLink = heading.querySelector('.header-link');
      if (headerLink) {
        headerLink.style.opacity = 1;
      }
    });
    
    heading.addEventListener('mouseleave', function() {
      const headerLink = heading.querySelector('.header-link');
      if (headerLink) {
        headerLink.style.opacity = 0;
      }
    });
  });

  // Add estimated reading time progress indicator
  if (document.querySelector('.page__content')) {
    const progressIndicator = document.createElement('div');
    progressIndicator.className = 'reading-progress';
    progressIndicator.style.position = 'fixed';
    progressIndicator.style.top = '0';
    progressIndicator.style.left = '0';
    progressIndicator.style.height = '3px';
    progressIndicator.style.backgroundColor = 'var(--link-color)';
    progressIndicator.style.zIndex = '1000';
    progressIndicator.style.width = '0%';
    progressIndicator.style.transition = 'width 0.2s ease';
    document.body.appendChild(progressIndicator);

    window.addEventListener('scroll', function() {
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight - windowHeight;
      const scrollPosition = window.scrollY;
      const scrollPercentage = (scrollPosition / documentHeight) * 100;
      progressIndicator.style.width = scrollPercentage + '%';
    });
  }

  // Add smooth scrolling for TOC links
  const tocLinks = document.querySelectorAll('.toc__menu a');
  tocLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      
      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 20,
          behavior: 'smooth'
        });
        
        // Update URL hash without jumping
        history.pushState(null, null, targetId);
      }
    });
  });

  // Highlight current section in TOC based on scroll position
  const sections = document.querySelectorAll('h2[id], h3[id]');
  const tocItems = document.querySelectorAll('.toc__menu li');
  
  window.addEventListener('scroll', function() {
    let currentSection = '';
    
    sections.forEach(function(section) {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      
      if (window.scrollY >= sectionTop - 100) {
        currentSection = '#' + section.getAttribute('id');
      }
    });
    
    tocItems.forEach(function(item) {
      const link = item.querySelector('a');
      item.classList.remove('active');
      
      if (link && link.getAttribute('href') === currentSection) {
        item.classList.add('active');
      }
    });
  });
});