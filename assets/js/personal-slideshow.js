document.addEventListener('DOMContentLoaded', function() {
  const slideshows = document.querySelectorAll('[data-personal-slideshow]');

  slideshows.forEach(function(slideshow) {
    const slides = Array.from(slideshow.querySelectorAll('[data-slide-item]'));
    const dots = Array.from(slideshow.querySelectorAll('[data-slide-to]'));
    const prevButton = slideshow.querySelector('[data-slide-prev]');
    const nextButton = slideshow.querySelector('[data-slide-next]');

    if (slides.length <= 1) {
      return;
    }

    let currentIndex = slides.findIndex(function(slide) {
      return slide.classList.contains('is-active');
    });
    if (currentIndex < 0) {
      currentIndex = 0;
      slides[0].classList.add('is-active');
      if (dots[0]) {
        dots[0].classList.add('is-active');
      }
    }

    function updateSlide(nextIndex) {
      slides[currentIndex].classList.remove('is-active');
      if (dots[currentIndex]) {
        dots[currentIndex].classList.remove('is-active');
      }

      currentIndex = (nextIndex + slides.length) % slides.length;

      slides[currentIndex].classList.add('is-active');
      if (dots[currentIndex]) {
        dots[currentIndex].classList.add('is-active');
      }
    }

    if (prevButton) {
      prevButton.addEventListener('click', function() {
        updateSlide(currentIndex - 1);
      });
    }

    if (nextButton) {
      nextButton.addEventListener('click', function() {
        updateSlide(currentIndex + 1);
      });
    }

    dots.forEach(function(dot) {
      dot.addEventListener('click', function() {
        const target = parseInt(dot.getAttribute('data-slide-to'), 10);
        if (!Number.isNaN(target)) {
          updateSlide(target);
        }
      });
    });

    let autoPlayTimer = setInterval(function() {
      updateSlide(currentIndex + 1);
    }, 5000);

    function pauseAutoPlay() {
      clearInterval(autoPlayTimer);
    }

    function resumeAutoPlay() {
      pauseAutoPlay();
      autoPlayTimer = setInterval(function() {
        updateSlide(currentIndex + 1);
      }, 5000);
    }

    slideshow.addEventListener('mouseenter', pauseAutoPlay);
    slideshow.addEventListener('mouseleave', resumeAutoPlay);
    slideshow.addEventListener('focusin', pauseAutoPlay);
    slideshow.addEventListener('focusout', resumeAutoPlay);
  });
});
