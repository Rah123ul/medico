import React, { useEffect } from 'react';
import HeroSection from './HeroSection';
import FeaturesSection from './FeaturesSection';
import SetupSection from './SetupSection';
import SessionsShowcase from './SessionsShowcase';

export default function OnboardingScreen({ onStartSession, multiplayer }) {
  useEffect(() => {
    // Parallax effect on Hero
    const handleScroll = () => {
      const hero = document.querySelector('.hero-image');
      if (hero) {
        const scrollY = window.scrollY;
        hero.style.transform = `scale(1.1) translateY(${scrollY * 0.15}px)`;
      }
    };
    window.addEventListener('scroll', handleScroll);

    // GSAP Animations
    let ctx;
    if (window.gsap && window.ScrollTrigger) {
      window.gsap.registerPlugin(window.ScrollTrigger);
      
      // Delay initialization slightly to let the DOM paint and calculate heights
      setTimeout(() => {
        ctx = window.gsap.context(() => {
          // Feature cards stagger
          window.gsap.fromTo('.feature-card', 
            { y: 40, autoAlpha: 0 },
            {
              scrollTrigger: {
                trigger: '.features-section',
                start: 'top 85%',
              },
              y: 0,
              autoAlpha: 1,
              duration: 0.6,
              stagger: 0.1,
              ease: 'power2.out'
            }
          );

          // Session cards stagger
          window.gsap.fromTo('.session-card', 
            { x: 30, autoAlpha: 0 },
            {
              scrollTrigger: {
                trigger: '.sessions-showcase',
                start: 'top 85%',
              },
              x: 0,
              autoAlpha: 1,
              duration: 0.6,
              stagger: 0.1,
              ease: 'power2.out'
            }
          );
          
          // Setup panels slide
          window.gsap.fromTo('.onboard-card', 
            { y: 30, autoAlpha: 0 },
            {
              scrollTrigger: {
                trigger: '.setup-section',
                start: 'top 85%',
              },
              y: 0,
              autoAlpha: 1,
              duration: 0.6,
              stagger: 0.2,
              ease: 'power2.out'
            }
          );
          
          ScrollTrigger.refresh();
        });
      }, 100);
    }

    return () => {
      window.removeEventListener('scroll', handleScroll);
      // Clean up scroll triggers and context
      if (ctx) ctx.revert();
      else if (window.ScrollTrigger) {
        window.ScrollTrigger.getAll().forEach(t => t.kill());
      }
    };
  }, []);

  return (
    <section id="onboardingScreen" className="screen screen--onboarding screen--entering">
      <HeroSection onScrollToSetup={() => document.getElementById('setupSection').scrollIntoView({ behavior: 'smooth' })} />
      <FeaturesSection />
      <SetupSection onStartSession={onStartSession} multiplayer={multiplayer} />
      <SessionsShowcase />
    </section>
  );
}
