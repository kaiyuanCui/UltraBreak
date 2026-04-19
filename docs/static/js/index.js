// Smooth scrolling is handled by CSS (html { scroll-behavior: smooth }).
// JS here: BibTeX copy button + Examples chip selector.

document.addEventListener('DOMContentLoaded', () => {
  // ── BibTeX copy ──
  const copyBtn = document.getElementById('copy-bibtex');
  if (copyBtn) {
    const src = document.getElementById('bibtex-content');
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(src.textContent).then(() => {
        const original = copyBtn.textContent;
        copyBtn.textContent = 'copied';
        setTimeout(() => { copyBtn.textContent = original; }, 1500);
      });
    });
  }

  // ── Examples chip selector ──
  const chips = document.querySelectorAll('.example-selector .chip');
  const cards = document.querySelectorAll('#examples .example-card');
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const target = chip.dataset.target;
      chips.forEach((c) => c.classList.toggle('is-active', c === chip));
      cards.forEach((card) => card.classList.toggle('hidden', card.id !== target));
    });
  });
});
