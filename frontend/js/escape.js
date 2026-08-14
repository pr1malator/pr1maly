/* Escaping text for insertion into HTML.
 *
 * The browser does the work: setting textContent and reading innerHTML back is
 * the one version of this that cannot be got subtly wrong. Note what it does
 * not do — quotes are left alone, so this is safe between tags and not inside
 * an attribute. For attributes use actionArgs from js/actions.js. */
export function esc(text) {
  const holder = document.createElement('div');
  holder.textContent = text;
  return holder.innerHTML;
}
