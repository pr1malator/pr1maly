/* theme.js stays a classic script: it has to run before the page paints, or a
   light-theme user sees a flash of the dark one, and the landing site under
   docs/ loads the same file the same way. A classic script cannot import, so
   the one thing its markup needs — the theme menu button — is registered here
   instead, by the module side.
 */
import { registerActions } from './actions.js';

registerActions({
  toggleThemeMenu: (...args) => window.toggleThemeMenu(...args),
});
