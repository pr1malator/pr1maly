/* The two things the shared panels need back from the page that loaded them.
 *
 * accounts.js has to refill the bulk-upload account list, and steam-panel.js
 * has to switch the settings modal to its own tab. Both are page-specific:
 * populateBulkAccountSelector and switchSettingsTab have genuinely different
 * bodies on the three pages that define them, so there is nothing to import.
 *
 * A page fills these in as it loads:
 *
 *     Object.assign(hooks, { switchSettingsTab, populateBulkAccountSelector });
 *
 * That is a dependency stated out loud, where the alternative was a shared file
 * calling a function it hoped its caller had defined. */
export const hooks = {};
