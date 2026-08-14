/* Where the API lives.
 *
 * Was a const declared separately at the top of three pages, which the shared
 * panels then read out of whichever page happened to have loaded them. That is
 * the dependency that made accounts.js and steam-panel.js impossible to import:
 * they needed something only their caller provided. */
export const API = window.location.origin + '/api';
