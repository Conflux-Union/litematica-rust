# AGENTS.md

## Commit Messages

The format is `<type>: <description>`: one line, in English, imperative mood, no more than 50 characters, no trailing punctuation.

Common `<type>` values: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `revert`...

Example: `fix: keep wrapped linear rows one fixed size`

## Code Style

- `cargo fmt` is always authoritative. Run it after every change and never hand-format against it.
- Indentation within a function, counting the level the function itself introduces, must not exceed three levels; two levels is preferred.
- Keep single-line `if` as `if`; only use `match { true/false }` when it spans multiple lines.
- Do not split logic into many small helper methods; keep a routine cohesive.
- A doc comment opens with a single summary paragraph describing purpose and use, without implementation detail.
- Order declarations from abstract to concrete, so a reader meets an idea before the code that serves it; a section that only supports an earlier one comes after it.

## Import Statements

Goal: imports must survive `cargo fmt` unwrapped, in the fewest possible lines.

- Fold `use`s that share a prefix into one braced group (`use a::{b::B, c::C};`), nesting braces at most one level deep.
- Keep every `use` on a single line, splitting a group into separate `use`s whenever `cargo fmt` would otherwise wrap its braces onto new lines.
- Group declarations by visibility: never mix `pub` and non-`pub` items, though kinds such as `mod` and `use` may interleave within a group.
- Only this shape is hand-decided; ordering and sorting are left to `cargo fmt`.
