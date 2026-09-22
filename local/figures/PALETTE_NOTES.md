# Newsletter figure palette: derivation and validation

Source: paradigm3.org's own CSS (`https://www.paradigm3.org/_astro/BaseLayout.C2ql0Vsj.css`), cross-checked
against `https://p3humansonai.substack.com/` (same hex values inlined there: `#e9e5da` background, `#436f7a`
accent, `#2c2b28` foreground each appear repeatedly). Design tokens:

```
--color-accent:      #436f7a   (brand teal)
--color-accent-light: #d8d9d0
--color-background:   #e9e5da  (page cream)
--color-border:       #cbc8bf
--color-foreground:   #2c2b28  (charcoal, primary text)
--color-p3-grey:      #a6a49d
--color-p3-grey-dark: #6a6862
--color-p3-ink:       #1c1b19
--color-p3-offwhite:  #f5f4f1
--color-white:        #fff
--font-serif: Literata, Georgia, serif     (headings)
--font-sans:  system-ui, sans-serif        (body)
```

## Why the raw brand teal isn't used directly on chart marks

`#436f7a` in OKLCH is `L=0.514 C=0.052 H=215.4`. The dataviz skill's chroma floor for identity-carrying marks
is C >= ~0.10 -- below it a hue reads as gray and stops doing identity work. At H=215.4 the sRGB gamut only
clears C=0.10 once lightness rises to roughly L=0.65+ (checked by binary-searching the max in-gamut chroma at
each lightness step; the gamut boundary for this blue-teal hue is unusually narrow at mid lightness). So the
brand hue is kept exactly (H=215.4) and lifted to L=0.65, C=0.11 -- same color family, usable as a chart color.

## Final categorical set (light mode, white surface)

| Slot | Role | Hex | OKLCH |
|---|---|---|---|
| 1 | brand teal (lifted) | `#1b9fb9` | L=0.65 C=0.11 H=215.4 |
| 2 | terracotta | `#c1542d` | L=0.58 C=0.15 H=39.6 |
| 3 | plum | `#8b428d` | L=0.50 C=0.14 H=326.5 |
| 4 | ochre | `#a59c2a` | L=0.68 C=0.13 H=100 (shifted from the site's H=83 gold for CVD separation from terracotta) |

Sequential ramp (magnitude/heatmap figures): 8 steps of the same brand hue (H=215.4), light -> dark:
`#e3f4f8 #b9e4ee #84cfdd #43b4c9 #1b9fb9 #127e93 #0a5f70 #04424e`

## Validator output (`scripts/validate_palette.js`, dataviz skill)

**Adjacent pairs (bar/line/stack forms), surface `#ffffff`:**
```
[PASS] Lightness band       all 4 inside L 0.43-0.77
[PASS] Chroma floor         all 4 >= 0.1
[PASS] CVD separation       worst adjacent #c1542d<->#1b9fb9  dE 18.3 (deutan)
[PASS] Normal-vision floor  worst adjacent #8b428d<->#c1542d  dE 18.9
[WARN] Contrast vs surface  #a59c2a 2.84:1 (below 3:1)  -> relief required
-> ALL CHECKS PASS
```

**All-pairs (scatter / small-multiples), surface `#ffffff`:** also ALL PASS (worst pair
`#a59c2a<->#c1542d`, dE 9.8 deutan / 18.3 normal) -- the set is safe for up to all 4 slots even
in an all-pairs chart form, not just adjacent.

**Mitigation for the one WARN:** ochre (`#a59c2a`) never carries a value alone -- every figure that uses it
gives it a visible direct label, per the skill's "a contrast WARN obligates a relief channel" rule (not
optional). In practice ochre is used sparingly (slot 4 only, when a 4th series is genuinely needed).

Re-run: `node scripts/validate_palette.js "#1b9fb9,#c1542d,#8b428d,#a59c2a" --mode light --surface "#ffffff" [--pairs all]`
