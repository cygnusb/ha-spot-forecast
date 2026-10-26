# Brand Icons

Since Home Assistant 2026.3.0 an integration ships its own brand images in
`custom_components/spot_forecast/brand/` (icon.png 256×256, icon@2x.png 512×512,
logo.png 667×128, logo@2x.png 1334×256). The sizes follow the image specification of
`home-assistant/brands`: square icons, logos with a shortest side of 128 (256 for @2x),
all trimmed to their content; the SVG view boxes are cut to the artwork for that.
Older Home Assistant versions (before 2026.3.0) show no icon for the integration.
Local brand images override the brands CDN,
so nothing is submitted to `home-assistant/brands` (that repo no longer accepts custom
integration icons, see https://developers.home-assistant.io/blog/2026/02/24/brands-proxy-api).

Source artwork:

- `assets/icon.svg`: the mark. A day-ahead price step line (horizon blue `#2a78d6`)
  runs from the known prices into the forecast; from "now" on, a hatched peak-risk band
  (risk red `#d03b3b`) rises above it to the P90 edge, as on the forecast card. Flat
  colours on a transparent background, so it works on light and dark themes; at 32 px
  the hatching reads as a red tint under the red edge.
- `assets/logo.svg`: the mark plus the wordmark "Spot Forecast" in Archivo Bold at
  width 75 (SIL Open Font License 1.1), converted to outlines so it renders the same
  everywhere without the font. To change the wordmark, set it in that font and cut,
  then convert the text to paths again.

Regenerate the PNGs with `rsvg-convert`:

```sh
cd assets
rsvg-convert -w  256 -h 256 icon.svg -o ../custom_components/spot_forecast/brand/icon.png
rsvg-convert -w  512 -h 512 icon.svg -o ../custom_components/spot_forecast/brand/icon@2x.png
rsvg-convert -w  667 -h 128 logo.svg -o ../custom_components/spot_forecast/brand/logo.png
rsvg-convert -w 1334 -h 256 logo.svg -o ../custom_components/spot_forecast/brand/logo@2x.png
cp ../custom_components/spot_forecast/brand/icon.png ../custom_components/spot_forecast/brand/logo.png .
```

`tests/test_brand.py` checks the sizes and that `assets/` holds the same renders.
