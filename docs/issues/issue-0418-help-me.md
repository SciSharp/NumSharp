# #418: help me

- **URL:** https://github.com/SciSharp/NumSharp/issues/418
- **State:** OPEN
- **Author:** @mak27arr
- **Created:** 2020-08-01T17:11:03Z
- **Updated:** 2020-08-02T00:52:38Z
- **Assignees:** @Nucs

## Description

what im do wrong np.meshgrid return only one NDArray, second always null

(scales, ratios) = np.meshgrid(np.array(scales), np.array(ratios));
            scales = scales.flatten();
            ratios = ratios.flatten();
            // Enumerate heights and widths from scales and ratios
            var heights = scales / np.sqrt(ratios);
            var widths = scales * np.sqrt(ratios);
            //Enumerate shifts in feature space
            var shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride;
            var shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride;
            (shifts_x, shifts_y) = np.meshgrid(shifts_x, shifts_y);
            // Enumerate combinations of shifts, widths, and heights
            var (box_widths, box_centers_x) = np.meshgrid(widths, shifts_x);
            var (box_heights, box_centers_y) = np.meshgrid(heights, shifts_y);
