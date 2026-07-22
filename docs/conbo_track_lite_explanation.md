# ConBo-Track Lite — 12-step method, steps 1–8 and 12

1. Open the video and read the selected base frame once.
2. Select one or more luminous targets at the original image resolution.
3. Extract a boundary-safe ROI for every target.
4. Use minimum-channel gamma correction (MVGC) and thresholding to isolate self-emitting pixels.
5. Apply morphological opening to remove small isolated responses.
6. Convolve the isolated field with a circular raised-cosine kernel to normalize target shape.
7. Detect the resulting circle or ellipse using EdgeDrawing; use geometric ellipse fitting only when EdgeDrawing returns no usable ellipse.
8. Update each target center and next-frame ROI from the detected geometry.
12. Draw the detected ellipse and coordinates, then save the tracking video and per-frame CSV.

The CSV contains original coordinates, displacement from the base frame, detected radius, validity, and detector method. The displayed ellipse comes from detection rather than the ROI size.
