# ConBo-Mag Lite — 12-step method, steps 1–12

ConBo-Mag runs the same steps 1–8 as ConBo-Track; it is not a separate preprocessing or target-management method.

9. At the initial mouse-selected ROI, convolve the isolated local intensity field with a Gaussian kernel to condition its spectral envelope. The same ROI coordinates are used in every frame.
10. Cache the base-frame phase and compute the wrapped phase difference to the current response.
11. Multiply that phase difference by the amplification factor, reconstruct the response with an inverse FFT, and detect its amplified ellipse.
12. Draw the amplified detected ellipse on the original video and save both video and CSV. The terminal reports the tracked position, amplified displacement, and displayed coordinate for each target and frame.

Magnification is intentionally confined to each fixed initial ROI. Tracking and MAG share target selection, preprocessing, response construction, and detection, but only Track recenters its ROI. The fixed MAG sampling window keeps ROI motion out of the phase difference. The demo visualizes amplified motion through detected geometry; it does not warp or reconstruct the full video frame.
