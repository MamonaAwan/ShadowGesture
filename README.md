# ShadowGesture

a vision-based shadow gesture recognition method for interactive projection system. Gesture recognition method is based on the screen image obtained the web camera installed in the same position as the projector. Recognition method separates only the shadow area by combining the binary image with learning algorithm isolated background from the input image. The region of interest is set by labeling the shadow of separated regions, and then it can isolate hand shadow using defect, convex-hull and moment of each region. To distinguish hand gestures, Hu’s invariant moment method is used. In addition, we have used the multiscale retinex algorithm to solve the problem that camera cannot recognize the gesture in bright place. Optical Flow algorithm is used for tracking the fingertip and OpenGL is used for representing the result of drawing.

Paper: https://www.researchgate.net/publication/328369130_A_Vision-based_Shadow_Gesture_Recognition_for_Interactive_Projection_System
