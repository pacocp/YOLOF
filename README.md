# YOLOF

## YOLO meats Optical Flow

---

### 1. Introduction

Why to combine YOLO and Optical Flow?

- To compute only the optical flow of objects we look for.
- Reduce computational cost.

### 2. Algorithm

1. Object detection in the actual frame using YOLO.
2. Initialize a zero matrix with the same height and width as the frame.
3. For each object that has been detected:
   1. Extract a *patch* corresponding to the object's bounding box in the frame.
   2. Compute optical flow between that patch and its corresponding one in the previous frame.
   3. Insert the computed optical flow in the matrix.
4. Show the matrix that has been created.

### 3. Implementation

- Python 3.6.
- YOLO [1] and Farneback [2] implementations in OpenCV 3.4.0.
- Numpy and imutils.
- Weights of YOLO Lite [3].

All the requirements are in the *requirements.txt* file.

```
pip install -r requirements.txt
```

### Bibliography

[1] You only look once: Unified, real-time object detection. **J. Redmon, S. Divvala, R. Girshick, and A. Farhadi.**

[2] Two-frame motion estimation based on polynomial expansion. **G. Farnebäck.**

[3] Yolo-lite: A real-time object detection algorithm optimized for non-gpu computers. **R. Huang, J. Pedoeem, and C. Chen.**
