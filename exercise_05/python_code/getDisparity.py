import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d

def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """
    left_img and right_img are both H x W and you should return a H x W matrix containing the disparity d for
    each pixel of left_img. Set disp_img to 0 for pixels where the SSD and/or d is not defined, and for
    d estimates rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy
    min_disp <= d <= max_disp.
    """

    h,w = left_img.shape
    result = np.zeros_like(left_img)

    
    for row in range(0+patch_radius,h-patch_radius):
        for col in range(max_disp + patch_radius, w-patch_radius):
            patch_0 = left_img[row-patch_radius:row+patch_radius+1,col-patch_radius:col+patch_radius+1]
            strip_1 = right_img[row-patch_radius:row+patch_radius+1, col-patch_radius-max_disp : col+patch_radius - min_disp+1]
            
            rsvecs = np.zeros([ 2 * patch_radius + 1,  2 * patch_radius + 1, max_disp - min_disp + 1])
            for i in range(0,  2 * patch_radius + 1):
                rsvecs[:, i, :] = strip_1[:, i:(max_disp - min_disp + i + 1)]

            # Transforming the patches into vectors so we can run them through pdist2.
            lpvec = patch_0.flatten()
            rsvecs = rsvecs.reshape([ (2 * patch_radius + 1)**2, max_disp - min_disp + 1])

            ssds = cdist(lpvec[None, :], rsvecs.T, 'sqeuclidean').squeeze(0)
            d = np.argmin(ssds)
            min_ssd = ssds[d]

            result[row,col]=d

            if (ssds <= 1.5 * min_ssd).sum() < 3 and d != 0 and d != ssds.shape[0] - 1:
       
                    x = np.asarray([d - 1, d, d + 1])
                    p = np.polyfit(x, ssds[x], 2)

                    # Minimum of p(0)x^2 + p(1)x + p(2), converted from neg_disp to disparity as above.
                    result[row, col] = max_disp + p[1] / (2 * p[0])


    return result   