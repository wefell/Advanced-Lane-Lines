import cv2
import numpy as np
import pickle
from moviepy.editor import VideoFileClip

# read objpoints and imgpoints
dist_pickle = pickle.load(open('./calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


# threshold, window mask, and find window functions
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # Apply threshold
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    # Apply threshold
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def color_threshold(img, s_thresh=(0, 255), v_thresh=(0, 255)):
    # apply threshold on S channel in HLS colorspace
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # apply threshold on V channel in HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(s_channel > v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    # combine S & V thresholding
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1

    return binary_output


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    # draw window areas
    output[int(img_ref.shape[0] - (level + 1)
               * height):int(img_ref.shape[0] - level * height),
           max(0, int(center - width / 2)):min(int(center + width / 2),
                                               img_ref.shape[1])] = 1

    return output


def find_window_centroids(warped, window_width, window_height, margin):

    global previous_centroids

    window_centroids = []
    window = np.ones(window_width)

    # Get vertical slice from bottom quarter of image on left side
    l_sum = np.sum(warped[int(3 * warped.shape[0] /
                              4):, :int(warped.shape[1] / 2)], axis=0)
    # Convolve the left vertical slice with window
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    # Get vertical slice from bottom quarter of image on left side
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] /
                   2):], axis=0)
    # Convolve the left vertical slice with window
    r_center = (np.argmax(np.convolve(window, r_sum)) - window_width / 2 +
                int(warped.shape[1] / 2))

    # Start sanity check of l_center and r_center after 5 frames
    if len(previous_centroids) > 5:
        l_insanity = abs(l_center - previous_centroids[-1][0][0])
        r_insanity = abs(r_center - previous_centroids[-1][0][1])
        if l_insanity > 25:
            l_center = previous_centroids[-1][0][0]
        if r_insanity > 25:
            r_center = previous_centroids[-1][0][1]
    # Append fist two centroids
    window_centroids.append((l_center, r_center))

    # Find centroids for each level of windows
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window on vertical slice
        image_layer = np.sum(warped[int(warped.shape[0] - (level + 1) *
                             window_height):int(warped.shape[0] - level *
                                                window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find left centroid by using previous left center
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = (np.argmax(conv_signal[l_min_index:l_max_index]) +
                    l_min_index - offset)
        # Find right centroid using previous right center
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = (np.argmax(conv_signal[r_min_index:r_max_index]) +
                    r_min_index - offset)
        # Append centroids
        window_centroids.append((l_center, r_center))
    # Add centroids in current frame to previous centroids
    previous_centroids.append(window_centroids)
    # Average previous 15 frames with current frame
    window_centroids = np.average(previous_centroids[-12:], axis=0)

    return window_centroids


previous_centroids = []


def process_image(img):

    global previous_centroids

    # Undistort image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    ksize = 7
    # Apply thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize,
                             thresh=(10, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize,
                             thresh=(1, 255))
    color_binary = color_threshold(img, s_thresh=(100, 255),
                                   v_thresh=(50, 255))
    # Combine thresholding
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | (color_binary == 1)] = 255

    # Define perspective transform areas
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[594, 450], [223, 720], [1106, 720], [688, 450]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

    # Transform
    M = cv2.getPerspectiveTransform(src, dst)
    M_inverted = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)

    # window settings
    window_width = 50
    window_height = 180
    margin = 100

    # find window centroids
    window_centroids = find_window_centroids(warped, window_width,
                                             window_height, margin)

    # If centroids are found
    if len(window_centroids) > 0:

        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        leftx = []
        rightx = []

        # Draw windows around centroids
        for level in range(0, len(window_centroids)):
            l_mask = window_mask(window_width, window_height, warped,
                                 window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped,
                                 window_centroids[level][1], level)
            # Add center value found in frame to lists of lane points
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

            # Add graphic points from window mask to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Add left and right windows
        template = np.array(r_points + l_points, np.uint8)
        zero_channel = np.zeros_like(template)
        # Color windows green
        template = (np.array(cv2.merge((zero_channel, template, zero_channel)),
                    np.uint8))
        # Make original road pixels 3 color channels
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
        # Combine windows with original image
        output = cv2.addWeighted(warpage, 0.2, template, 0.2, 0.0)

    # Else display original image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Find y values of centers
    yvals = range(0, output.shape[0])
    res_yvals = np.arange(output.shape[0] - (window_height / 2.0), 0,
                          -window_height)

    # Fit second order polynomials to lanes
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = (right_fit[0] * yvals * yvals + right_fit[1] *
                  yvals + right_fit[2])
    right_fitx = np.array(right_fitx, np.int32)
    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 4,
                         left_fitx[::-1] + window_width / 4), axis=0),
                          np.concatenate((yvals, yvals[::-1]), axis=0))),
                         np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width /
                          4, right_fitx[::-1] + window_width / 4), axis=0),
                           np.concatenate((yvals, yvals[::-1]), axis=0))),
                          np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx - window_width /
                          4, right_fitx[::-1] + window_width / 4), axis=0),
                           np.concatenate((yvals, yvals[::-1]), axis=0))),
                          np.int32)

    # Color left, right, and center lanes
    road = np.zeros_like(img)
    cv2.fillPoly(road, [inner_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [left_lane], color=[0, 255, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 255, 0])

    # Transform image back
    road_warped = cv2.warpPerspective(road, M_inverted, img_size,
                                      flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1.0, road_warped, 0.25, 0.0)

    # Define meters per pixel in x and y
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    # Measure radius of curvature
    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                              np.array(leftx, np.float32) * xm_per_pix, 2)
    curvurad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix +
                curve_fit_cr[1]) ** 2) **
                1.5) / np.absolute(2 * curve_fit_cr[0])

    # Calculate offset of car
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    # Display radius of curvature and vehicle offset
    cv2.putText(result, 'Radius of Curvature = ' + str(round(curvurad, 3)) +
                '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm '
                + side_pos + ' of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result


Output_video = 'output1.mp4'
Input_video = 'project_video.mp4'

clip1 = VideoFileClip(Input_video, audio=False)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(Output_video, audio=False)
