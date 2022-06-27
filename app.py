import streamlit as st
from PIL import Image
import cv2
import numpy as np
import colour
import pickle

import utils


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def load_model():
    with open('bcp0.model', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    st.sidebar.title("Filter Extraction Visualizer")
    st.subheader("Inspect intermediate images in the extraction pipeline")
    st.sidebar.image("RC_RGB_v1.2.4.jpg", use_column_width=True)

    st.sidebar.subheader("Options")
    use_sift = st.sidebar.checkbox("Apply Sift", True)
    apply_color_correction = st.sidebar.checkbox(
        "Apply Color Correction", True)

    st.sidebar.subheader("Display")
    show_original_image = st.sidebar.checkbox("Original Image", True)
    show_image_after_sift = st.sidebar.checkbox("Image After Sift", False)
    show_bw_image = st.sidebar.checkbox("B/W Image", False)
    show_extracted_boxes = st.sidebar.checkbox("Extracted Boxes", False)
    if apply_color_correction:
        show_image_after_cc = st.sidebar.checkbox(
            "Color Corrected Image", False
        )
    show_extracted_circles = st.sidebar.checkbox(
        "Extracted Filter Area", False)

    # Load up the reference image, the positions and colors of all 30 boxes
    (reference_image, reference_contours, ref_colors) = utils.load_image_with_features(
        "RC_RGB_v1.2.4.jpg"
    )
    assert len(reference_contours) == 30
    assert len(ref_colors) == 30

    file_up = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))

    if file_up:

        image = Image.open(file_up)
        original_input_image = utils.resize(convert_from_image_to_cv2(image))

        if show_original_image:
            st.caption("Original Image")
            st.image(image, use_column_width=True)

        input_image = (
            utils.run_sift(reference_image, original_input_image)
            if use_sift
            else original_input_image
        )

        if show_image_after_sift:
            st.caption("Image After SIFT")
            st.image(convert_from_cv2_to_image(
                input_image), use_column_width=True)

        input_image_grayscale = utils.convert_to_grayscale(input_image)

        # Convert grayscale image to B/W: low = 127, high = 255
        (input_thresh, input_threshold) = cv2.threshold(
            input_image_grayscale, 127, 255, cv2.THRESH_BINARY_INV
        )

        if show_bw_image:
            st.caption("B/W Image")
            st.image(convert_from_cv2_to_image(
                input_threshold), use_column_width=True)

        # Load up image, extract the positions and colors of all 30 boxes
        (target_image, target_contours, target_colors) = utils.extract_all_points(
            input_image, input_threshold
        )

        if show_extracted_boxes:
            target_image_with_boxes = utils.draw_contours(
                target_image.copy(), target_contours)
            st.caption("Extracted Boxes")
            st.image(convert_from_cv2_to_image(
                target_image_with_boxes), use_column_width=True)

        if len(reference_contours) != 30 or len(ref_colors) != 30:
            st.error(f"Could not extract all 30 boxes")

        # Store RGB values of extracted boxes
        rgbs = [np.array(list(reversed(bgrs))) for bgrs in target_colors]

        # """
        # Apply color correction
        # This technique ensures that the color correction process does not leave red artifacts
        # https://stackoverflow.com/questions/62993366/color-calibration-with-color-checker-using-using-root-polynomial-regression-not
        # """
        # Create a float copy
        color_corrected_image = input_image.astype(np.float)
        # Normalise the image to have pixel values from 0 to 1
        color_corrected_image = (color_corrected_image - np.min(color_corrected_image))/np.ptp(color_corrected_image)
        # Decode the image with sRGB EOTF
        color_corrected_image = colour.models.eotf_sRGB(color_corrected_image)
        if apply_color_correction:
            for row in color_corrected_image:
                row[:] = colour.colour_correction(row[:], target_colors, ref_colors, 'Finlayson 2015')
        
        # Encode image back to sRGB
        color_corrected_image = colour.models.eotf_inverse_sRGB(color_corrected_image)  
        # Denormalize image to fit 255 pixel values (also clip to ensure values fall between 0 - 255)
        color_corrected_image = np.clip((color_corrected_image * 255), 0, 255)
        # Convert floats back to integers
        color_corrected_image = color_corrected_image.astype(np.uint8)

        if apply_color_correction and show_image_after_cc:
            st.caption("Color Corrected Image")
            st.image(
                convert_from_cv2_to_image(color_corrected_image), use_column_width=True
            )

        # Extract filter
        filter_value, corrected_image_with_outline = utils.extract_filter(
            color_corrected_image, radius=24, show_circle=show_extracted_circles
        )

        if show_extracted_circles:
            st.caption("Extracted Filter Area")
            st.image(
                convert_from_cv2_to_image(corrected_image_with_outline),
                use_column_width=True,
            )

        model = load_model()

        BC = model.predict([[filter_value[2]]])

        st.markdown("### Extracted RGB Values")
        col1, col2, col3, col4 = st.columns(4)
        col1.error(f"R: {filter_value[2]:.3f}")
        col2.success(f"G: {filter_value[1]:.3f}")
        col3.info(f"B: {filter_value[0]:.3f}")
        col4.warning(f"BC: {BC[0]:.3f}")




if __name__ == "__main__":
    main()
