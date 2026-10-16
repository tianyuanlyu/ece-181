import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    from io import BytesIO
    from urllib.request import Request, urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return BytesIO, Image, Request, mo, np, plt, urlopen


@app.cell(hide_code=True)
def _(BytesIO, Image, Request, np, urlopen):
    def image_from_url(image_url):
        """Download image_url and return it as a grayscale NumPy array."""
        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return np.asarray(Image.open(BytesIO(response.read())).convert("L"), dtype=float)

    return (image_from_url,)


@app.cell(hide_code=True)
def _(image_from_url):
    mare_image = image_from_url(
        "https://bigdatahealth.ucsb.edu/sites/default/files/styles/logo/"
        "public/logo/img_2289.jpeg"
    )
    mare_side = min(mare_image.shape)
    mare_top = (mare_image.shape[0] - mare_side) // 2
    mare_left = (mare_image.shape[1] - mare_side) // 2
    mare_image = mare_image[
        mare_top : mare_top + mare_side, mare_left : mare_left + mare_side
    ]
    storke_tower_image = image_from_url(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/48/"
        "Ucsb-storke-tower01.jpg/960px-Ucsb-storke-tower01.jpg"
    )
    kavli_image = image_from_url(
        "https://www.kitp.ucsb.edu/sites/default/files/kitp/kavlisb.jpeg"
    )
    return kavli_image, mare_image, storke_tower_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 1

    ## Editing

    Marimo notebooks are just Python files, where each "cell" is a function. You can either edit here, in [Molab](https://molab.marimo.io/), or directly in Python.

    ### Option 1 - Molab [Recommended]

    This is the most recommended option. Take this notebook (.py file), upload it to Molab, and start working. Once you are finished making your edits, click "Export" and select "Python" to export it as a notebook source.

    ### Option 2 - Local Marimo

    To work locally, first create a python project and place this file `hw1.py` within it. Then, install marimo using your favorite package manager (`uv add marimo`). Finally, launch this notebook by typing in your terminal `uv run marimo edit` and a new tab should open in your browser.

    ### Option 3 - Regular Python [Not Recommended]

    It is theoretically possible to edit the `.py` file directly (although we don't recommend it). If you choose this route, make sure that your Python/Markdown renders properly in Marimo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: Image Convolution and Edge Detection

    You are given the following two images $I_1$ and $I_2$, with three convolution kernels $S_x$, $S_y$ and $L$. [.] indicates the origin of each image.

    \[
    I_1 = \begin{bmatrix}
    [1] & 1 & 1 \\
    3 & 3 & 3 \\
    6 & 6 & 6
    \end{bmatrix}
    \qquad
    I_2 = \begin{bmatrix}
    [10] & 10 & 10 \\
    10 & 100 & 100 \\
    10 & 100 & 100
    \end{bmatrix}
    \]

    \[
    S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & [0] & 2 \\ -1 & 0 & 1\end{bmatrix}
    \qquad
    S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & [0] & 0 \\ 1 & 2 & 1\end{bmatrix}
    \qquad
    L = \begin{bmatrix}0 & 1 & 0 \\ 1 & [-4] & 1 \\ 0 & 1 & 0\end{bmatrix}
    \]
    """)
    return


@app.cell
def _(np):
    # EDIT BELOW: transcribe the matrices displayed above into NumPy arrays.
    # Replace the zero defaults with the instructor-provided entries for I_1 and I_2.
    image_1 = np.array([
        [1,1,1],
        [3,3,3],
        [6,6,6]
    ])
    image_2 = np.array([
        [10,10,10],
        [10,100,100],
        [10,100,100]
    ])

    sobel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    sobel_y = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]])
    laplacian = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])
    print(image_1)
    print(image_2)
    print(sobel_x)
    print(sobel_y)
    print(laplacian)
    return image_1, image_2, laplacian, sobel_x, sobel_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(a) Sobel X and Sobel Y By Hand

    Compute the full linear convolutions \(S_x * I_1\) and \(S_y * I_1\) **by hand**, where \(*\) denotes linear convolution. Mark the origin in each output matrix. Use the two responses to locate the edges, and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(a)


    Given:

    $$
    I_1 =
    \begin{bmatrix}
    1 & 1 & 1\\
    3 & 3 & 3\\
    6 & 6 & 6
    \end{bmatrix}
    $$

    $$
    S_x =
    \begin{bmatrix}
    -1 & 0 & 1\\
    -2 & 0 & 2\\
    -1 & 0 & 1
    \end{bmatrix},
    \qquad
    S_y =
    \begin{bmatrix}
    -1 & -2 & -1\\
    0 & 0 & 0\\
    1 & 2 & 1
    \end{bmatrix}
    $$

    **Full Convolution Output Size**

    $$
    (3+3-1)\times(3+3-1)=5\times5
    $$
    $$
    I_1 =
    \begin{bmatrix}
    0 & 0 & 0 & 0 & 0\\
    0 & 1 & 1 & 1 & 0\\
    0 & 3 & 3 & 3 & 0\\
    0 & 6 & 6 & 6 & 0\\
    0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    **Sobel X convolution:**

    $$
    S_x * I_1 =
    \begin{bmatrix}
    -1 & -1 & 0 & 1 & 1\\
    -5 & \boxed{-5} & 0 & 5 & 5\\
    -13 & -13 & 0 & 13 & 13\\
    -15 & -15 & 0 & 15 & 15\\
    -6 & -6 & 0 & 6 & 6
    \end{bmatrix}
    $$

    **Sobel Y convolution:**

    $$
    S_y * I_1 =
    \begin{bmatrix}
    -1 & -3 & -4 & -3 & -1\\
    -3 & \boxed{-9} & -12 & -9 & -3\\
    -5 & -15 & -20 & -15 & -5\\
    3 & 9 & 12 & 9 & 3\\
    6 & 18 & 24 & 18 & 6
    \end{bmatrix}
    $$


    **Edge detection:**
    The pixel values stay the same across each row but increase from top to bottom, especially from 3 to 6. This means the image has horizontal edges. Sobel Y detects these changes, while Sobel X gives zero in the central column. The nonzero Sobel X values on the sides are mainly caused by zero padding.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(b) Laplacian By Hand

    Compute the full linear convolution \(L * I_2\) **by hand**. Mark the origin in the output matrix. Use the response to locate any corner(s), and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(b)


    ### 1(b) Laplacian By Hand

    Given:

    $$
    I_2=
    \begin{bmatrix}
    10&10&10\\
    10&100&100\\
    10&100&100
    \end{bmatrix}
    $$

    $$
    L=
    \begin{bmatrix}
    0&1&0\\
    1&-4&1\\
    0&1&0
    \end{bmatrix}
    $$

    The Laplacian kernel is symmetric, so it remains the same after flipping.

    The full convolution output size is:

    $$
    (3+3-1)\times(3+3-1)=5\times5
    $$


    For the center element:

    $$
    \begin{aligned}
    O[2,2]
    &=10+10+100+100-4(100)\\
    &=220-400\\
    &=-180
    \end{aligned}
    $$

    *Full convolution result:*

    $$
    L*I_2=
    \begin{bmatrix}
    0&10&10&10&0\\
    10&\boxed{-20}&80&70&10\\
    10&80&-180&-190&100\\
    10&70&-190&-200&100\\
    0&10&100&100&0
    \end{bmatrix}
    $$

    **Origin:** The boxed value marks the origin at (1,1).

    **Corner Detection:**

    The main corner is at (1,1) in the original image. At this point, the pixel intensity changes from 10 to 100 in both horizontal and vertical directions. The Laplacian shows strong responses near these changes, which helps identify the corner region. However, the Laplacian also responds to edges, and some large values are caused by zero padding.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(c) Verify in Python

    Complete the two code cells below. The function must perform full 2D linear convolution without calling a library convolution routine (you may not use any numpy functions).
    """)
    return


@app.cell
def _(np):
    def convolve2d_full(image, kernel):
        """Return the full 2D linear convolution of image and kernel."""
        output_shape = (
            image.shape[0] + kernel.shape[0] - 1,
            image.shape[1] + kernel.shape[1] - 1,
        )
        output = np.empty(output_shape, dtype=float)

        # Get the number of rows and columns
        rows = image.shape[0]
        cols = image.shape[1]
        k_rows = kernel.shape[0]
        k_cols = kernel.shape[1]

        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                output[i, j] = 0

        # Calculate convolution
        for i in range(rows):
            for j in range(cols):
                for m in range(k_rows):
                    for n in range(k_cols):
                        output[i+m, j+n] += image[i,j] * kernel[m,n]

        return output


    return (convolve2d_full,)


@app.cell(hide_code=True)
def _(convolve2d_full, image_1, image_2, laplacian, sobel_x, sobel_y):
    try:
        convolve2d_full(image_1, sobel_x)
        convolve2d_full(image_1, sobel_y)
        convolve2d_full(image_2, laplacian)
    except Exception as error:
        print(f"Problem 1(c) test is not ready: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Your Convolution on a Real Image

    The next cell downloads a photograph, converts it to grayscale, and runs your convolution function with both Sobel kernels. It displays the input, the horizontal and vertical responses, and their combined edge magnitude. You can use this as a quick check after implementing `convolve2d_full`.
    """)
    return


@app.cell(hide_code=True)
def _(convolve2d_full, mare_image, np, sobel_x, sobel_y):
    try:
        mare_sobel_x = convolve2d_full(mare_image, sobel_x)
        mare_sobel_y = convolve2d_full(mare_image, sobel_y)
        mare_edges = np.hypot(mare_sobel_x, mare_sobel_y)
        mare_error = None
    except Exception as error:
        mare_sobel_x = mare_sobel_y = mare_edges = None
        mare_error = str(error)
    return mare_edges, mare_error, mare_sobel_x, mare_sobel_y


@app.cell(hide_code=True)
def _(mare_edges, mare_error, mare_image, mare_sobel_x, mare_sobel_y, mo):
    try:
        if mare_error:
            mo.output.replace(mo.md(f"Sobel test on mare is not ready: {mare_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack(
                        [mo.image(mare_image, width=220), mo.image(mare_sobel_x, width=220)]
                    ),
                    mo.hstack(
                        [mo.image(mare_sobel_y, width=220), mo.image(mare_edges, width=220)]
                    ),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the Sobel test: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Gaussian and Laplacian Pyramids

    You are given a color image. Convert it to grayscale, then crop the 512×512 region beginning at the upper-left corner. Use this crop as the input image for both parts.

    You will build pyramid levels at 512×512, 256×256, 128×128, and 32×32. To reach 32×32 from 128×128, blur and downsample twice. Do not use a prebuilt pyramid implementation.

    A normalized 5×5 Gaussian kernel is provided in the starter code. In your report, state your boundary convention, show the requested pyramid images at their corresponding resolutions, and include your code. If you use any non-standard-library helper, include it with a brief explanation.
    """)
    return


@app.cell
def _(storke_tower_image):
    # EDIT BELOW: take the upper-left crop from storke_tower_image.
    input_image = storke_tower_image[:512, :512]
    return (input_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(a) Gaussian Pyramid

    Create a Gaussian pyramid from the 512×512 grayscale crop using Gaussian convolution and the provided normalized 5×5 kernel. Apply blur and downsampling to produce four levels, in this order: 512×512, 256×256, 128×128, and 32×32. Document your boundary convention, and include the four images and your code in the report. Do not use a prebuilt pyramid implementation.
    """)
    return


@app.cell
def blur_and_downsample(convolve2d_full):
    def blur_and_downsample(image, kernel):
        """Blur image with kernel, then reduce each spatial dimension by two."""

        # Apply Gaussian blur
        full_blurred_image = convolve2d_full(image, kernel)

        image_height = image.shape[0]
        image_width = image.shape[1]

        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]

        crop_start_row = (kernel_height - 1) // 2
        crop_start_col = (kernel_width - 1) // 2


        blurred_image = full_blurred_image[
            crop_start_row:crop_start_row + image_height,
            crop_start_col:crop_start_col + image_width
        ]

        return blurred_image[::2, ::2]

    return (blur_and_downsample,)


@app.cell
def _(blur_and_downsample, np):
    def build_gaussian_pyramid(
        image, level_shapes=((512, 512), (256, 256), (128, 128), (32, 32))
    ):
        """Return Gaussian-pyramid levels with exactly the requested shapes."""
        kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256
        pyramid = [image]

        # EDIT BELOW: call blur_and_downsample until each requested level is reached.
        # Keep levels in the same order as level_shapes.


        current_image = image

        # Build Gaussian pyramid
        for target_shape in level_shapes[1:]:

            while current_image.shape != target_shape:
                current_image = blur_and_downsample(current_image, kernel)

            pyramid.append(current_image)

        return pyramid


 


    return (build_gaussian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b) Laplacian Pyramid

    Using your Gaussian pyramid, construct the corresponding Laplacian pyramid. Construct the Laplacian pyramid for each of the four resolutions given above. For example, to construct the Laplacian pyramid at resolution 512x512, you take the Gaussian smoothed image at resolution 512x512 and subtract it from the original 512x512 image.
    """)
    return


@app.cell
def build_laplacian_pyramid(convolve2d_full, np):
    def build_laplacian_pyramid(gaussian_pyramid):
        """Return Laplacian levels followed by the coarsest Gaussian residual."""
        pyramid = []

        # EDIT BELOW: construct the Laplacian level at each requested resolution,
        # then append the final coarsest Gaussian level.
    

        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=float) / 256


        def expand_image(small_image):
            image_height, image_width = small_image.shape

            expanded_image = np.zeros(
                (image_height * 2, image_width * 2),
                dtype=float
            )

            expanded_image[::2, ::2] = small_image

            # Apply Gaussian smoothing
            full_blurred_image = convolve2d_full(
                expanded_image, kernel * 4
            )

            smoothed_image = full_blurred_image[
                2:2 + image_height * 2,
                2:2 + image_width * 2
            ]

            return smoothed_image


        pyramid = []

        for level_index in range(len(gaussian_pyramid) - 1):

            current_image = gaussian_pyramid[level_index]
            smaller_image = gaussian_pyramid[level_index + 1]

            expanded_image = smaller_image

            while expanded_image.shape != current_image.shape:
                expanded_image = expand_image(expanded_image)

            laplacian_image = current_image - expanded_image

            pyramid.append(laplacian_image)

        # Step 3: Keep the smallest Gaussian image
        pyramid.append(gaussian_pyramid[-1])

        return pyramid


    return (build_laplacian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pyramid Visualizations

    After completing your functions, the hidden cells below display the Gaussian pyramid in the first row and the Laplacian pyramid in the second row.
    """)
    return


@app.cell(hide_code=True)
def _(build_gaussian_pyramid, build_laplacian_pyramid, input_image):
    try:
        gaussian_pyramid = build_gaussian_pyramid(input_image)
        laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
        pyramid_error = None
    except Exception as error:
        gaussian_pyramid = laplacian_pyramid = None
        pyramid_error = str(error)
    return gaussian_pyramid, laplacian_pyramid, pyramid_error


@app.cell(hide_code=True)
def _(gaussian_pyramid, laplacian_pyramid, mo, pyramid_error):
    try:
        if pyramid_error:
            mo.output.replace(mo.md(f"Pyramid visualization is not ready: {pyramid_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack([mo.image(level, width=150) for level in gaussian_pyramid]),
                    mo.hstack([mo.image(level, width=150) for level in laplacian_pyramid]),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the pyramids: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Harris Corner Detector

    For this part, implement a Harris corner detector. At each image location, compute the corner response function (CRF)

    \[
    R(x, y) = \det(A) - k\bigl(\operatorname{trace}(A)\bigr)^2,
    \]

    where \(A\) is the local structure matrix

    \[
    A =
    \begin{bmatrix}
    \langle I_x^2 \rangle & \langle I_x I_y \rangle \\
    \langle I_x I_y \rangle & \langle I_y^2 \rangle
    \end{bmatrix}.
    \]

    Use a simple constant 3×3 averaging window for the local averages. You may use any discrete derivative approximation, but write down the equations you use. You may ignore image-border effects by zero-padding or by limiting calculations to the image interior.

    Use the CRF to identify the top \(N\) candidate corners, where \(N\) is around 20–50. Threshold and sort the candidates before selecting the final locations. Your function should return `(row, column)` locations.

    Put your complete implementation, including any helper functions, in the cell below. You may use functions defined elsewhere in this notebook. Use the KAVLI image to demonstrate your detector: include your code in the report, overlay the detected corners on the image, list their coordinates, state your parameter values, and comment on the results.
    """)
    return


@app.cell
def harris_corners(np):
    def harris_corners(image, threshold, max_corners=40, k=0.04):
        """Return up to max_corners Harris corners as (row, column) locations."""
        # EDIT BELOW: write your complete detector and any helpers here.
    
        image = image.astype(float)

        #image gradients
        gradient_y, gradient_x = np.gradient(image)

        # gradient products
        gradient_x_squared = gradient_x ** 2
        gradient_y_squared = gradient_y ** 2
        gradient_xy = gradient_x * gradient_y

        # Calculate the average in a 3x3 window
        def average_3x3(values):
            padded_values = np.pad(values, 1, mode='constant')
            averages = np.zeros_like(values)

            for row_offset in range(3):
                for col_offset in range(3):
                    averages += padded_values[
                        row_offset:row_offset + values.shape[0],
                        col_offset:col_offset + values.shape[1]
                    ]

            return averages / 9

        avg_x_squared = average_3x3(gradient_x_squared)
        avg_y_squared = average_3x3(gradient_y_squared)
        avg_xy = average_3x3(gradient_xy)

        # Harris response
        determinant = avg_x_squared * avg_y_squared - avg_xy ** 2
        trace = avg_x_squared + avg_y_squared

        response = determinant - k * trace ** 2

        # sort strong corner candidates
        minimum_response = threshold * response.max()

        candidate_points = np.argwhere(
            (response > minimum_response) & (response > 0)
        )

        candidate_points = sorted(
            candidate_points,
            key=lambda point: response[point[0], point[1]],
            reverse=True
        )

        # Select corners that are not too close
        selected_corners = []

        for point in candidate_points:
            too_close = False

            for chosen_corner in selected_corners:
                row_difference = point[0] - chosen_corner[0]
                col_difference = point[1] - chosen_corner[1]

                distance_squared = row_difference**2 + col_difference**2

                if distance_squared <= 9:
                    too_close = True
                    break

            if not too_close:
                selected_corners.append(point)

            if len(selected_corners) >= max_corners:
                break

        return np.array(selected_corners, dtype=int).reshape(-1, 2)



    return (harris_corners,)


@app.cell
def _(harris_corners, kavli_image, plt):
    # EDIT BELOW: choose a threshold, then plot the detected corners.
    corners = harris_corners(kavli_image, threshold=0.01)
    plt.imshow(kavli_image, cmap="gray")
    plt.axis("off")
    plt.scatter(corners[:, 1], corners[:, 0], c="r", s=10)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 4: Philosophy of Computer Vision

    ## 4(a) Biological Facial Recognition

    How do humans recognize faces? Describe the features the visual system could use and the processing that might occur at each stage, from seeing a face to identifying the person.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(a)

    First of all, our eyes receive light reflected from a person's face, and the light is refracted to form an image on the retina. Then, the visual information is sent to the brain for processing. The brain detects basic features like edges and shapes and identifies facial features like the eyes, nose, and mouth, as well as where they are positioned. Finally, the brain compares these features with faces stored in our memory to recognize who the person is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(b) Computer Vision in the Age of AI

    As AI becomes better at writing code and optimizing pipelines, what should students learn in a computer vision course?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(b)

    I think students should learn how to use computers to quickly recognize objects and extract useful information from images. More importantly, we should understand the mathematical principles and logic behind the recognition process, instead of just learning how to write code. With this knowledge, we can use AI more effectively to improve our code, solve problems, and check whether the results are correct.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(c) Course Milestone

    What would you like to accomplish by the end of this course? Describe a project, a skill you want to develop, or another goal you hope to reach.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(c)
    At the end of this course, I hope to independently develop a computer vision program that can analyze the surface features of plants, such as leaf color, texture, and shape. I want to use this information to identify whether a plant might need more water or changes in humidity and other growing conditions. I think this would be an interesting way to apply what I learn in this course to real-life problems.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submitting Your Work

    If you used Molab, click **Export** and choose **Python** to download your completed notebook as a `.py` file. Submit that `.py` notebook file to Gradescope.
    """)
    return


if __name__ == "__main__":
    app.run()
